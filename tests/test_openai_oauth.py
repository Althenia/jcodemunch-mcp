"""OpenAI ChatGPT subscription OAuth behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from jcodemunch_mcp import openai_oauth
from jcodemunch_mcp.parser import Symbol


def test_authorize_url_uses_pkce_state_and_loopback_callback():
    url = openai_oauth.build_authorize_url(
        "http://127.0.0.1:1455/auth/callback", "challenge", "csrf-state"
    )

    assert "response_type=code" in url
    assert "code_challenge=challenge" in url
    assert "code_challenge_method=S256" in url
    assert "state=csrf-state" in url
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A1455%2Fauth%2Fcallback" in url


def test_exchange_code_returns_sanitized_credential(monkeypatch):
    response = MagicMock()
    response.json.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_in": 3600,
        "id_token": "header.eyJjaGF0Z3B0X2FjY291bnRfaWQiOiJhY2N0In0.signature",
    }
    response.raise_for_status.return_value = None
    monkeypatch.setattr(openai_oauth.httpx, "post", lambda *args, **kwargs: response)

    credential = openai_oauth.exchange_code(
        "code", "http://127.0.0.1:1455/auth/callback", "verifier"
    )

    assert credential.access_token == "access"
    assert credential.refresh_token == "refresh"
    assert credential.account_id == "acct"
    assert credential.expires_at > 0


def test_exchange_code_rejects_missing_refresh_token(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"access_token": "access"}
    response.raise_for_status.return_value = None
    monkeypatch.setattr(openai_oauth.httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(ValueError, match="refresh token"):
        openai_oauth.exchange_code(
            "code", "http://127.0.0.1:1455/auth/callback", "verifier"
        )


def test_headless_authorization_polls_then_stores_credential(monkeypatch):
    device = MagicMock()
    device.json.return_value = {
        "device_auth_id": "device-id",
        "user_code": "USER-CODE",
        "interval": "1",
    }
    device.raise_for_status.return_value = None
    pending = MagicMock(status_code=403)
    complete = MagicMock(status_code=200)
    complete.json.return_value = {
        "authorization_code": "code",
        "code_verifier": "verifier",
    }
    responses = iter([device, pending, complete])
    stored: list[openai_oauth.OpenAIOAuthCredential] = []
    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access", refresh_token="refresh", expires_at=9999999999
    )

    monkeypatch.setattr(openai_oauth.httpx, "post", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(openai_oauth, "exchange_code", lambda *args: credential)
    monkeypatch.setattr(openai_oauth, "store_credential", stored.append)
    monkeypatch.setattr(openai_oauth.time, "sleep", lambda seconds: None)

    assert openai_oauth.authorize_headless(timeout_seconds=1) == credential
    assert stored == [credential]


def test_load_credential_refreshes_expired_token_and_persists(monkeypatch):
    stored = json.dumps(
        {
            "access_token": "expired",
            "refresh_token": "refresh",
            "expires_at": 0,
            "account_id": "acct",
        }
    )
    persisted: list[str] = []
    monkeypatch.setattr(openai_oauth.credentials, "keyring_get", lambda name: stored)
    monkeypatch.setattr(
        openai_oauth,
        "refresh_credential",
        lambda credential: openai_oauth.OpenAIOAuthCredential(
            access_token="fresh",
            refresh_token="new-refresh",
            expires_at=9999999999,
            account_id=credential.account_id,
        ),
    )
    monkeypatch.setattr(
        openai_oauth.credentials, "keyring_set", lambda name, value: persisted.append(value)
    )

    credential = openai_oauth.load_credential()

    assert credential is not None
    assert credential.access_token == "fresh"
    assert json.loads(persisted[0])["refresh_token"] == "new-refresh"


def test_load_credential_serializes_refresh_token_rotation(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    stored = {
        "value": json.dumps(
            {
                "access_token": "expired",
                "refresh_token": "refresh",
                "expires_at": 0,
                "account_id": "acct",
            }
        )
    }
    fresh = openai_oauth.OpenAIOAuthCredential(
        access_token="fresh",
        refresh_token="new-refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    first_get = Event()
    second_get = Event()
    release_first = Event()
    count_lock = Lock()
    get_count = 0
    refresh_count = 0

    def fake_get(name):
        nonlocal get_count
        with count_lock:
            get_count += 1
            current = get_count
        if current == 1:
            first_get.set()
            release_first.wait(timeout=1)
        else:
            second_get.set()
        return stored["value"]

    def fake_refresh(credential):
        nonlocal refresh_count
        refresh_count += 1
        return fresh

    monkeypatch.setattr(openai_oauth.credentials, "keyring_get", fake_get)
    monkeypatch.setattr(
        openai_oauth.credentials,
        "keyring_set",
        lambda name, value: stored.update(value=value),
    )
    monkeypatch.setattr(openai_oauth, "refresh_credential", fake_refresh)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(openai_oauth.load_credential)
        assert first_get.wait(timeout=1)
        second = executor.submit(openai_oauth.load_credential)
        second_entered_while_first_loading = second_get.wait(timeout=0.1)
        release_first.set()
        credentials = [first.result(timeout=1), second.result(timeout=1)]

    assert not second_entered_while_first_loading
    assert refresh_count == 1
    assert [credential.access_token for credential in credentials] == ["fresh", "fresh"]


def test_load_credential_rejects_malformed_keyring_value(monkeypatch):
    monkeypatch.setattr(openai_oauth.credentials, "keyring_get", lambda name: "not-json")

    with pytest.raises(ValueError, match="invalid OpenAI OAuth credential"):
        openai_oauth.load_credential()


def test_openai_provider_uses_codex_responses_and_account_header(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import _create_summarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    monkeypatch.setenv("JCODEMUNCH_SUMMARIZER_PROVIDER", "openai")
    monkeypatch.setattr(openai_oauth, "load_credential", lambda: credential)

    summarizer = _create_summarizer()

    assert summarizer is not None
    assert summarizer.api_base == openai_oauth.CODEX_API_BASE
    assert summarizer.wire_api == "responses"
    assert summarizer.client.headers["ChatGPT-Account-Id"] == "acct"


def test_openai_codex_request_omits_temperature_for_reasoning_model(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import _create_summarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    monkeypatch.setenv("JCODEMUNCH_SUMMARIZER_PROVIDER", "openai")
    monkeypatch.setattr(openai_oauth, "load_credential", lambda: credential)

    summarizer = _create_summarizer()
    assert summarizer is not None

    _, payload = summarizer._request_spec("Summarize this symbol.")

    assert payload["model"] == "gpt-5.4-mini"
    assert "temperature" not in payload


def test_openai_codex_request_uses_backend_contract(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import _create_summarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    monkeypatch.setenv("JCODEMUNCH_SUMMARIZER_PROVIDER", "openai")
    monkeypatch.setattr(openai_oauth, "load_credential", lambda: credential)

    summarizer = _create_summarizer()
    assert summarizer is not None

    path, payload = summarizer._request_spec("Summarize this symbol.")

    assert path == "/responses"
    assert payload == {
        "model": "gpt-5.4-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Summarize this symbol."}
                ],
            }
        ],
        "store": False,
        "stream": True,
    }


def test_openai_codex_request_preserves_required_fields(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import _create_summarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    monkeypatch.setenv("JCODEMUNCH_SUMMARIZER_PROVIDER", "openai")
    monkeypatch.setattr(openai_oauth, "load_credential", lambda: credential)

    summarizer = _create_summarizer()
    assert summarizer is not None
    summarizer.extra_body = {
        "input": "invalid",
        "store": True,
        "stream": False,
        "reasoning": {"effort": "low"},
    }

    _, payload = summarizer._request_spec("Summarize this symbol.")

    assert payload["input"][0]["content"][0]["text"] == "Summarize this symbol."
    assert payload["store"] is False
    assert payload["stream"] is True
    assert payload["reasoning"] == {"effort": "low"}


def test_openai_codex_summarizer_reads_sse_output():
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    response = MagicMock()
    response.iter_lines.return_value = [
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","delta":"1. Summarizes "}',
        "event: response.output_text.delta",
        'data:{"type":"response.output_text.delta","delta":"a symbol."}',
        "event: response.completed",
        'data: {"type":"response.completed","response":{}}',
        "data: [DONE]",
    ]
    client = MagicMock()
    client.post.return_value = response

    with patch.object(OpenAIBatchSummarizer, "_init_client"):
        summarizer = OpenAIBatchSummarizer(oauth_credential=credential)
    summarizer.client = client
    symbol = Symbol(
        id="test::thing",
        file="test.py",
        name="thing",
        qualified_name="thing",
        kind="function",
        language="python",
        signature="def thing():",
    )

    summarizer.summarize_batch([symbol])

    assert symbol.summary == "Summarizes a symbol."


def test_openai_codex_refreshes_expired_credential_before_request(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    expired = openai_oauth.OpenAIOAuthCredential(
        access_token="expired-access",
        refresh_token="old-refresh",
        expires_at=0,
        account_id="acct",
    )
    rotated = openai_oauth.OpenAIOAuthCredential(
        access_token="rotated-access",
        refresh_token="rotated-refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    monkeypatch.setattr(openai_oauth, "load_credential", lambda: rotated)
    response = MagicMock()
    response.iter_lines.return_value = [
        'data: {"type":"response.output_text.delta","delta":"1. Refreshed."}',
        "data: [DONE]",
    ]
    client = MagicMock()
    client.post.return_value = response

    with patch.object(OpenAIBatchSummarizer, "_init_client"):
        summarizer = OpenAIBatchSummarizer(oauth_credential=expired)
    summarizer.client = client
    symbol = Symbol(
        id="test::thing",
        file="test.py",
        name="thing",
        qualified_name="thing",
        kind="function",
        language="python",
        signature="def thing():",
    )

    summarizer.summarize_batch([symbol])

    headers = client.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer rotated-access"
    assert headers["ChatGPT-Account-Id"] == "acct"
    assert summarizer.oauth_credential is rotated


def test_openai_provider_preserves_api_key_fallback_when_keyring_is_unavailable(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import _create_summarizer

    monkeypatch.setenv("JCODEMUNCH_SUMMARIZER_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "api-key")
    monkeypatch.setattr(
        openai_oauth,
        "load_credential",
        lambda: (_ for _ in ()).throw(ImportError("keyring unavailable")),
    )

    summarizer = _create_summarizer()

    assert summarizer is not None
    assert summarizer.api_base == "http://localhost:11434/v1"
    assert summarizer.client.headers["Authorization"] == "Bearer api-key"
