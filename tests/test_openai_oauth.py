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


def test_openai_codex_retries_429_then_succeeds():
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    request = openai_oauth.httpx.Request("POST", f"{openai_oauth.CODEX_API_BASE}/responses")
    rate_limited = openai_oauth.httpx.Response(
        429, headers={"Retry-After": "0"}, request=request
    )
    success = MagicMock(status_code=200)
    success.iter_lines.return_value = [
        'data: {"type":"response.output_text.delta","delta":"1. Retried."}',
        "data: [DONE]",
    ]
    client = MagicMock()
    client.post.side_effect = [rate_limited, success]

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

    with patch("time.sleep") as sleep:
        summarizer.summarize_batch([symbol])

    assert symbol.summary == "Retried."
    assert client.post.call_count == 2
    sleep.assert_called_once_with(0.0)
    assert summarizer._consecutive_failures == 0


def test_openai_codex_rechecks_token_before_retry(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    initial = openai_oauth.OpenAIOAuthCredential(
        access_token="initial-access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    expired = openai_oauth.OpenAIOAuthCredential(
        access_token="expired-access",
        refresh_token="refresh",
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
    request = openai_oauth.httpx.Request("POST", f"{openai_oauth.CODEX_API_BASE}/responses")
    rate_limited = openai_oauth.httpx.Response(
        429, headers={"Retry-After": "0"}, request=request
    )
    success = MagicMock(status_code=200)
    success.iter_lines.return_value = [
        'data: {"type":"response.output_text.delta","delta":"1. Rotated."}',
        "data: [DONE]",
    ]

    with patch.object(OpenAIBatchSummarizer, "_init_client"):
        summarizer = OpenAIBatchSummarizer(oauth_credential=initial)
    authorization_headers = []

    def post(url, json, headers):
        authorization_headers.append(headers["Authorization"])
        if len(authorization_headers) == 1:
            summarizer.oauth_credential = expired
            return rate_limited
        return success

    summarizer.client = MagicMock()
    summarizer.client.post.side_effect = post
    symbol = Symbol(
        id="test::thing",
        file="test.py",
        name="thing",
        qualified_name="thing",
        kind="function",
        language="python",
        signature="def thing():",
    )

    with patch("time.sleep"):
        summarizer.summarize_batch([symbol])

    assert authorization_headers == ["Bearer initial-access", "Bearer rotated-access"]
    assert symbol.summary == "Rotated."


def test_openai_codex_stops_after_429_retries():
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    request = openai_oauth.httpx.Request("POST", f"{openai_oauth.CODEX_API_BASE}/responses")
    rate_limited = openai_oauth.httpx.Response(
        429, headers={"Retry-After": "0"}, request=request
    )
    client = MagicMock()
    client.post.side_effect = [rate_limited] * 4

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

    with patch("time.sleep") as sleep:
        summarizer.summarize_batch([symbol])

    assert symbol.summary == "def thing():"
    assert client.post.call_count == 4
    assert sleep.call_count == 3
    assert summarizer._consecutive_failures == 1


def test_openai_codex_rate_limit_delay():
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    summarizer = OpenAIBatchSummarizer()
    capped = MagicMock(headers={"Retry-After": "120"})
    exponential = MagicMock(headers={})

    assert summarizer._rate_limit_delay(capped, 0) == 60.0
    with patch(
        "jcodemunch_mcp.summarizer.batch_summarize.random.uniform",
        return_value=1.0,
    ):
        assert [summarizer._rate_limit_delay(exponential, attempt) for attempt in range(3)] == [
            1.0,
            2.0,
            4.0,
        ]


@pytest.mark.parametrize("retry_after", ["invalid", "-1", "nan", "inf"])
def test_openai_codex_invalid_retry_after_uses_backoff(retry_after):
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    summarizer = OpenAIBatchSummarizer()
    response = MagicMock(headers={"Retry-After": retry_after})

    with patch(
        "jcodemunch_mcp.summarizer.batch_summarize.random.uniform",
        return_value=1.0,
    ):
        assert summarizer._rate_limit_delay(response, 1) == 2.0


def test_openai_codex_defaults_to_one_worker(monkeypatch):
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    monkeypatch.delenv("OPENAI_CONCURRENCY", raising=False)
    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    with patch.object(OpenAIBatchSummarizer, "_init_client"):
        summarizer = OpenAIBatchSummarizer(oauth_credential=credential)

    assert summarizer._worker_count() == 1


def test_openai_concurrency_override_is_preserved():
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    with patch.dict("os.environ", {"OPENAI_CONCURRENCY": "2"}, clear=True), patch.object(
        OpenAIBatchSummarizer, "_init_client"
    ):
        summarizer = OpenAIBatchSummarizer(oauth_credential=credential)
        assert summarizer._worker_count() == 2


def test_openai_api_key_keeps_configured_workers():
    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    with patch.dict("os.environ", {}, clear=True), patch(
        "jcodemunch_mcp.summarizer.batch_summarize._config.get", return_value=4
    ):
        summarizer = OpenAIBatchSummarizer()
        assert summarizer._worker_count() == 4


def test_openai_codex_burst_is_serialized_by_default(monkeypatch):
    import time
    from threading import Lock

    from jcodemunch_mcp.summarizer.batch_summarize import OpenAIBatchSummarizer

    monkeypatch.delenv("OPENAI_CONCURRENCY", raising=False)
    credential = openai_oauth.OpenAIOAuthCredential(
        access_token="access",
        refresh_token="refresh",
        expires_at=9999999999,
        account_id="acct",
    )
    with patch.object(OpenAIBatchSummarizer, "_init_client"):
        summarizer = OpenAIBatchSummarizer(oauth_credential=credential)
    summarizer.client = object()
    lock = Lock()
    active = 0
    peak = 0

    def run_batch(batch):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        for symbol in batch:
            symbol.summary = "Done."

    summarizer._run_batch = run_batch
    symbols = [
        Symbol(
            id=f"test::thing_{index}",
            file="test.py",
            name=f"thing_{index}",
            qualified_name=f"thing_{index}",
            kind="function",
            language="python",
            signature=f"def thing_{index}():",
        )
        for index in range(8)
    ]

    summarizer.summarize_batch(symbols, batch_size=1)

    assert peak == 1


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
