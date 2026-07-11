"""OpenAI ChatGPT subscription OAuth behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from jcodemunch_mcp import openai_oauth


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
