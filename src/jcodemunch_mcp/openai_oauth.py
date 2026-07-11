"""OpenAI ChatGPT subscription OAuth authentication and credential storage."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from queue import Empty, Queue
import secrets
from threading import Thread
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import httpx

from . import credentials

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ISSUER = "https://auth.openai.com"
CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"
CREDENTIAL_NAME = "OPENAI_OAUTH"
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
OAUTH_SCOPE = "openid profile email offline_access"
EXPIRY_SAFETY_SECONDS = 60
DEVICE_AUTH_HEADERS = {"User-Agent": "jcodemunch-mcp"}


@dataclass(frozen=True)
class OpenAIOAuthCredential:
    """A persisted OAuth credential for a ChatGPT subscription."""

    access_token: str
    refresh_token: str
    expires_at: float
    account_id: str | None = None

    def is_expired(self) -> bool:
        """Return whether the access token needs refreshing."""
        return self.expires_at <= time.time() + EXPIRY_SAFETY_SECONDS


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Generate a PKCE verifier and its S256 challenge."""
    verifier = _base64url(secrets.token_bytes(64))
    challenge = _base64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(redirect_uri: str, challenge: str, state: str) -> str:
    """Build the OpenAI authorization URL for the browser OAuth flow."""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": "jcodemunch-mcp",
    }
    return f"{ISSUER}/oauth/authorize?{urlencode(params)}"


def _extract_account_id(token: str | None) -> str | None:
    if not token or token.count(".") != 2:
        return None
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return None
    account_id = claims.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    auth_claim = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claim, dict) and isinstance(auth_claim.get("chatgpt_account_id"), str):
        return auth_claim["chatgpt_account_id"]
    organizations = claims.get("organizations")
    if isinstance(organizations, list) and organizations:
        first = organizations[0]
        if isinstance(first, dict) and isinstance(first.get("id"), str):
            return first["id"]
    return None


def _credential_from_response(data: dict[str, Any]) -> OpenAIOAuthCredential:
    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        raise ValueError("OpenAI token response did not contain an access token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise ValueError("OpenAI token response did not contain a refresh token")
    expires_in = data.get("expires_in", 3600)
    if not isinstance(expires_in, (int, float)) or expires_in <= 0:
        raise ValueError("OpenAI token response contained an invalid expiry")
    return OpenAIOAuthCredential(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + float(expires_in),
        account_id=_extract_account_id(data.get("id_token"))
        or _extract_account_id(access_token),
    )


def exchange_code(code: str, redirect_uri: str, verifier: str) -> OpenAIOAuthCredential:
    """Exchange an authorization code for an OAuth credential."""
    response = httpx.post(
        f"{ISSUER}/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": CLIENT_ID,
            "code_verifier": verifier,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return _credential_from_response(response.json())


def refresh_credential(credential: OpenAIOAuthCredential) -> OpenAIOAuthCredential:
    """Refresh an expired OAuth credential, preserving its known account ID."""
    response = httpx.post(
        f"{ISSUER}/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": credential.refresh_token,
            "client_id": CLIENT_ID,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    refreshed = _credential_from_response(response.json())
    if refreshed.account_id is None:
        return OpenAIOAuthCredential(
            access_token=refreshed.access_token,
            refresh_token=refreshed.refresh_token,
            expires_at=refreshed.expires_at,
            account_id=credential.account_id,
        )
    return refreshed


def store_credential(credential: OpenAIOAuthCredential) -> None:
    """Persist an OAuth credential in the existing system keyring."""
    credentials.keyring_set(CREDENTIAL_NAME, json.dumps(asdict(credential)))


def load_credential() -> OpenAIOAuthCredential | None:
    """Load and, when necessary, refresh the stored OAuth credential."""
    raw = credentials.keyring_get(CREDENTIAL_NAME)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("credential must be a JSON object")
        access_token = value["access_token"]
        refresh_token = value["refresh_token"]
        account_id = value.get("account_id")
        if not isinstance(access_token, str) or not isinstance(refresh_token, str):
            raise ValueError("credential tokens must be strings")
        if account_id is not None and not isinstance(account_id, str):
            raise ValueError("credential account ID must be a string")
        credential = OpenAIOAuthCredential(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=float(value["expires_at"]),
            account_id=account_id,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("invalid OpenAI OAuth credential in system keyring") from error
    if credential.is_expired():
        credential = refresh_credential(credential)
        store_credential(credential)
    return credential


def _callback_server(
    expected_state: str, result: Queue[tuple[str | None, str | None]]
) -> type[BaseHTTPRequestHandler]:
    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_error(404, "Not found")
                return
            params = parse_qs(parsed.query)
            error = params.get("error_description", params.get("error", [None]))[0]
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            if error:
                result.put((None, error))
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"OpenAI authorization failed. You may close this window.")
                return
            if not code or state != expected_state:
                result.put((None, "Missing authorization code or invalid OAuth state"))
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"OpenAI authorization failed. You may close this window.")
                return
            result.put((code, None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OpenAI authorization complete. You may close this window.")

        def log_message(self, format: str, *args: object) -> None:
            return

    return CallbackHandler


def authorize_browser(timeout_seconds: float = 300.0) -> OpenAIOAuthCredential:
    """Authorize through a browser loopback callback and persist the result."""
    verifier, challenge = generate_pkce()
    state = _base64url(secrets.token_bytes(32))
    redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
    result: Queue[tuple[str | None, str | None]] = Queue(maxsize=1)
    server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), _callback_server(state, result))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        if not webbrowser.open(build_authorize_url(redirect_uri, challenge, state)):
            raise RuntimeError("could not open a browser; retry with --headless")
        try:
            code, error = result.get(timeout=timeout_seconds)
        except Empty as exc:
            raise TimeoutError("OpenAI authorization timed out") from exc
        if error:
            raise RuntimeError(f"OpenAI authorization failed: {error}")
        if code is None:
            raise RuntimeError("OpenAI authorization returned no code")
        credential = exchange_code(code, redirect_uri, verifier)
        store_credential(credential)
        return credential
    finally:
        server.shutdown()
        server.server_close()


def authorize_headless(timeout_seconds: float = 300.0) -> OpenAIOAuthCredential:
    """Authorize through the OpenAI device flow and persist the result."""
    device_response = httpx.post(
        f"{ISSUER}/api/accounts/deviceauth/usercode",
        json={"client_id": CLIENT_ID},
        headers=DEVICE_AUTH_HEADERS,
        timeout=30.0,
    )
    device_response.raise_for_status()
    device = device_response.json()
    device_id = device.get("device_auth_id")
    user_code = device.get("user_code")
    if not isinstance(device_id, str) or not isinstance(user_code, str):
        raise ValueError("OpenAI device authorization response was incomplete")
    interval = max(float(device.get("interval", 5)), 1.0)
    print(f"Open {ISSUER}/codex/device and enter code: {user_code}")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = httpx.post(
            f"{ISSUER}/api/accounts/deviceauth/token",
            json={"device_auth_id": device_id, "user_code": user_code},
            headers=DEVICE_AUTH_HEADERS,
            timeout=30.0,
        )
        if response.status_code == 200:
            data = response.json()
            code = data.get("authorization_code")
            verifier = data.get("code_verifier")
            if not isinstance(code, str) or not isinstance(verifier, str):
                raise ValueError("OpenAI device token response was incomplete")
            credential = exchange_code(code, f"{ISSUER}/deviceauth/callback", verifier)
            store_credential(credential)
            return credential
        if response.status_code not in (403, 404):
            response.raise_for_status()
        time.sleep(interval)
    raise TimeoutError("OpenAI device authorization timed out")
