"""Stateless-MCP forward cover (v1.108.150): auth-principal session keying.

The MCP 2026-07-28 spec removes protocol sessions. _session_key() gains an
auth-principal fallback between the transport session_id and the weakref UUID:
it fires only when the streamable-http handler captured a principal AND the
session exposes no session_id — i.e. exactly the future stateless shape.
"""

from jcodemunch_mcp import server as server_mod


class _FakeRequestContext:
    def __init__(self, session):
        self.session = session


class _FakeServer:
    def __init__(self, request_context):
        self.request_context = request_context


class _Session:
    pass


class _SessionWithId:
    session_id = "transport-session-id"


def _install_session(monkeypatch, session_obj):
    monkeypatch.setattr(
        server_mod, "server", _FakeServer(_FakeRequestContext(session_obj))
    )


class TestPrincipalDerivation:
    def test_no_header_is_none(self):
        assert server_mod._principal_from_authorization(None) is None
        assert server_mod._principal_from_authorization("") is None

    def test_stable_and_prefixed(self):
        a = server_mod._principal_from_authorization("Bearer tok-1")
        b = server_mod._principal_from_authorization("Bearer tok-1")
        assert a == b
        assert a.startswith("principal-")

    def test_distinct_credentials_distinct_keys(self):
        a = server_mod._principal_from_authorization("Bearer tok-1")
        b = server_mod._principal_from_authorization("Bearer tok-2")
        assert a != b

    def test_never_contains_raw_credential(self):
        key = server_mod._principal_from_authorization("Bearer super-secret-token")
        assert "super-secret-token" not in key


class TestSessionKeyPrincipalFallback:
    def setup_method(self):
        server_mod._reset_session_tiers()

    def teardown_method(self):
        server_mod._reset_session_tiers()

    def test_session_id_wins_over_principal(self, monkeypatch):
        _install_session(monkeypatch, _SessionWithId())
        token = server_mod._HTTP_PRINCIPAL.set("principal-abc")
        try:
            assert server_mod._session_key() == "transport-session-id"
        finally:
            server_mod._HTTP_PRINCIPAL.reset(token)

    def test_principal_fires_when_session_lacks_id(self, monkeypatch):
        _install_session(monkeypatch, _Session())
        token = server_mod._HTTP_PRINCIPAL.set("principal-abc")
        try:
            assert server_mod._session_key() == "principal-abc"
        finally:
            server_mod._HTTP_PRINCIPAL.reset(token)

    def test_principal_fires_with_no_session_at_all(self, monkeypatch):
        # The fully stateless shape: no request context, principal captured.
        monkeypatch.setattr(server_mod, "server", _FakeServer(None))
        token = server_mod._HTTP_PRINCIPAL.set("principal-abc")
        try:
            assert server_mod._session_key() == "principal-abc"
        finally:
            server_mod._HTTP_PRINCIPAL.reset(token)

    def test_no_principal_preserves_weakref_uuid_path(self, monkeypatch):
        session = _Session()
        _install_session(monkeypatch, session)
        key1 = server_mod._session_key()
        key2 = server_mod._session_key()
        assert key1 == key2
        assert key1 != server_mod._SESSION_TIER_DEFAULT_KEY
        assert not str(key1).startswith("principal-")

    def test_no_session_no_principal_is_default_sentinel(self, monkeypatch):
        monkeypatch.setattr(server_mod, "server", _FakeServer(None))
        assert server_mod._session_key() == server_mod._SESSION_TIER_DEFAULT_KEY


class TestNoPrincipalDemandLog:
    def test_logs_once_per_process(self, monkeypatch, caplog):
        monkeypatch.setattr(server_mod, "_no_principal_logged", False)
        import logging

        with caplog.at_level(logging.INFO, logger=server_mod.logger.name):
            server_mod._note_no_principal_session()
            server_mod._note_no_principal_session()
        hits = [r for r in caplog.records if "no Authorization header" in r.message]
        assert len(hits) == 1
