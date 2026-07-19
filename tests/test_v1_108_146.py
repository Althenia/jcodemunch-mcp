"""v1.108.146 — advisory session token budget + token-yield session stats.

Budget: `session_token_budget` config over response tokens served;
{limit, spent, state} with state ok/approaching/over; advisory only.
Yield: `yield` block in session stats from served/fetched/edited signals
plus repeated identical calls. Both live in storage/token_tracker.py.
"""

import pytest

from jcodemunch_mcp.storage import token_tracker
from jcodemunch_mcp.storage.token_tracker import _State


@pytest.fixture()
def fresh_state(monkeypatch):
    state = _State()
    monkeypatch.setattr(token_tracker, "_state", state)
    return state


def _set_budget(monkeypatch, limit):
    real_get = token_tracker._config.get

    def fake_get(key, default=None, **kwargs):
        if key == "session_token_budget":
            return limit
        return real_get(key, default, **kwargs)

    monkeypatch.setattr(token_tracker._config, "get", fake_get)


class TestSessionBudget:
    def test_unconfigured_budget_absent_from_stats(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        stats = fresh_state.session_stats(str(tmp_path))
        assert "budget" not in stats
        assert token_tracker.budget_status() is None

    def test_configured_budget_always_in_stats(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 1000)
        stats = fresh_state.session_stats(str(tmp_path))
        assert stats["budget"] == {"limit": 1000, "spent": 0, "state": "ok"}

    def test_state_edges_79_80_100(self, fresh_state, monkeypatch):
        _set_budget(monkeypatch, 1000)
        fresh_state.record_response_tokens(799)
        assert token_tracker.budget_status()["state"] == "ok"
        fresh_state.record_response_tokens(1)  # 800 = 80%
        assert token_tracker.budget_status()["state"] == "approaching"
        fresh_state.record_response_tokens(200)  # 1000 = 100%
        b = token_tracker.budget_status()
        assert b["state"] == "over"
        assert b["spent"] == 1000

    def test_record_response_text_uses_4_bytes_per_token(self, fresh_state, monkeypatch):
        _set_budget(monkeypatch, 100)
        total = token_tracker.record_response_text("x" * 400)
        assert total == 100
        assert token_tracker.budget_status()["state"] == "over"

    def test_session_response_tokens_in_stats(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        fresh_state.record_response_tokens(123)
        stats = fresh_state.session_stats(str(tmp_path))
        assert stats["session_response_tokens"] == 123

    def test_negative_and_garbage_limits_disable(self, fresh_state, monkeypatch):
        _set_budget(monkeypatch, -5)
        assert token_tracker.budget_status() is None


class TestSessionYield:
    def test_yield_absent_when_nothing_served(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        stats = fresh_state.session_stats(str(tmp_path))
        assert "yield" not in stats

    def test_fetch_through_marks_served(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        token_tracker.note_served(["a.py::Foo#class", "b.py::bar#function"])
        token_tracker.note_fetched(["a.py::Foo#class"])
        stats = fresh_state.session_stats(str(tmp_path))
        y = stats["yield"]
        assert y["served_results"] == 2
        assert y["followed_through"] == 1
        assert y["rate"] == 0.5

    def test_fetch_of_unserved_id_is_noop(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        token_tracker.note_served(["a.py::Foo#class"])
        token_tracker.note_fetched(["never/served.py::x#function"])
        stats = fresh_state.session_stats(str(tmp_path))
        assert stats["yield"]["followed_through"] == 0

    def test_edit_through_marks_symbols_in_file(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        token_tracker.note_served(
            ["src/mod.py::Foo#class", "src/mod.py::bar#function", "other.py::baz#function"]
        )
        token_tracker.note_edited_files(["src/mod.py"])
        stats = fresh_state.session_stats(str(tmp_path))
        y = stats["yield"]
        assert y["followed_through"] == 2
        assert y["rate"] == round(2 / 3, 3)

    def test_edit_through_windows_paths(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        token_tracker.note_served(["src/mod.py::Foo#class"])
        token_tracker.note_edited_files(["src\\mod.py"])
        stats = fresh_state.session_stats(str(tmp_path))
        assert stats["yield"]["followed_through"] == 1

    def test_repeated_identical_calls_counted(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        token_tracker.note_served(["a.py::Foo#class"])  # so the block appears
        token_tracker.note_call_signature("search_symbols", "abc123")
        token_tracker.note_call_signature("search_symbols", "abc123")
        token_tracker.note_call_signature("search_symbols", "abc123")
        token_tracker.note_call_signature("search_symbols", "different")
        stats = fresh_state.session_stats(str(tmp_path))
        assert stats["yield"]["repeated_identical_calls"] == {"search_symbols": 2}

    def test_no_repeats_omits_key(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        token_tracker.note_served(["a.py::Foo#class"])
        token_tracker.note_call_signature("search_symbols", "only-once")
        stats = fresh_state.session_stats(str(tmp_path))
        assert "repeated_identical_calls" not in stats["yield"]

    def test_served_cap_evicts_oldest(self, fresh_state, monkeypatch):
        cap = token_tracker._YIELD_SERVED_MAXSIZE
        token_tracker.note_served([f"f{i}.py::s#function" for i in range(cap + 10)])
        assert len(fresh_state._yield_served) == cap
        assert "f0.py::s#function" not in fresh_state._yield_served


class TestServerWiring:
    @pytest.mark.asyncio
    async def test_get_session_stats_carries_budget_block(self, fresh_state, monkeypatch):
        import json
        from jcodemunch_mcp import config as cfg
        from jcodemunch_mcp.server import call_tool

        monkeypatch.setitem(cfg._GLOBAL_CONFIG, "session_token_budget", 1000)
        res = await call_tool("get_session_stats", {"format": "json"})
        body = json.loads(res[0].text)
        assert body["budget"]["limit"] == 1000
        assert body["budget"]["state"] in ("ok", "approaching", "over")

    @pytest.mark.asyncio
    async def test_meta_budget_attached_when_approaching(self, fresh_state, monkeypatch):
        import json
        from jcodemunch_mcp import config as cfg
        from jcodemunch_mcp.server import call_tool

        monkeypatch.setitem(cfg._GLOBAL_CONFIG, "session_token_budget", 1000)
        fresh_state.record_response_tokens(900)
        res = await call_tool("get_session_stats", {"format": "json"})
        body = json.loads(res[0].text)
        assert body["_meta"]["budget"]["state"] == "approaching"
        assert body["_meta"]["budget"]["spent"] == 900

    @pytest.mark.asyncio
    async def test_responses_accumulate_into_spent(self, fresh_state, monkeypatch):
        from jcodemunch_mcp import config as cfg
        from jcodemunch_mcp.server import call_tool

        monkeypatch.setitem(cfg._GLOBAL_CONFIG, "session_token_budget", 10)
        assert fresh_state._session_response_tokens == 0
        await call_tool("get_session_stats", {"format": "json"})
        assert fresh_state._session_response_tokens > 0


class TestConfigKey:
    def test_session_token_budget_registered(self):
        from jcodemunch_mcp import config as cfg
        assert cfg.DEFAULTS["session_token_budget"] == 0
        assert cfg.CONFIG_TYPES["session_token_budget"] is int
        assert "session_token_budget" in cfg.generate_template()
