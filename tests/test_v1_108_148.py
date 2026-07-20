"""v1.108.148 — estimate-vs-actual consumption receipts.

plan_turn opens a token estimate for its recommended route; the next
plan_turn closes it against response tokens actually served in between.
After 3 closed samples the median actual/estimated ratio surfaces as
`estimate_calibration` in session stats, `actual_vs_estimated` on the
budget block, and calibrated figures on plan_turn's consumption_estimate.
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


def _close_samples(state, pairs):
    """Drive (estimated, actual) pairs through the open/close cycle."""
    for estimated, actual in pairs:
        state.record_turn_estimate(estimated)
        state.record_response_tokens(actual)
    return state.record_turn_estimate(0)


class TestCalibrationCore:
    def test_no_calibration_below_sample_floor(self, fresh_state):
        cal = _close_samples(fresh_state, [(100, 200), (100, 300)])
        assert cal is None

    def test_median_ratio_at_floor(self, fresh_state):
        cal = _close_samples(fresh_state, [(100, 200), (100, 300), (100, 400)])
        assert cal == {"samples": 3, "actual_vs_estimated": 3.0}

    def test_median_is_outlier_robust(self, fresh_state):
        # One 30x runaway turn must not swamp the signal.
        cal = _close_samples(
            fresh_state, [(100, 100), (100, 110), (100, 90), (100, 3000)]
        )
        assert cal["samples"] == 4
        assert cal["actual_vs_estimated"] < 2.0

    def test_zero_estimate_never_produces_sample(self, fresh_state):
        fresh_state.record_turn_estimate(0)
        fresh_state.record_response_tokens(500)
        cal = _close_samples(
            fresh_state, [(100, 100), (100, 100), (100, 100)]
        )
        assert cal["samples"] == 3

    def test_zero_actual_never_produces_sample(self, fresh_state):
        for _ in range(4):
            fresh_state.record_turn_estimate(100)
        assert fresh_state.record_turn_estimate(100) is None

    def test_avg_response_tokens_per_call(self, fresh_state):
        assert token_tracker.avg_response_tokens_per_call() == 0
        fresh_state.record_response_tokens(300)
        fresh_state.record_response_tokens(100)
        assert token_tracker.avg_response_tokens_per_call() == 200


class TestSurfaces:
    def test_stats_omit_calibration_without_samples(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        stats = fresh_state.session_stats(str(tmp_path))
        assert "estimate_calibration" not in stats

    def test_stats_carry_calibration_block(self, fresh_state, monkeypatch, tmp_path):
        _set_budget(monkeypatch, 0)
        _close_samples(fresh_state, [(100, 200), (100, 200), (100, 200)])
        stats = fresh_state.session_stats(str(tmp_path))
        assert stats["estimate_calibration"] == {"samples": 3, "actual_vs_estimated": 2.0}

    def test_budget_block_carries_ratio(self, fresh_state, monkeypatch):
        _set_budget(monkeypatch, 100000)
        _close_samples(fresh_state, [(100, 200), (100, 200), (100, 200)])
        b = token_tracker.budget_status()
        assert b["actual_vs_estimated"] == 2.0

    def test_budget_block_lean_without_samples(self, fresh_state, monkeypatch):
        _set_budget(monkeypatch, 1000)
        assert "actual_vs_estimated" not in token_tracker.budget_status()


class TestPlanTurnWiring:
    def _plan(self, tmp_path, query="handler"):
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.plan_turn import plan_turn

        src = tmp_path / "proj"
        src.mkdir(exist_ok=True)
        (src / "handlers.py").write_text(
            "def handler_alpha():\n    return 1\n\n\ndef handler_beta():\n    return 2\n"
        )
        store = tmp_path / "store"
        index_folder(str(src), storage_path=str(store), use_ai_summaries=False)
        return plan_turn("proj", query, storage_path=str(store))

    def test_plan_turn_emits_consumption_estimate(self, fresh_state, tmp_path):
        result = self._plan(tmp_path)
        ce = result["consumption_estimate"]
        assert ce["expected_calls"] > 0
        assert ce["estimated_tokens"] == ce["expected_calls"] * token_tracker._DEFAULT_TOKENS_PER_CALL
        assert ce["basis"] == "default"
        assert "actual_vs_estimated" not in ce

    def test_plan_turn_uses_session_avg_basis(self, fresh_state, tmp_path):
        fresh_state.record_response_tokens(500)
        result = self._plan(tmp_path)
        ce = result["consumption_estimate"]
        assert ce["basis"] == "session_avg"

    def test_plan_turn_calibrated_after_three_turns(self, fresh_state, tmp_path):
        _close_samples(fresh_state, [(100, 200), (100, 200), (100, 200)])
        result = self._plan(tmp_path)
        ce = result["consumption_estimate"]
        assert ce["actual_vs_estimated"] == 2.0
        assert ce["calibrated_tokens"] == int(ce["estimated_tokens"] * 2.0)

    def test_plan_turn_opens_estimate_in_tracker(self, fresh_state, tmp_path):
        self._plan(tmp_path)
        assert fresh_state._open_estimate is not None
        assert fresh_state._open_estimate["estimated"] > 0
