"""v1.108.102 — audit W6: the dead ``identity_boost`` learned parameter is gone.

The tuner used to learn an ``identity_boost`` weight from the ranking ledger and
persist it to ``tuning.jsonc``, but nothing ever read it back at query time (there
was no ``get_identity_boost`` consumer the way there is ``get_semantic_weight``).
It was write-only dead state. These tests pin the removal: the tuner no longer
proposes, persists, or reports an identity weight, while ``semantic_weight``
learning and the ``identity_hit`` telemetry column are untouched.
"""

from __future__ import annotations

import inspect

from jcodemunch_mcp.retrieval import tuning as _tuning
from jcodemunch_mcp.storage import token_tracker as tt


def _enable(monkeypatch):
    from jcodemunch_mcp import config as _config
    real_get = _config.get

    def patched_get(key, default=None, *args, **kwargs):
        if key == "perf_telemetry_enabled":
            return True
        return real_get(key, default, *args, **kwargs)

    monkeypatch.setattr(_config, "get", patched_get)


def _reset(monkeypatch, tmp_path):
    fresh = tt._State()
    fresh._base_path = str(tmp_path)
    monkeypatch.setattr(tt, "_state", fresh)
    monkeypatch.setattr(_tuning, "_cache", {})
    monkeypatch.setattr(_tuning, "_cache_loaded_from", None)


def test_identity_boost_symbols_removed():
    # The constants that only fed the dead parameter are gone.
    assert not hasattr(_tuning, "_DEFAULT_IDENTITY_BOOST")
    assert not hasattr(_tuning, "_IDENTITY_BOUNDS")
    # There is (still) no query-time consumer.
    assert not hasattr(_tuning, "get_identity_boost")


def test_propose_returns_two_tuple():
    # _propose now returns (new_semantic_weight, signals) — no identity slot.
    tuner = _tuning.WeightTuner()
    new_sem, signals = tuner._propose([], {})
    assert new_sem is None
    assert signals == {}
    assert len(inspect.signature(tuner._propose).parameters) == 2  # events, existing


def test_identity_correlated_confidence_does_not_persist_a_weight(monkeypatch, tmp_path):
    # Every event has identity_hit=True with high confidence and the opposite
    # semantic split is flat, so the ONLY correlated signal is identity. The old
    # tuner would have written identity_boost; the new one must find no signal.
    _reset(monkeypatch, tmp_path)
    _enable(monkeypatch)
    for _ in range(30):
        tt.record_ranking_event(
            tool="search_symbols", repo="local/r", query="x", returned_ids=[],
            confidence=0.9, semantic_used=True, identity_hit=True,
        )
    for _ in range(30):
        tt.record_ranking_event(
            tool="search_symbols", repo="local/r", query="y", returned_ids=[],
            confidence=0.5, semantic_used=False, identity_hit=False,
        )
    # semantic_used and identity_hit are perfectly collinear here, so the tuner
    # attributes the confidence lift to semantic_weight (the only learned knob).
    tuner = _tuning.WeightTuner(base_path=str(tmp_path))
    result = tuner.learn("local/r", min_events=50)
    assert "identity_boost" not in result["after"]
    assert "identity_step" not in result.get("signals", {})
    assert "mean_confidence_identity_on" not in result.get("signals", {})


def test_semantic_learning_still_works(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path)
    _enable(monkeypatch)
    for _ in range(30):
        tt.record_ranking_event(
            tool="search_symbols", repo="local/r", query="x", returned_ids=[],
            confidence=0.9, semantic_used=True, identity_hit=False,
        )
    for _ in range(30):
        tt.record_ranking_event(
            tool="search_symbols", repo="local/r", query="y", returned_ids=[],
            confidence=0.5, semantic_used=False, identity_hit=False,
        )
    tuner = _tuning.WeightTuner(base_path=str(tmp_path))
    result = tuner.learn("local/r", min_events=50)
    assert result["applied"] is True
    assert result["after"]["semantic_weight"] == 0.55
    assert "identity_boost" not in result["after"]
