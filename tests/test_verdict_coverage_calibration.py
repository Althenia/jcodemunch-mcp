"""Coverage contract + planted-query calibration (v1.108.145).

Two halves of the same honesty upgrade:

1. Coverage contract: an ``absent`` verdict disclosing scan counts alone can
   lie by omission when files were excluded at index time (unsupported
   extensions, oversize, binary, secret, cap-dropped) or parsed to zero
   symbols. The index now persists a coverage block from every full discovery
   walk, and absent/degraded verdicts carry it.

2. Planted-query calibration: the 0-1 confidence surface is a heuristic, so
   its meaning must be pinned to a scorer version and measured against planted
   positive/negative queries. ``benchmarks/calibration/planted_queries.json``
   records the measured rates; this suite RE-RUNS the measurement live so the
   artifact can never drift silently from the code.

Planted names are slug-unique on purpose (fixture-decay guard): they must
never appear in real corpora, so a hit/miss is unambiguous.
"""

import json
from pathlib import Path

from tests.conftest_helpers import _write

ARTIFACT = (
    Path(__file__).resolve().parent.parent
    / "benchmarks" / "calibration" / "planted_queries.json"
)

# Slug-unique planted symbols (positives) — indexed into the corpus below.
PLANTED_POSITIVE = [
    "qorvex_flux_router",
    "zimral_packet_weaver",
    "drovny_ledger_sync",
    "makelvin_shard_probe",
]
# Slug-unique plausible-but-absent names (negatives) — NEVER in the corpus.
PLANTED_NEGATIVE = [
    "velcron_token_mixer",
    "ostrafel_cache_warden",
    "jubrant_queue_stapler",
    "wexolim_frame_dicer",
]


def _planted_corpus(tmp_path: Path) -> tuple[str, str]:
    """Index a corpus containing the planted positives + coverage trip-wires."""
    from jcodemunch_mcp.tools.index_folder import index_folder

    body = "\n\n".join(
        f"def {name}(payload):\n    '''Planted calibration symbol.'''\n    return payload"
        for name in PLANTED_POSITIVE
    )
    _write(tmp_path / "planted.py", body + "\n")
    # Coverage trip-wires: an unsupported extension + a zero-symbol source file.
    _write(tmp_path / "blob.qzx", "not a source language\n")
    _write(tmp_path / "notes.py", "# commentary only, no symbols\n")
    sp = str(tmp_path / "idx")
    result = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=sp)
    assert result["success"]
    return result["repo"], sp


def _load_index(repo: str, storage_path: str):
    from jcodemunch_mcp.storage.index_store import IndexStore

    store = IndexStore(base_path=storage_path)
    owner, name = repo.split("/", 1)
    return store.load_index(owner, name)


class TestCoverageContract:
    def test_coverage_persisted_on_full_walk(self, tmp_path: Path):
        repo, sp = _planted_corpus(tmp_path)
        index = _load_index(repo, sp)
        cov = getattr(index, "coverage", {})
        assert cov, "coverage contract missing after full-walk index"
        assert cov["walk"] == "full"
        assert cov["files_indexed"] >= 2  # planted.py + notes.py
        assert cov["skip_counts"].get("wrong_extension", 0) >= 1
        assert cov["no_symbols_count"] >= 1  # notes.py
        assert cov["recorded_at"]

    def test_absent_verdict_discloses_coverage(self, tmp_path: Path):
        from jcodemunch_mcp.tools.search_symbols import search_symbols

        repo, sp = _planted_corpus(tmp_path)
        result = search_symbols(
            repo=repo, query=PLANTED_NEGATIVE[0], storage_path=sp,
        )
        verdict = result["_meta"]["verdict"]
        assert verdict["state"] == "absent"
        cov = verdict.get("coverage")
        assert cov, "absent verdict must disclose coverage"
        assert cov["excluded"].get("wrong_extension", 0) >= 1
        assert cov["no_symbols_files"] >= 1
        assert cov["generation"]["indexed_at"]
        assert cov["generation"]["index_version"]

    def test_ok_verdict_stays_lean(self, tmp_path: Path):
        from jcodemunch_mcp.tools.search_symbols import search_symbols

        repo, sp = _planted_corpus(tmp_path)
        result = search_symbols(
            repo=repo, query=PLANTED_POSITIVE[0], storage_path=sp,
        )
        verdict = result["_meta"]["verdict"]
        assert verdict["state"] == "ok"
        assert "coverage" not in verdict

    def test_unknown_coverage_omitted_not_fabricated(self):
        """A pre-upgrade index (empty coverage) yields no coverage block."""
        from jcodemunch_mcp.retrieval.verdict import index_coverage_meta

        class _Legacy:
            coverage: dict = {}

        assert index_coverage_meta(_Legacy()) is None

    def test_symbol_verdict_wrapper_attaches_coverage(self, tmp_path: Path):
        from jcodemunch_mcp.retrieval.verdict import symbol_verdict_for_index

        repo, sp = _planted_corpus(tmp_path)
        index = _load_index(repo, sp)
        verdict = symbol_verdict_for_index(
            index, found_count=0, requested_id="nope::nothing#function",
        )
        assert verdict["state"] == "absent"
        assert "coverage" in verdict

    def test_scorer_version_pinned_on_verdict(self, tmp_path: Path):
        from jcodemunch_mcp.retrieval.verdict import SCORER_VERSION
        from jcodemunch_mcp.tools.search_symbols import search_symbols

        repo, sp = _planted_corpus(tmp_path)
        result = search_symbols(
            repo=repo, query=PLANTED_POSITIVE[0], storage_path=sp,
        )
        assert result["_meta"]["verdict"]["scorer"] == SCORER_VERSION


class TestPlantedQueryCalibration:
    def _measure(self, tmp_path: Path) -> dict:
        from jcodemunch_mcp.tools.search_symbols import search_symbols

        repo, sp = _planted_corpus(tmp_path)
        pos_states, pos_scores, neg_states = [], [], []
        for q in PLANTED_POSITIVE:
            r = search_symbols(repo=repo, query=q, storage_path=sp)
            v = r["_meta"]["verdict"]
            pos_states.append(v["state"])
            pos_scores.append(v["best_score"] or 0.0)
        for q in PLANTED_NEGATIVE:
            r = search_symbols(repo=repo, query=q, storage_path=sp)
            neg_states.append(r["_meta"]["verdict"]["state"])
        return {
            "positive_hit_rate": pos_states.count("ok") / len(pos_states),
            "negative_absent_rate": neg_states.count("absent") / len(neg_states),
            "min_positive_best_score": min(pos_scores),
        }

    def test_planted_queries_separate_cleanly(self, tmp_path: Path):
        m = self._measure(tmp_path)
        assert m["positive_hit_rate"] == 1.0
        assert m["negative_absent_rate"] == 1.0
        assert m["min_positive_best_score"] > 0

    def test_artifact_matches_live_measurement(self, tmp_path: Path):
        from jcodemunch_mcp.retrieval.verdict import SCORER_VERSION

        assert ARTIFACT.exists(), f"missing calibration artifact: {ARTIFACT}"
        artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        assert artifact["scorer_version"] == SCORER_VERSION, (
            "scorer changed without re-measuring calibration: bump the artifact"
        )
        m = self._measure(tmp_path)
        assert artifact["planted_positive"]["hit_rate"] == m["positive_hit_rate"]
        assert artifact["planted_negative"]["absent_rate"] == m["negative_absent_rate"]
        assert artifact["planted_positive"]["n"] == len(PLANTED_POSITIVE)
        assert artifact["planted_negative"]["n"] == len(PLANTED_NEGATIVE)
