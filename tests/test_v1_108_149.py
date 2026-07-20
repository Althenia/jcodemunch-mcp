"""v1.108.149 — deterministic emission (cache-stable context P1).

Every context-serving sort now carries a total-order tiebreak (symbol id /
file path) so tied float scores can never fall back to index storage order,
which shuffles across rebuilds. Ordering becomes a function of content and
scores alone: two indexes built from the same tree in different discovery
orders serve byte-identical rankings.
"""

from pathlib import Path

import pytest

from jcodemunch_mcp.retrieval.signal_fusion import ChannelResult, fuse


class TestFuseTiebreak:
    def test_tied_scores_order_by_symbol_id(self):
        # Mirror-ranked channels give both symbols the same fused WRR score;
        # only the tiebreak can order them.
        ch1 = ChannelResult(name="lexical", ranked_ids=["b.py::y", "a.py::x"], weight=1.0)
        ch2 = ChannelResult(name="structural", ranked_ids=["a.py::x", "b.py::y"], weight=1.0)
        forward = fuse([ch1, ch2])
        backward = fuse([ch2, ch1])
        assert forward[0].score == forward[1].score
        assert [r.symbol_id for r in forward] == ["a.py::x", "b.py::y"]
        assert [r.symbol_id for r in forward] == [r.symbol_id for r in backward]

    def test_score_still_dominates_tiebreak(self):
        ch = ChannelResult(name="lexical", ranked_ids=["z.py::high", "a.py::low"])
        results = fuse([ch])
        assert results[0].symbol_id == "z.py::high"


def _build_corpus(root: Path, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        # Identical bodies → identical scores; only the tiebreak can order them.
        (root / f"{name}.py").write_text(
            "def widget_zorble_handler():\n    return 42\n"
        )


def _index(src: Path, store: Path):
    from jcodemunch_mcp.tools.index_folder import index_folder

    result = index_folder(str(src), storage_path=str(store), use_ai_summaries=False)
    assert result.get("success"), result
    return result


class TestCrossBuildOrderStability:
    """Two stores built from the same tree in different write orders must
    serve identical rankings — the cross-rebuild guarantee P1 exists for."""

    @pytest.fixture()
    def two_stores(self, tmp_path):
        names = ["mm_bravo", "mm_alpha", "mm_delta", "mm_charlie"]
        src_a = tmp_path / "src_a" / "proj"
        src_b = tmp_path / "src_b" / "proj"
        _build_corpus(src_a, names)
        _build_corpus(src_b, list(reversed(names)))
        store_a = tmp_path / "store_a"
        store_b = tmp_path / "store_b"
        _index(src_a, store_a)
        _index(src_b, store_b)
        return store_a, store_b

    def test_ranked_context_order_identical(self, two_stores):
        from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context

        store_a, store_b = two_stores
        out_a = get_ranked_context("proj", "widget zorble", storage_path=str(store_a))
        out_b = get_ranked_context("proj", "widget zorble", storage_path=str(store_b))
        ids_a = [i.get("symbol_id") or i.get("id") for i in out_a["context_items"]]
        ids_b = [i.get("symbol_id") or i.get("id") for i in out_b["context_items"]]
        assert ids_a, out_a
        assert ids_a == ids_b
        assert ids_a == sorted(ids_a)  # tied scores → id order, not walk order

    def test_repo_map_order_identical(self, two_stores):
        from jcodemunch_mcp.tools.get_repo_map import get_repo_map

        store_a, store_b = two_stores
        map_a = get_repo_map("proj", storage_path=str(store_a))
        map_b = get_repo_map("proj", storage_path=str(store_b))
        files_a = [f["path"] for f in map_a.get("files", [])]
        files_b = [f["path"] for f in map_b.get("files", [])]
        assert files_a, map_a
        assert files_a == files_b
