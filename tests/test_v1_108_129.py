"""Tests for v1.108.129 — keystone-protected structural compression.

Covers:
  - retrieval/entropy_prune: keystone detection, signal ranking, honest elision
  - get_ranked_context(compress=True): default byte-identical, fits more,
    keystone lines never dropped, honest per-item metadata (both paths)
"""

from pathlib import Path

from jcodemunch_mcp.retrieval.entropy_prune import (
    prune_source,
    is_keystone,
    line_signal,
)
from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context
from jcodemunch_mcp.tools.index_folder import index_folder


def _count_tokens(s: str) -> int:
    # Same cheap proxy shape the tool packs with (chars/4-ish); good enough here.
    return max(1, len(s) // 4)


# ---------------------------------------------------------------------------
# entropy_prune unit tests
# ---------------------------------------------------------------------------

class TestKeystoneDetection:
    def test_control_flow_is_keystone(self):
        for ln in ["    return x", "    if a and b:", "        raise ValueError(x)",
                   "    for i in items:", "    while running:"]:
            assert is_keystone(ln), ln

    def test_signature_is_keystone(self):
        for ln in ["def run(self):", "class Engine:", "    @property",
                   "async def fetch(self):"]:
            assert is_keystone(ln), ln

    def test_constraint_cues_are_keystone(self):
        assert is_keystone("    # value must be positive")
        assert is_keystone("    x = a != b")

    def test_plain_lines_not_keystone(self):
        assert not is_keystone("    total = total + 1")
        assert not is_keystone("    name = compute_name(row)")
        assert not is_keystone("")

    def test_blank_line_zero_signal(self):
        assert line_signal("") == 0.0
        assert line_signal("      ") == 0.0
        assert line_signal("x = compute_the_thing(a, b, c)") > 0.0


class TestPruneSource:
    def test_small_body_unchanged(self):
        src = "def f():\n    return 1\n"
        r = prune_source(src, max_tokens=1000, count_tokens=_count_tokens)
        assert not r.is_pruned
        assert r.text == src
        assert r.elided_lines == 0

    def test_large_body_pruned_with_marker(self):
        # A big body of low-signal filler around a few keystones.
        filler = "\n".join(f"    v{i} = v{i}" for i in range(200))
        src = f"def f(x):\n{filler}\n    return x\n"
        r = prune_source(src, max_tokens=40, count_tokens=_count_tokens)
        assert r.is_pruned
        assert r.elided_lines > 0
        assert r.kept_lines + r.elided_lines == r.total_lines
        assert "elided" in r.text

    def test_keystones_survive_prune(self):
        filler = "\n".join(f"    noise_{i} = {i}" for i in range(200))
        src = f"def handler(req):\n{filler}\n    if req.bad:\n        raise Abort()\n    return ok\n"
        r = prune_source(src, max_tokens=30, count_tokens=_count_tokens)
        assert r.is_pruned
        # Every keystone line must appear verbatim in the pruned view.
        for keystone in ["def handler(req):", "    if req.bad:", "        raise Abort()", "    return ok"]:
            assert keystone in r.text, keystone

    def test_empty_source(self):
        r = prune_source("", max_tokens=10, count_tokens=_count_tokens)
        assert not r.is_pruned
        assert r.total_lines == 0


# ---------------------------------------------------------------------------
# get_ranked_context(compress=...) integration
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path, files: dict) -> tuple:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    storage = str(tmp_path / ".index")
    result = index_folder(str(tmp_path), use_ai_summaries=False, storage_path=storage)
    return result.get("repo", str(tmp_path)), storage


# A repo whose symbols have large low-signal bodies wrapped around keystones.
def _big_symbol(name: str, n: int) -> str:
    body = "\n".join(f"    tmp_{i} = tmp_{i}" for i in range(n))
    return f"def {name}(query):\n{body}\n    if query:\n        return handle(query)\n    return None\n"


_REPO = {
    "search.py": _big_symbol("search_index", 120) + "\n" + _big_symbol("rank_results", 120),
    "handler.py": _big_symbol("handle_query", 120),
}


class TestCompressIntegration:
    def test_default_off_is_byte_identical(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _REPO)
        base = get_ranked_context(repo, query="search index rank query", token_budget=4000, storage_path=storage)
        again = get_ranked_context(repo, query="search index rank query", token_budget=4000,
                                   compress=False, storage_path=storage)
        assert "error" not in base
        # No compression metadata leaks onto the default path.
        for item in base.get("context_items", []):
            assert "source_pruned" not in item
        # Same sources when compress is off.
        assert [i["source"] for i in base["context_items"]] == [i["source"] for i in again["context_items"]]

    def test_compress_fits_more_or_equal_items(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _REPO)
        budget = 300
        plain = get_ranked_context(repo, query="search index rank query handle", token_budget=budget, storage_path=storage)
        comp = get_ranked_context(repo, query="search index rank query handle", token_budget=budget,
                                  compress=True, storage_path=storage)
        assert "error" not in comp
        # Compression should never fit FEWER items under the same tight budget.
        assert comp["items_included"] >= plain["items_included"]
        assert comp["total_tokens"] <= budget

    def test_compress_marks_pruned_items_and_keeps_keystones(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _REPO)
        comp = get_ranked_context(repo, query="search index rank query handle", token_budget=300,
                                  compress=True, storage_path=storage)
        pruned = [i for i in comp["context_items"] if i.get("source_pruned")]
        assert pruned, "expected at least one pruned item under a tight budget"
        for item in pruned:
            assert item["source_is_pruned_view"] is True
            assert item["source_elided_lines"] > 0
            assert item["source_kept_lines"] + item["source_elided_lines"] == item["source_total_lines"]
            assert "elided" in item["source"]
            # Keystone (control-flow / return) preserved even in the pruned view.
            assert "return" in item["source"] or "if " in item["source"]

    def test_compress_fusion_path(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _REPO)
        comp = get_ranked_context(repo, query="search index rank query handle", token_budget=300,
                                  fusion=True, compress=True, storage_path=storage)
        assert "error" not in comp
        assert comp["_meta"].get("fusion") is True
        # Fusion path honors compression + surfaces the same honest metadata.
        assert any(i.get("source_pruned") for i in comp["context_items"])
