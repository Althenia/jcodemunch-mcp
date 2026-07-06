"""v1.108.101 — audit W5: hybrid BM25 normalization no longer collapses the
lexical dynamic range when an exact-name match co-occurs.

The semantic/hybrid scorer folded the identity boost (exact 50 / prefix 30 /
segment 20) into the BM25 value used for `max_bm25`, so as soon as any exact
match existed the normalization denominator jumped far above the real lexical
range and every non-exact result's genuine lexical score was divided down toward
zero — hybrid ranking degenerated toward pure-semantic among them. The fix
normalizes lexical (identity-excluded) and identity on their own scales and takes
the max, so an exact match still dominates while non-exact results keep their
lexical range.
"""

from __future__ import annotations

import math
from unittest.mock import patch

from jcodemunch_mcp.parser.symbols import Symbol
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools.search_symbols import search_symbols


def _sym(id_, name, summary=""):
    return Symbol(
        id=id_, file="src/a.py", name=name, qualified_name=name, kind="function",
        language="python", signature=f"def {name}():", byte_offset=0,
        byte_length=50, summary=summary,
    )


def _vec(seed: float, dim: int = 8) -> list[float]:
    v = [math.cos(seed + i * 0.5) for i in range(dim)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _seed(tmp_path, symbols):
    store = IndexStore(base_path=str(tmp_path))
    store.save_index(
        owner="test", name="w5",
        source_files=["src/a.py"],
        symbols=symbols,
        raw_files={"src/a.py": "".join(f"def {s.name}(): pass\n" for s in symbols)},
        languages={"python": 1},
        file_languages={"src/a.py": "python"},
    )
    return "test/w5"


def test_w5_lexical_range_survives_when_an_exact_match_is_present(tmp_path, monkeypatch):
    # "pool" is an exact-name match (identity boost). "manager" has the query
    # term only in its summary (lexical overlap, no identity). "widget" has no
    # overlap at all. manager and widget get the SAME embedding, so cosine cannot
    # separate them — only the lexical channel can.
    symbols = [
        _sym("s_pool", "pool", summary="object pool"),
        _sym("s_mgr", "manager", summary="database pool handler for connections"),
        _sym("s_widget", "widget", summary="renders html output to the page"),
    ]
    repo = _seed(tmp_path, symbols)

    query_vec = _vec(0.0)
    equal_vec = _vec(1.3)  # identical cosine for manager AND widget

    def _mock_embed(texts, provider, model, task_type=None):
        if len(texts) == 1 and texts[0].startswith("pool"):
            return [query_vec]
        out = []
        for t in texts:
            out.append(_vec(0.02) if t.startswith("pool") else equal_vec)
        return out

    monkeypatch.setenv("JCODEMUNCH_EMBED_MODEL", "all-MiniLM-L6-v2")
    with patch("jcodemunch_mcp.tools.embed_repo.embed_texts", side_effect=_mock_embed):
        from jcodemunch_mcp.tools.embed_repo import embed_repo
        embed_repo(repo, storage_path=str(tmp_path))
        res = search_symbols(
            repo, "pool", semantic=True, semantic_weight=0.5,
            debug=True, storage_path=str(tmp_path),
        )

    assert res.get("error") is None, res
    scores = {r["name"]: r["score"] for r in res["results"] if "score" in r}
    assert {"pool", "manager", "widget"} <= set(scores), scores

    # Exact match wins.
    assert scores["pool"] == max(scores.values())
    # With equal cosine, the lexical-overlap symbol must outrank the no-overlap
    # one by a real margin — the collapse would leave them ~tied on cosine alone.
    assert scores["manager"] > scores["widget"]
    assert scores["manager"] - scores["widget"] > 0.1, scores


def test_w5_exact_match_still_ranks_first(tmp_path, monkeypatch):
    """The fix must not demote exact matches (its identity channel stays at 1.0,
    exactly as the old identity-boosted bm25_norm did). Checked on the semantic
    path the replay gate does not exercise. With semantic_weight <= 1/3 an exact
    match (lexical channel 1.0) provably outranks any non-lexical result even if
    that result has the maximum possible cosine and the exact match the minimum."""
    symbols = [
        _sym("s_pool", "connection_pool", summary="manages a database pool"),
        _sym("s_other", "render_html", summary="html rendering helpers"),
    ]
    repo = _seed(tmp_path, symbols)

    def _mock_embed(texts, provider, model, task_type=None):
        # Deliberately hand the NON-exact symbol the closer embedding; the exact
        # match must still win at this semantic weight.
        if len(texts) == 1:
            return [_vec(0.0)]
        return [_vec(2.5) if t.startswith("connection_pool") else _vec(0.02) for t in texts]

    monkeypatch.setenv("JCODEMUNCH_EMBED_MODEL", "all-MiniLM-L6-v2")
    with patch("jcodemunch_mcp.tools.embed_repo.embed_texts", side_effect=_mock_embed):
        from jcodemunch_mcp.tools.embed_repo import embed_repo
        embed_repo(repo, storage_path=str(tmp_path))
        res = search_symbols(
            repo, "connection_pool", semantic=True, semantic_weight=0.3,
            storage_path=str(tmp_path),
        )
    assert res.get("error") is None, res
    assert res["results"][0]["name"] == "connection_pool"
