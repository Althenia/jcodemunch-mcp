"""Regression tests for v1.108.127 — index_repo full re-index summary preservation.

Bug (#367): a non-incremental (full) `index_repo` re-index crashed with
`'dict' object has no attribute 'summary'`. Loaded indexes carry symbols as
dicts (SQLiteIndexStore._build_index_from_rows → _row_to_symbol_dict), but the
full-path summary-preservation map accessed them by attribute (`s.summary`,
`s.file`, `s.name`, `s.kind`) instead of by key. index_folder used dict access
and worked; index_repo diverged. Fix aligns index_repo with dict access.
"""

from unittest.mock import AsyncMock

import pytest

from jcodemunch_mcp.parser import Symbol
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools import index_repo as index_repo_mod


_FILE = "mod.py"
_CONTENT = "def foo():\n    return 1\n"


def _preseed_index(storage_path: str) -> None:
    """Save an index whose one symbol carries an AI summary, so the full
    re-index path builds a non-empty summary-preservation map."""
    store = IndexStore(base_path=storage_path)
    symbols = [
        Symbol(
            id=f"{_FILE}::foo",
            file=_FILE,
            name="foo",
            qualified_name="foo",
            kind="function",
            language="python",
            signature="def foo():",
            summary="Returns the integer one.",
            byte_offset=0,
            byte_length=len(_CONTENT),
        )
    ]
    store.save_index(
        owner="jgravelle",
        name="GroqApiLibrary",
        source_files=[_FILE],
        symbols=symbols,
        raw_files={_FILE: _CONTENT},
        languages={"python": 1},
    )


@pytest.mark.asyncio
async def test_full_reindex_preserves_summaries_without_attribute_error(tmp_path, monkeypatch):
    """Full re-index over an existing index must not raise AttributeError."""
    storage = str(tmp_path)
    _preseed_index(storage)

    # Mock the GitHub fetch surface so no network is hit. Content matches the
    # pre-seeded raw_files so the file hash is unchanged → the summary-
    # preservation comprehension (the crash site) actually runs.
    monkeypatch.setattr(
        index_repo_mod,
        "fetch_repo_tree",
        AsyncMock(return_value=([{"path": _FILE, "type": "blob", "sha": "abc123", "size": len(_CONTENT)}], "treesha")),
    )
    monkeypatch.setattr(index_repo_mod, "fetch_gitignore", AsyncMock(return_value=""))
    monkeypatch.setattr(index_repo_mod, "fetch_file_content", AsyncMock(return_value=_CONTENT))

    result = await index_repo_mod.index_repo(
        url="jgravelle/GroqApiLibrary",
        use_ai_summaries=False,
        storage_path=storage,
        incremental=False,
    )

    assert result.get("success") is True, result
    assert result.get("symbol_count", 0) >= 1


@pytest.mark.asyncio
async def test_loaded_index_symbols_are_dicts(tmp_path):
    """Guard the root cause: loaded index symbols are dicts, so attribute
    access on them would fail. Pins the contract index_repo now relies on."""
    storage = str(tmp_path)
    _preseed_index(storage)

    loaded = IndexStore(base_path=storage).load_index("jgravelle", "GroqApiLibrary")
    assert loaded is not None
    assert loaded.symbols, "expected at least one symbol"
    sym = loaded.symbols[0]
    assert isinstance(sym, dict)
    assert sym["summary"] == "Returns the integer one."
    assert not hasattr(sym, "summary")
