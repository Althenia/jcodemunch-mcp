"""v1.108.106 — audit W7: incremental_save refreshes package_names.

The full save_index persisted `package_names` (the published-package registry
entry used by cross-repo features) into the meta table, but incremental_save
never touched it. So after a delta write that added or renamed a manifest, the
registry saw a stale package_names until the next full reindex. incremental_save
now accepts and persists package_names; None preserves the existing value (the
common delta that didn't touch a manifest).

Branch deltas are intentionally NOT plumbed: they don't write base meta and the
cross-repo registry reads base indexes, so a branch-scoped package_names has no
consumer today.
"""

from __future__ import annotations

from jcodemunch_mcp.parser.symbols import Symbol
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.storage import sqlite_store as _ss


def _sym(id_, name):
    return Symbol(
        id=id_, file="a.py", name=name, qualified_name=name, kind="function",
        language="python", signature=f"def {name}():", byte_offset=0,
        byte_length=10, summary="",
    )


def _store(tmp_path):
    store = IndexStore(base_path=str(tmp_path))
    store.save_index(
        owner="t", name="r", source_files=["a.py"], symbols=[_sym("s1", "foo")],
        raw_files={"a.py": "def foo(): pass\n"}, languages={"python": 1},
        file_languages={"a.py": "python"}, package_names=["oldpkg"],
    )
    return store


def _reload_names(store):
    _ss._index_cache.clear()
    return store.load_index("t", "r").package_names


def test_incremental_without_package_names_preserves_existing(tmp_path):
    store = _store(tmp_path)
    store.incremental_save(
        owner="t", name="r", changed_files=["a.py"], new_files=[], deleted_files=[],
        new_symbols=[_sym("s1", "foo2")], raw_files={"a.py": "def foo2(): pass\n"},
    )
    assert _reload_names(store) == ["oldpkg"]


def test_incremental_with_package_names_refreshes_returned_index(tmp_path):
    store = _store(tmp_path)
    idx = store.incremental_save(
        owner="t", name="r", changed_files=["a.py"], new_files=[], deleted_files=[],
        new_symbols=[_sym("s1", "foo3")], raw_files={"a.py": "def foo3(): pass\n"},
        package_names=["newpkg"],
    )
    # fast (patch) path: the in-memory index returned to the caller is refreshed
    assert idx.package_names == ["newpkg"]


def test_incremental_with_package_names_persists_to_meta(tmp_path):
    store = _store(tmp_path)
    store.incremental_save(
        owner="t", name="r", changed_files=["a.py"], new_files=[], deleted_files=[],
        new_symbols=[_sym("s1", "foo3")], raw_files={"a.py": "def foo3(): pass\n"},
        package_names=["newpkg"],
    )
    # cold path: reload from the DB meta table reflects the rename
    assert _reload_names(store) == ["newpkg"]


def test_incremental_can_clear_package_names(tmp_path):
    # An explicit empty list is a real value (manifest removed), distinct from
    # None (not recomputed). It must overwrite, not preserve.
    store = _store(tmp_path)
    store.incremental_save(
        owner="t", name="r", changed_files=["a.py"], new_files=[], deleted_files=[],
        new_symbols=[_sym("s1", "foo3")], raw_files={"a.py": "def foo3(): pass\n"},
        package_names=[],
    )
    assert _reload_names(store) == []
