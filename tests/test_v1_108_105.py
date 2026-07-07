"""v1.108.105 — audit V9: index-write lock on the delta paths + load_index
corruption guard.

V9a: incremental_save / save_branch_delta / the SCIP ingest wrote to the .db
without the `indexwrite` process lock that the full save_index holds, so a
watcher delta (or a branch-delta / import-scip) could interleave its
DELETE/INSERT batches with a concurrent full reindex from another process and
corrupt the index. All three now acquire the same lock, keyed identically to
save_index so they actually serialise against it.

V9b: load_index had no corruption guard (inspect_index already had one), so a
corrupt .db raised a raw sqlite3.DatabaseError traceback out of every retrieval
tool. It now returns None on DatabaseError; callers routed through
load_repo_index_or_error fall through to inspect_index and get a structured
`sqlite_corrupt` + re-index hint.
"""

from __future__ import annotations

from unittest.mock import patch

from jcodemunch_mcp.parser.symbols import Symbol
from jcodemunch_mcp.storage import IndexStore, process_locks
from jcodemunch_mcp.storage import sqlite_store as _ss
from jcodemunch_mcp.tools._utils import load_repo_index_or_error


def _sym(id_, name):
    return Symbol(
        id=id_, file="a.py", name=name, qualified_name=name, kind="function",
        language="python", signature=f"def {name}():", byte_offset=0,
        byte_length=10, summary="",
    )


def _seed(tmp_path):
    store = IndexStore(base_path=str(tmp_path))
    store.save_index(
        owner="t", name="r", source_files=["a.py"], symbols=[_sym("s1", "foo")],
        raw_files={"a.py": "def foo(): pass\n"}, languages={"python": 1},
        file_languages={"a.py": "python"},
    )
    return store


def _lock_spy():
    """Return (spy_fn, calls) that records held() calls and delegates."""
    calls: list[tuple] = []
    real = process_locks.held

    def spy(name, target, root, **kw):
        calls.append((name, target))
        return real(name, target, root, **kw)

    return spy, calls


# ------------------------------------------------------------------ V9a locks

def test_incremental_save_acquires_indexwrite_lock(tmp_path):
    store = _seed(tmp_path)
    spy, calls = _lock_spy()
    with patch.object(process_locks, "held", side_effect=spy):
        idx = store.incremental_save(
            owner="t", name="r", changed_files=["a.py"], new_files=[],
            deleted_files=[], new_symbols=[_sym("s1", "foo2")],
            raw_files={"a.py": "def foo2(): pass\n"}, file_languages={"a.py": "python"},
        )
    assert ("indexwrite", "t/r") in calls
    # behavior preserved: the delta still applied
    assert idx is not None
    assert [s["name"] for s in idx.symbols] == ["foo2"]


def test_save_branch_delta_acquires_indexwrite_lock(tmp_path):
    store = _seed(tmp_path)
    spy, calls = _lock_spy()
    with patch.object(process_locks, "held", side_effect=spy):
        store.save_branch_delta(
            owner="t", name="r", branch="feature", changed_files=["a.py"],
            new_files=[], deleted_files=[], new_symbols=[_sym("s1", "foo")],
            raw_files={"a.py": "def foo(): pass\n"},
        )
    assert ("indexwrite", "t/r") in calls
    # the delta is retrievable (read method lives on the sqlite store)
    delta = store._sqlite.load_branch_delta("t", "r", "feature")
    assert delta is not None


def test_import_scip_acquires_indexwrite_lock(tmp_path):
    _seed(tmp_path)
    from jcodemunch_mcp.tools.import_scip import import_scip
    spy, calls = _lock_spy()
    # A non-existent .scip file makes ingest raise FileNotFoundError, but only
    # AFTER the lock is acquired inside the `with` block — which is what we assert.
    with patch.object(process_locks, "held", side_effect=spy):
        out = import_scip(path=str(tmp_path / "nope.scip"), repo="t/r", storage_path=str(tmp_path))
    assert ("indexwrite", "t/r") in calls
    assert out["success"] is False  # the missing scip file surfaces as an error


def test_locked_helpers_exist(tmp_path):
    store = _seed(tmp_path)
    # The public methods delegate to _locked inner bodies (mirrors save_index).
    assert hasattr(store._sqlite, "_incremental_save_locked")
    assert hasattr(store._sqlite, "_save_branch_delta_locked")


# ------------------------------------------------------------------ V9b guard

def _corrupt_db(store):
    dbp = store._sqlite._db_path("t", "r")
    with open(dbp, "wb") as fh:
        fh.write(b"this is not a sqlite database" * 100)
    _ss._index_cache.clear()  # force a re-read past the in-memory cache


def test_load_index_returns_none_on_corrupt_db(tmp_path):
    store = _seed(tmp_path)
    _corrupt_db(store)
    # No raw traceback — just a graceful None.
    assert store.load_index("t", "r") is None


def test_corrupt_db_surfaces_sqlite_corrupt_through_tool_helper(tmp_path):
    store = _seed(tmp_path)
    _corrupt_db(store)
    index, err, status = load_repo_index_or_error("t/r", storage_path=str(tmp_path))
    assert index is None
    assert err is not None
    assert err["status"] == "sqlite_corrupt"
    assert err["hint"]  # a re-index hint, not a traceback
