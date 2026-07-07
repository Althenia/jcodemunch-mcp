"""v1.108.108 — audit WS-5 tail: atomic content-body writes + a single
parametrized test that drives all four store-writing paths.

Atomic writes: _write_cached_text now writes to a sibling temp file and swaps it
in with os.replace, so an interrupted write leaves the temp file rather than a
truncated body at the real path (which readers would serve as source).

Four-path persistence: save_index, incremental_save, save_branch_delta, and the
SCIP ingest all mutate a repo's .db and must serialise via the indexwrite lock
(V9). One parametrized test asserts the shared invariant across all four.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jcodemunch_mcp.parser.symbols import Symbol
from jcodemunch_mcp.storage import IndexStore, process_locks
from jcodemunch_mcp.storage import sqlite_store as _ss
from jcodemunch_mcp.tools.import_scip import import_scip


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


# ----------------------------------------------------------- four-path lock

def _drive_save_index(tmp_path, store):
    store.save_index(
        owner="t", name="r", source_files=["a.py"], symbols=[_sym("s1", "foo")],
        raw_files={"a.py": "def foo(): pass\n"}, languages={"python": 1},
        file_languages={"a.py": "python"},
    )


def _drive_incremental_save(tmp_path, store):
    store.incremental_save(
        owner="t", name="r", changed_files=["a.py"], new_files=[], deleted_files=[],
        new_symbols=[_sym("s1", "foo2")], raw_files={"a.py": "def foo2(): pass\n"},
        file_languages={"a.py": "python"},
    )


def _drive_save_branch_delta(tmp_path, store):
    store.save_branch_delta(
        owner="t", name="r", branch="b", changed_files=["a.py"], new_files=[],
        deleted_files=[], new_symbols=[_sym("s1", "foo")],
        raw_files={"a.py": "def foo(): pass\n"},
    )


def _drive_import_scip(tmp_path, store):
    # A missing .scip file errors AFTER the lock is taken inside the with-block.
    import_scip(path=str(tmp_path / "nope.scip"), repo="t/r", storage_path=str(tmp_path))


@pytest.mark.parametrize("driver", [
    _drive_save_index,
    _drive_incremental_save,
    _drive_save_branch_delta,
    _drive_import_scip,
])
def test_all_store_write_paths_hold_indexwrite_lock(tmp_path, driver):
    """Every path that mutates a repo's .db must serialise via the indexwrite
    lock keyed on the repo, so a watcher delta / branch write / scip ingest
    can't interleave with a full reindex from another process."""
    store = _seed(tmp_path)  # seed outside the spy so only the driver is recorded

    calls: list[tuple] = []
    real = process_locks.held

    def spy(name, target, root, **kw):
        calls.append((name, target))
        return real(name, target, root, **kw)

    with patch.object(process_locks, "held", side_effect=spy):
        driver(tmp_path, store)

    assert ("indexwrite", "t/r") in calls, f"{driver.__name__} did not take the lock: {calls}"


# ----------------------------------------------------------- atomic writes

def test_write_cached_text_is_atomic_and_leaves_no_temp(tmp_path):
    store = IndexStore(base_path=str(tmp_path))._sqlite
    dest = tmp_path / "body.txt"
    store._write_cached_text(dest, "hello\nworld\n")
    assert dest.read_text() == "hello\nworld\n"
    assert not list(tmp_path.glob("*.tmp.*")), "temp file leaked after a successful write"


def test_write_cached_text_overwrites_cleanly(tmp_path):
    store = IndexStore(base_path=str(tmp_path))._sqlite
    dest = tmp_path / "body.txt"
    store._write_cached_text(dest, "first")
    store._write_cached_text(dest, "second")
    assert dest.read_text() == "second"  # replaced, not appended
    assert not list(tmp_path.glob("*.tmp.*"))


def test_write_cached_text_failure_leaves_original_intact(tmp_path):
    store = IndexStore(base_path=str(tmp_path))._sqlite
    dest = tmp_path / "body.txt"
    dest.write_text("original")
    # A failed swap must not corrupt the existing body, and must clean the temp.
    with patch.object(_ss.os, "replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            store._write_cached_text(dest, "new content that never lands")
    assert dest.read_text() == "original"
    assert not list(tmp_path.glob("*.tmp.*"))
