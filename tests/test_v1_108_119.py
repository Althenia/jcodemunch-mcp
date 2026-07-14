"""SCIP compile-time evidence P2 — find_implementations channel (v1.108.119).

Increment 2 of P2: find_implementations gains a `scip_implementation` channel
(confidence 1.0) fed by `scip_edges.kind='implementation'` (impl→iface). It
surfaces implementations the AST/duck channels can't see — e.g. an interface
with no declared subclassing.

Unit tests drive the reader `_scip_implementation_ids` against a hand-built db;
one integration test builds a real index, injects a compiler-verified
implementation edge, and confirms find_implementations surfaces it.
"""

import sqlite3
from pathlib import Path

from jcodemunch_mcp.evidence.scip_ingest import _ensure_scip_tables
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools.find_implementations import (
    find_implementations,
    _scip_implementation_ids,
)
from jcodemunch_mcp.tools.index_folder import index_folder


IFACE_ID = "iface.py::Iface#class"
IMPL_ID = "impl.py::Impl#class"


class _FakeSqlite:
    def __init__(self, db_path):
        self._db = db_path

    def _db_path(self, owner, name):
        return self._db


class _FakeStore:
    def __init__(self, db_path):
        self._sqlite = _FakeSqlite(db_path)


def _make_scip_db(tmp_path, *, with_scip=True, git_head="abc123", scip_head="abc123"):
    db = tmp_path / "local-proj.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE symbols (
            id TEXT PRIMARY KEY, name TEXT, file TEXT, kind TEXT,
            line INTEGER, end_line INTEGER
        );
        """
    )
    conn.execute("INSERT INTO meta VALUES ('git_head', ?)", (git_head,))
    conn.executemany(
        "INSERT INTO symbols (id, name, file, kind, line, end_line) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (IFACE_ID, "Iface", "iface.py", "class", 1, 5),
            (IMPL_ID, "Impl", "impl.py", "class", 1, 10),
        ],
    )
    if with_scip:
        _ensure_scip_tables(conn)
        conn.execute(
            "INSERT INTO scip_edges (from_symbol_id, to_symbol_id, kind, count, "
            "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
            (IMPL_ID, IFACE_ID, "implementation", 1, "t", "t"),  # impl → iface
        )
        conn.executemany(
            "INSERT INTO scip_meta (key, value) VALUES (?, ?)",
            [("tool", "scip-python"), ("ingested_at", "2026-07-11"), ("git_head", scip_head)],
        )
    conn.commit()
    conn.close()
    return db


# ── reader unit tests ────────────────────────────────────────────────────


class TestScipImplementationIds:
    def test_honest_empty_without_scip(self, tmp_path):
        db = _make_scip_db(tmp_path, with_scip=False)
        ids, meta, stale = _scip_implementation_ids(_FakeStore(db), "local", "proj", IFACE_ID)
        assert ids == []
        assert meta == {}
        assert stale is False

    def test_returns_impl_ids_for_interface(self, tmp_path):
        db = _make_scip_db(tmp_path)
        ids, meta, stale = _scip_implementation_ids(_FakeStore(db), "local", "proj", IFACE_ID)
        assert ids == [IMPL_ID]
        assert meta["tool"] == "scip-python"
        assert stale is False

    def test_no_edges_for_unrelated_target(self, tmp_path):
        db = _make_scip_db(tmp_path)
        ids, _meta, _stale = _scip_implementation_ids(_FakeStore(db), "local", "proj", IMPL_ID)
        assert ids == []  # nothing implements Impl

    def test_blank_target_id_is_noop(self, tmp_path):
        db = _make_scip_db(tmp_path)
        assert _scip_implementation_ids(_FakeStore(db), "local", "proj", "") == ([], {}, False)

    def test_staleness_reflected(self, tmp_path):
        db = _make_scip_db(tmp_path, git_head="newhead", scip_head="abc123")
        _ids, _meta, stale = _scip_implementation_ids(_FakeStore(db), "local", "proj", IFACE_ID)
        assert stale is True


# ── integration against a real index ─────────────────────────────────────


def _make_repo(tmp_path: Path, files: dict):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    storage = str(tmp_path / ".index")
    result = index_folder(str(tmp_path), use_ai_summaries=False, storage_path=storage)
    return result.get("repo", str(tmp_path)), storage


class TestFindImplementationsScipChannel:
    # Impl does NOT inherit Iface and shares no method name, so the AST/duck
    # channels cannot link them for a class target — only SCIP can.
    _FILES = {
        "iface.py": "class Iface:\n    def handle(self):\n        raise NotImplementedError\n",
        "impl.py": "class Impl:\n    def run(self):\n        return 1\n",
    }

    def test_scip_surfaces_impl_the_ast_misses(self, tmp_path):
        repo, storage = _make_repo(tmp_path, self._FILES)
        owner, name = repo.split("/", 1)
        store = IndexStore(base_path=storage)
        db = store._sqlite._db_path(owner, name)

        # Control: with no SCIP edge, Impl is not an implementation of Iface.
        baseline = find_implementations(repo, symbol="Iface", storage_path=storage)
        assert all(i["name"] != "Impl" for i in baseline["implementations"])
        assert "scip" not in baseline["_meta"]

        # Inject a compiler-verified implementation edge with the real symbol ids.
        conn = sqlite3.connect(str(db))
        ids = {
            r[1]: r[0]
            for r in conn.execute(
                "SELECT id, name FROM symbols WHERE name IN ('Iface', 'Impl')"
            )
        }
        head = conn.execute("SELECT value FROM meta WHERE key='git_head'").fetchone()
        head_val = head[0] if head and head[0] else ""
        _ensure_scip_tables(conn)
        conn.execute(
            "INSERT INTO scip_edges (from_symbol_id, to_symbol_id, kind, count, "
            "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
            (ids["Impl"], ids["Iface"], "implementation", 1, "t", "t"),
        )
        conn.executemany(
            "INSERT INTO scip_meta (key, value) VALUES (?, ?)",
            [("tool", "scip-python"), ("ingested_at", "2026-07-11"), ("git_head", head_val)],
        )
        conn.commit()
        conn.close()

        result = find_implementations(repo, symbol="Iface", storage_path=storage)
        impl = next((i for i in result["implementations"] if i["name"] == "Impl"), None)
        assert impl is not None, "SCIP channel should surface the implementation"
        assert impl["source"] == "scip"
        assert impl["confidence"] == 1.0
        assert impl["relationship"] == "interface_impl"
        assert impl["verification"] == "compiler_verified"
        assert result["_meta"]["scip"]["implementations"] >= 1
        assert result["_meta"]["scip"]["stale"] is False
        assert result["relationship_counts"].get("interface_impl", 0) >= 1
