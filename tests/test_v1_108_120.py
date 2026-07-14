"""SCIP compile-time evidence P2 — safety preflights (v1.108.120).

Increment 3 of P2 (the marquee): check_delete_safe and check_edit_safe treat
compiler-verified references (SCIP) as stronger blockers. A dynamic-dispatch or
barrel-re-export caller that the import graph and text search both miss is proof
of use once SCIP is ingested — flipping check_delete_safe from safe_to_delete to
scip_referenced, and check_edit_safe from safe_to_edit to signature_impact.

Unit tests drive the shared reader `scip_reference_files`; two integration tests
build a real index and prove the verdict flip after injecting a compiler-verified
reference edge the heuristic channels can't see.
"""

import sqlite3
from pathlib import Path

from jcodemunch_mcp.evidence.scip_ingest import _ensure_scip_tables
from jcodemunch_mcp.storage import IndexStore
from jcodemunch_mcp.tools._scip_consume import scip_reference_files
from jcodemunch_mcp.tools.check_delete_safe import check_delete_safe
from jcodemunch_mcp.tools.check_edit_safe import check_edit_safe
from jcodemunch_mcp.tools.index_folder import index_folder


WIDGET_ID = "widget.py::Widget#class"
HOST_ID = "host.py::Host#class"


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
            (WIDGET_ID, "Widget", "widget.py", "class", 1, 5),
            (HOST_ID, "Host", "host.py", "class", 1, 10),
        ],
    )
    if with_scip:
        _ensure_scip_tables(conn)
        conn.execute(
            "INSERT INTO scip_edges (from_symbol_id, to_symbol_id, kind, count, "
            "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
            (HOST_ID, WIDGET_ID, "reference", 1, "t", "t"),  # Host references Widget
        )
        conn.executemany(
            "INSERT INTO scip_meta (key, value) VALUES (?, ?)",
            [("tool", "scip-python"), ("ingested_at", "2026-07-11"), ("git_head", scip_head)],
        )
    conn.commit()
    conn.close()
    return db


# ── reader unit tests ────────────────────────────────────────────────────


class TestScipReferenceFiles:
    def test_honest_empty_without_scip(self, tmp_path):
        db = _make_scip_db(tmp_path, with_scip=False)
        files, meta, stale = scip_reference_files(_FakeStore(db), "local", "proj", WIDGET_ID)
        assert files == {}
        assert meta == {}
        assert stale is False

    def test_returns_referencing_files(self, tmp_path):
        db = _make_scip_db(tmp_path)
        files, meta, stale = scip_reference_files(_FakeStore(db), "local", "proj", WIDGET_ID)
        assert files == {"host.py": 1}
        assert meta["tool"] == "scip-python"
        assert stale is False

    def test_no_refs_into_unreferenced_target(self, tmp_path):
        db = _make_scip_db(tmp_path)
        files, _meta, _stale = scip_reference_files(_FakeStore(db), "local", "proj", HOST_ID)
        assert files == {}  # nothing references Host

    def test_blank_target_id_is_noop(self, tmp_path):
        db = _make_scip_db(tmp_path)
        assert scip_reference_files(_FakeStore(db), "local", "proj", "") == ({}, {}, False)

    def test_staleness_reflected(self, tmp_path):
        db = _make_scip_db(tmp_path, git_head="newhead", scip_head="abc123")
        _files, _meta, stale = scip_reference_files(_FakeStore(db), "local", "proj", WIDGET_ID)
        assert stale is True


# ── integration: the safe→blocked marquee ────────────────────────────────


def _make_repo(tmp_path: Path, files: dict):
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    storage = str(tmp_path / ".index")
    result = index_folder(str(tmp_path), use_ai_summaries=False, storage_path=storage)
    return result.get("repo", str(tmp_path)), storage


# host.py neither imports widget.py nor mentions "Widget" as a token, so the
# import graph and text-reference channels both miss the dependency — only a
# compiler-verified SCIP edge can prove Host uses Widget.
_FILES = {
    "widget.py": "class Widget:\n    def render(self):\n        return 1\n",
    "host.py": "class Host:\n    def go(self):\n        return 2\n",
}


def _inject_reference_edge(db, from_name, to_name):
    conn = sqlite3.connect(str(db))
    ids = {
        r[1]: r[0]
        for r in conn.execute(
            "SELECT id, name FROM symbols WHERE name IN (?, ?)", (from_name, to_name)
        )
    }
    head = conn.execute("SELECT value FROM meta WHERE key='git_head'").fetchone()
    head_val = head[0] if head and head[0] else ""
    _ensure_scip_tables(conn)
    conn.execute(
        "INSERT INTO scip_edges (from_symbol_id, to_symbol_id, kind, count, "
        "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
        (ids[from_name], ids[to_name], "reference", 1, "t", "t"),
    )
    conn.executemany(
        "INSERT INTO scip_meta (key, value) VALUES (?, ?)",
        [("tool", "scip-typescript"), ("ingested_at", "2026-07-11"), ("git_head", head_val)],
    )
    conn.commit()
    conn.close()


class TestSafetyPreflightScipMarquee:
    def test_check_delete_safe_flips_safe_to_blocked(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _FILES)
        owner, name = repo.split("/", 1)
        db = IndexStore(base_path=storage)._sqlite._db_path(owner, name)

        # Before import-scip: Widget looks deletable.
        before = check_delete_safe(repo, symbol="Widget", storage_path=storage)
        assert before["verdict"] == "safe_to_delete"
        assert "scip" not in before["_meta"]

        _inject_reference_edge(db, "Host", "Widget")

        # After import-scip: the compiler-verified reference blocks the delete.
        after = check_delete_safe(repo, symbol="Widget", storage_path=storage)
        assert after["verdict"] == "scip_referenced"
        assert after["signals"]["scip_external_ref_count"] == 1
        assert after["_meta"]["scip"]["verified_external_refs"] == 1
        assert after["_meta"]["scip"]["stale"] is False
        assert after["confidence"] <= 0.2
        assert any(
            b["kind"] == "scip_reference" and b.get("verification") == "compiler_verified"
            for b in after["blockers"]
        )

    def test_check_edit_safe_flips_safe_to_signature_impact(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _FILES)
        owner, name = repo.split("/", 1)
        db = IndexStore(base_path=storage)._sqlite._db_path(owner, name)

        before = check_edit_safe(repo, symbol="Widget", storage_path=storage)
        assert before["verdict"] == "safe_to_edit"
        assert "scip" not in before["_meta"]

        _inject_reference_edge(db, "Host", "Widget")

        after = check_edit_safe(repo, symbol="Widget", storage_path=storage)
        assert after["verdict"] == "signature_impact"
        assert after["signals"]["scip_external_ref_count"] == 1
        assert after["_meta"]["scip"]["verified_external_refs"] == 1
        assert any(
            b["kind"] == "scip_reference" and b.get("verification") == "compiler_verified"
            for b in after["blockers"]
        )
