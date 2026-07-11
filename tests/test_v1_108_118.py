"""SCIP compile-time evidence P2 — graph consumers (v1.108.118).

P1 (v1.108.96) wired SCIP compiler-verified edges into find_references. P2
unions the same `scip_edges` reference edges into the graph traversal tools:
get_blast_radius (file-level affected set) and get_call_hierarchy (caller edges).

These exercise the attach helpers directly against a hand-built index database —
symbols + meta + the scip_* family populated via SQL (using the real
_ensure_scip_tables), so no protobuf ingest is needed. Scenario mirrors the P1
fixture: a.py's `caller` references b.py's `callee`.
"""

import copy
import sqlite3

from jcodemunch_mcp.evidence.scip_ingest import _ensure_scip_tables
from jcodemunch_mcp.tools.get_blast_radius import _attach_scip_to_blast
from jcodemunch_mcp.tools.get_call_hierarchy import _attach_scip_to_hierarchy


CALLER_ID = "src/a.py::caller#function"
CALLEE_ID = "src/b.py::callee#function"


class _FakeSqlite:
    def __init__(self, db_path):
        self._db = db_path

    def _db_path(self, owner, name):
        return self._db


class _FakeStore:
    def __init__(self, db_path):
        self._sqlite = _FakeSqlite(db_path)


def _make_db(tmp_path, *, with_scip=True, git_head="abc123", scip_head="abc123"):
    """Two-symbol index: caller (src/a.py) references callee (src/b.py)."""
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
            (CALLER_ID, "caller", "src/a.py", "function", 10, 20),
            (CALLEE_ID, "callee", "src/b.py", "function", 5, 15),
        ],
    )
    if with_scip:
        _ensure_scip_tables(conn)
        conn.execute(
            "INSERT INTO scip_edges (from_symbol_id, to_symbol_id, kind, count, "
            "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
            (CALLER_ID, CALLEE_ID, "reference", 1, "t", "t"),
        )
        conn.executemany(
            "INSERT INTO scip_meta (key, value) VALUES (?, ?)",
            [("tool", "scip-python"), ("ingested_at", "2026-07-11"), ("git_head", scip_head)],
        )
    conn.commit()
    conn.close()
    return db


def _blast_response(confirmed_files):
    return {
        "repo": "local/proj",
        "symbol": {"id": CALLEE_ID, "name": "callee", "kind": "function",
                   "file": "src/b.py", "line": 5},
        "importer_count": len(confirmed_files),
        "confirmed_count": len(confirmed_files),
        "confirmed": [{"file": f, "reference_count": 1} for f in confirmed_files],
        "potential": [],
        "_meta": {"timing_ms": 1.0},
    }


def _hierarchy_response(callers, direction="both"):
    return {
        "repo": "local/proj",
        "symbol": {"id": CALLEE_ID, "name": "callee", "kind": "function",
                   "file": "src/b.py", "line": 5},
        "direction": direction,
        "caller_count": len(callers),
        "callee_count": 0,
        "callers": list(callers),
        "callees": [],
        "_meta": {"timing_ms": 1.0, "resolution_tiers": {}},
    }


# ── get_blast_radius ────────────────────────────────────────────────────


class TestBlastRadiusScip:
    def test_honest_empty_without_scip(self, tmp_path):
        db = _make_db(tmp_path, with_scip=False)
        resp = _blast_response(["src/x.py"])
        before = copy.deepcopy(resp)
        out = _attach_scip_to_blast(resp, _FakeStore(db), "local", "proj")
        assert out == before

    def test_scip_only_file_appended(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _blast_response([])  # import graph missed src/a.py (dynamic dispatch)
        out = _attach_scip_to_blast(resp, _FakeStore(db), "local", "proj")
        scip_rows = [c for c in out["confirmed"] if c.get("source") == "scip"]
        assert len(scip_rows) == 1
        assert scip_rows[0]["file"] == "src/a.py"
        assert scip_rows[0]["verification"] == "compiler_verified"
        assert out["confirmed_count"] == 1
        assert out["_meta"]["scip"]["scip_only_files"] == 1
        assert out["_meta"]["scip"]["verified_files"] == 0
        assert out["_meta"]["scip"]["stale"] is False

    def test_existing_confirmed_gains_verification(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _blast_response(["src/a.py"])  # import graph already found it
        out = _attach_scip_to_blast(resp, _FakeStore(db), "local", "proj")
        assert out["confirmed"][0]["verification"] == "compiler_verified"
        assert out["_meta"]["scip"]["verified_files"] == 1
        assert out["_meta"]["scip"]["scip_only_files"] == 0
        assert out["confirmed_count"] == 1  # no new row, importer already present

    def test_importer_count_untouched(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _blast_response([])
        out = _attach_scip_to_blast(resp, _FakeStore(db), "local", "proj")
        # SCIP-only files are exactly what the import graph missed, so the raw
        # importer_count must not be inflated by them.
        assert out["importer_count"] == 0

    def test_idempotent(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _blast_response([])
        once = _attach_scip_to_blast(resp, _FakeStore(db), "local", "proj")
        first = copy.deepcopy(once)
        twice = _attach_scip_to_blast(once, _FakeStore(db), "local", "proj")
        assert twice == first

    def test_staleness_flagged(self, tmp_path):
        db = _make_db(tmp_path, git_head="newhead", scip_head="abc123")
        resp = _blast_response([])
        out = _attach_scip_to_blast(resp, _FakeStore(db), "local", "proj")
        assert out["_meta"]["scip"]["stale"] is True
        assert "import-scip" in out["_meta"]["scip"]["note"]


# ── get_call_hierarchy ──────────────────────────────────────────────────


class TestCallHierarchyScip:
    def test_honest_empty_without_scip(self, tmp_path):
        db = _make_db(tmp_path, with_scip=False)
        resp = _hierarchy_response([])
        before = copy.deepcopy(resp)
        out = _attach_scip_to_hierarchy(resp, _FakeStore(db), "local", "proj")
        assert out == before

    def test_scip_caller_added(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _hierarchy_response([])  # AST found no callers (string dispatch)
        out = _attach_scip_to_hierarchy(resp, _FakeStore(db), "local", "proj")
        scip_callers = [c for c in out["callers"] if c.get("source") == "scip"]
        assert len(scip_callers) == 1
        c = scip_callers[0]
        assert c["id"] == CALLER_ID
        assert c["name"] == "caller"
        assert c["kind"] == "function"
        assert c["file"] == "src/a.py"
        assert c["line"] == 10
        assert c["depth"] == 1
        assert c["resolution"] == "scip_reference"
        assert c["verification"] == "compiler_verified"
        assert out["caller_count"] == 1
        assert out["_meta"]["resolution_tiers"]["scip_reference"] == 1
        assert out["_meta"]["scip"]["scip_only_callers"] == 1

    def test_existing_caller_gains_verification(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _hierarchy_response([
            {"id": CALLER_ID, "name": "caller", "kind": "function",
             "file": "src/a.py", "line": 10, "depth": 1, "resolution": "ast_resolved"},
        ])
        out = _attach_scip_to_hierarchy(resp, _FakeStore(db), "local", "proj")
        assert out["callers"][0]["verification"] == "compiler_verified"
        assert out["caller_count"] == 1  # no duplicate caller
        assert out["_meta"]["scip"]["verified_callers"] == 1
        assert out["_meta"]["scip"]["scip_only_callers"] == 0

    def test_callees_only_is_noop(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _hierarchy_response([], direction="callees")
        before = copy.deepcopy(resp)
        out = _attach_scip_to_hierarchy(resp, _FakeStore(db), "local", "proj")
        assert out == before  # callers direction not requested → no SCIP union

    def test_idempotent(self, tmp_path):
        db = _make_db(tmp_path)
        resp = _hierarchy_response([])
        once = _attach_scip_to_hierarchy(resp, _FakeStore(db), "local", "proj")
        first = copy.deepcopy(once)
        twice = _attach_scip_to_hierarchy(once, _FakeStore(db), "local", "proj")
        assert twice == first
