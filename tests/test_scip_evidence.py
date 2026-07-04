"""Tests for SCIP compile-time evidence ingestion (v1.108.96).

Fixture SCIP indexes are built in-test with a tiny protobuf writer (varint +
length-delimited emitters) rather than checked-in binary blobs — readable,
deterministic, and exercises both packed and unpacked range encodings.
"""

import copy
import gzip
import sqlite3
from pathlib import Path

import pytest

from jcodemunch_mcp.evidence.scip import (
    SYMBOL_ROLE_DEFINITION,
    SYMBOL_ROLE_IMPORT,
    display_name_from_symbol,
    is_local_symbol,
    parse_scip_bytes,
    parse_scip_file,
)
from jcodemunch_mcp.evidence.scip_ingest import (
    _ensure_scip_tables,
    ingest_scip_file,
)
from jcodemunch_mcp.tools.find_references import _attach_scip_to_response
from jcodemunch_mcp.tools.import_scip import import_scip


# ── Minimal protobuf writer ─────────────────────────────────────────────


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field_no: int, wire_type: int) -> bytes:
    return _varint((field_no << 3) | wire_type)


def _ld(field_no: int, payload: bytes) -> bytes:
    return _tag(field_no, 2) + _varint(len(payload)) + payload


def _vi(field_no: int, value: int) -> bytes:
    return _tag(field_no, 0) + _varint(value)


def _occurrence(range_vals, symbol: str, roles: int = 0, packed: bool = True) -> bytes:
    if packed:
        buf = _ld(1, b"".join(_varint(v) for v in range_vals))
    else:
        buf = b"".join(_vi(1, v) for v in range_vals)
    buf += _ld(2, symbol.encode())
    if roles:
        buf += _vi(3, roles)
    return buf


def _symbol_info(symbol: str, implements: str = "") -> bytes:
    buf = _ld(1, symbol.encode())
    if implements:
        rel = _ld(1, implements.encode()) + _vi(3, 1)  # is_implementation=true
        buf += _ld(4, rel)
    return buf


def _document(path: str, occurrences=(), symbol_infos=()) -> bytes:
    buf = _ld(1, path.encode())
    for occ in occurrences:
        buf += _ld(2, occ)
    for info in symbol_infos:
        buf += _ld(3, info)
    return buf


def _metadata(tool: str = "scip-test", version: str = "9.9", root: str = "file:///proj") -> bytes:
    tool_info = _ld(1, tool.encode()) + _ld(2, version.encode())
    return _ld(2, tool_info) + _ld(3, root.encode())


def _index(documents=(), metadata: bytes = None, external_symbols=()) -> bytes:
    buf = b""
    if metadata is None:
        metadata = _metadata()
    buf += _ld(1, metadata)
    for doc in documents:
        buf += _ld(2, doc)
    for info in external_symbols:
        buf += _ld(3, info)
    return buf


# SCIP symbol strings for the fixture repo
CALLEE_SYM = "scip-python pypi proj 1.0 src/`b.py`/callee()."
IMPL_SYM = "scip-python pypi proj 1.0 src/`impl.py`/Impl#"
IFACE_SYM = "scip-python pypi proj 1.0 src/`iface.py`/Iface#"
EXTERNAL_SYM = "scip-python pypi requests 2.31 requests/`api.py`/get()."


def _fixture_index_bytes(packed: bool = True) -> bytes:
    """Two-file repo: a.py's `caller` references b.py's `callee`;
    Impl implements Iface; plus import/local/external occurrences that
    must be skipped or recorded unmapped."""
    doc_b = _document("src/b.py", occurrences=[
        _occurrence([4, 0, 4, 6], CALLEE_SYM, roles=SYMBOL_ROLE_DEFINITION, packed=packed),
    ])
    doc_a = _document("src/a.py", occurrences=[
        _occurrence([11, 4, 11, 10], CALLEE_SYM, packed=packed),               # the real reference
        _occurrence([0, 0, 0, 6], CALLEE_SYM, roles=SYMBOL_ROLE_IMPORT, packed=packed),  # import → skipped
        _occurrence([12, 0, 12], "local 3", packed=packed),                    # local → skipped (3-int range)
        _occurrence([13, 0, 13, 3], EXTERNAL_SYM, packed=packed),              # external → unmapped
    ])
    doc_impl = _document(
        "src/impl.py",
        occurrences=[_occurrence([0, 6, 0, 10], IMPL_SYM, roles=SYMBOL_ROLE_DEFINITION, packed=packed)],
        symbol_infos=[_symbol_info(IMPL_SYM, implements=IFACE_SYM)],
    )
    doc_iface = _document("src/iface.py", occurrences=[
        _occurrence([0, 6, 0, 11], IFACE_SYM, roles=SYMBOL_ROLE_DEFINITION, packed=packed),
    ])
    return _index(documents=[doc_b, doc_a, doc_impl, doc_iface])


def _make_index_db(db_path: Path) -> None:
    """Minimal index database: the symbols/meta subset the resolver reads."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE symbols (id TEXT PRIMARY KEY, name TEXT, file TEXT, line INTEGER, end_line INTEGER);
    """)
    conn.execute("INSERT INTO meta VALUES ('git_head', 'abc123')")
    conn.executemany(
        "INSERT INTO symbols VALUES (?, ?, ?, ?, ?)",
        [
            ("src/a.py::caller#function", "caller", "src/a.py", 10, 20),
            ("src/b.py::callee#function", "callee", "src/b.py", 5, 15),
            ("src/impl.py::Impl#class", "Impl", "src/impl.py", 1, 30),
            ("src/iface.py::Iface#class", "Iface", "src/iface.py", 1, 20),
        ],
    )
    conn.commit()
    conn.close()


class _FakeSqlite:
    def __init__(self, db_path: Path):
        self._db = db_path

    def _db_path(self, owner: str, name: str) -> Path:
        return self._db


class _FakeStore:
    def __init__(self, db_path: Path):
        self._sqlite = _FakeSqlite(db_path)


# ── Parser ──────────────────────────────────────────────────────────────


class TestScipParser:
    def test_parses_metadata_documents_and_ranges(self):
        index = parse_scip_bytes(_fixture_index_bytes())
        assert index.tool_name == "scip-test"
        assert index.tool_version == "9.9"
        assert index.project_root == "file:///proj"
        assert [d.relative_path for d in index.documents] == [
            "src/b.py", "src/a.py", "src/impl.py", "src/iface.py",
        ]
        ref_occ = index.documents[1].occurrences[0]
        assert ref_occ.symbol == CALLEE_SYM
        assert ref_occ.start_line == 12  # 0-based 11 → 1-based 12

    def test_three_int_range_form(self):
        index = parse_scip_bytes(_fixture_index_bytes())
        local_occ = index.documents[1].occurrences[2]
        assert local_occ.range == (12, 0, 12)
        assert local_occ.start_line == 13

    def test_unpacked_range_encoding(self):
        index = parse_scip_bytes(_fixture_index_bytes(packed=False))
        assert index.documents[1].occurrences[0].range == (11, 4, 11, 10)

    def test_implementation_relationship(self):
        index = parse_scip_bytes(_fixture_index_bytes())
        info = index.documents[2].symbols[0]
        assert info.symbol == IMPL_SYM
        assert info.relationships[0].is_implementation
        assert info.relationships[0].symbol == IFACE_SYM

    def test_gzip_transparent(self, tmp_path):
        path = tmp_path / "index.scip.gz"
        path.write_bytes(gzip.compress(_fixture_index_bytes()))
        index = parse_scip_file(str(path))
        assert len(index.documents) == 4

    def test_non_scip_input_raises(self):
        with pytest.raises(ValueError):
            parse_scip_bytes(b"\x99\x99\x99\x99")
        with pytest.raises(ValueError):
            parse_scip_bytes(b"")

    def test_display_name_extraction(self):
        assert display_name_from_symbol(CALLEE_SYM) == "callee"
        assert display_name_from_symbol(IMPL_SYM) == "Impl"
        assert display_name_from_symbol("local 3") is None
        assert is_local_symbol("local 3")
        assert not is_local_symbol(CALLEE_SYM)


# ── Ingest ──────────────────────────────────────────────────────────────


class TestScipIngest:
    def _ingest_fixture(self, tmp_path, **kwargs):
        db = tmp_path / "local-proj.db"
        _make_index_db(db)
        scip = tmp_path / "index.scip"
        scip.write_bytes(_fixture_index_bytes())
        result = ingest_scip_file(db_path=str(db), file_path=str(scip), **kwargs)
        return db, result

    def test_round_trip_reference_edge(self, tmp_path):
        db, result = self._ingest_fixture(tmp_path)
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scip_edges WHERE kind = 'reference'"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["from_symbol_id"] == "src/a.py::caller#function"
        assert rows[0]["to_symbol_id"] == "src/b.py::callee#function"
        assert rows[0]["count"] == 1
        assert result["reference_edges"] == 1

    def test_implementation_edge(self, tmp_path):
        db, result = self._ingest_fixture(tmp_path)
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM scip_edges WHERE kind = 'implementation'"
        ).fetchone()
        conn.close()
        assert row["from_symbol_id"] == "src/impl.py::Impl#class"
        assert row["to_symbol_id"] == "src/iface.py::Iface#class"
        assert result["implementation_edges"] == 1

    def test_skips_and_unmapped_are_counted(self, tmp_path):
        _db, result = self._ingest_fixture(tmp_path)
        assert result["skipped_import"] == 1
        assert result["skipped_local"] == 1
        assert result["unmapped_reasons"].get("target_unresolved") == 1

    def test_scip_meta_written(self, tmp_path):
        db, _result = self._ingest_fixture(tmp_path)
        conn = sqlite3.connect(str(db))
        meta = dict(conn.execute("SELECT key, value FROM scip_meta").fetchall())
        conn.close()
        assert meta["tool"] == "scip-test 9.9"
        assert meta["git_head"] == "abc123"
        assert meta["ingested_at"]

    def test_reingest_accumulates_counts(self, tmp_path):
        db, _ = self._ingest_fixture(tmp_path)
        scip = tmp_path / "index.scip"
        ingest_scip_file(db_path=str(db), file_path=str(scip))
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT count FROM scip_edges WHERE kind = 'reference'"
        ).fetchone()
        conn.close()
        assert row["count"] == 2  # upsert accumulated, no duplicate row

    def test_fifo_eviction_cap(self, tmp_path):
        db, _ = self._ingest_fixture(tmp_path, max_rows=1)
        conn = sqlite3.connect(str(db))
        n = conn.execute("SELECT COUNT(*) FROM scip_edges").fetchone()[0]
        conn.close()
        assert n <= 1


# ── CLI backend ─────────────────────────────────────────────────────────


class TestImportScipCli:
    def test_success_path(self, tmp_path):
        from jcodemunch_mcp.storage import IndexStore

        store = IndexStore(base_path=str(tmp_path))
        db_path = store._sqlite._db_path("local", "proj")
        _make_index_db(db_path)
        scip = tmp_path / "index.scip"
        scip.write_bytes(_fixture_index_bytes())
        result = import_scip(path=str(scip), repo="local/proj", storage_path=str(tmp_path))
        assert result["success"] is True
        assert result["repo"] == "local/proj"
        assert result["unique_edges"] == 2

    def test_missing_index_db(self, tmp_path):
        result = import_scip(path="whatever.scip", repo="local/nope", storage_path=str(tmp_path))
        assert result["success"] is False
        assert "index database not found" in result["error"]

    def test_non_scip_file_is_honest_error(self, tmp_path):
        from jcodemunch_mcp.storage import IndexStore

        store = IndexStore(base_path=str(tmp_path))
        db_path = store._sqlite._db_path("local", "proj")
        _make_index_db(db_path)
        bogus = tmp_path / "notscip.scip"
        bogus.write_bytes(b"\x99\x99\x99\x99")
        result = import_scip(path=str(bogus), repo="local/proj", storage_path=str(tmp_path))
        assert result["success"] is False
        assert "not a SCIP index" in result["error"]


# ── find_references annotation ──────────────────────────────────────────


def _singular_response(ref_files):
    return {
        "repo": "local/proj",
        "identifier": "callee",
        "reference_count": len(ref_files),
        "references": [
            {"file": f, "matches": [{"specifier": "./b", "names": ["callee"], "match_type": "named"}]}
            for f in ref_files
        ],
        "_meta": {"timing_ms": 1.0},
    }


class TestFindReferencesScipAnnotation:
    def _ingested_db(self, tmp_path):
        db = tmp_path / "local-proj.db"
        _make_index_db(db)
        scip = tmp_path / "index.scip"
        scip.write_bytes(_fixture_index_bytes())
        ingest_scip_file(db_path=str(db), file_path=str(scip))
        return db

    def test_honest_empty_without_scip_tables(self, tmp_path):
        db = tmp_path / "local-proj.db"
        _make_index_db(db)  # no scip tables at all (pre-v17 shape)
        response = _singular_response(["src/x.py"])
        before = copy.deepcopy(response)
        out = _attach_scip_to_response(response, _FakeStore(db), "local", "proj")
        assert out == before

    def test_honest_empty_with_empty_tables(self, tmp_path):
        db = tmp_path / "local-proj.db"
        _make_index_db(db)
        conn = sqlite3.connect(str(db))
        _ensure_scip_tables(conn)
        conn.close()
        response = _singular_response(["src/x.py"])
        before = copy.deepcopy(response)
        out = _attach_scip_to_response(response, _FakeStore(db), "local", "proj")
        assert out == before

    def test_scip_only_row_appended_and_tagged(self, tmp_path):
        db = self._ingested_db(tmp_path)
        response = _singular_response(["src/x.py"])  # import graph missed src/a.py
        out = _attach_scip_to_response(response, _FakeStore(db), "local", "proj")
        scip_rows = [r for r in out["references"] if r.get("source") == "scip"]
        assert len(scip_rows) == 1
        assert scip_rows[0]["file"] == "src/a.py"
        assert scip_rows[0]["verification"] == "compiler_verified"
        assert out["reference_count"] == 2
        assert out["_meta"]["scip"]["scip_only_files"] == 1
        assert out["_meta"]["scip"]["stale"] is False

    def test_existing_ref_gains_verification(self, tmp_path):
        db = self._ingested_db(tmp_path)
        response = _singular_response(["src/a.py"])  # import graph already found it
        out = _attach_scip_to_response(response, _FakeStore(db), "local", "proj")
        assert out["references"][0]["verification"] == "compiler_verified"
        assert out["_meta"]["scip"]["verified_files"] == 1
        assert out["_meta"]["scip"]["scip_only_files"] == 0

    def test_idempotent_across_result_cache(self, tmp_path):
        db = self._ingested_db(tmp_path)
        response = _singular_response(["src/x.py"])
        once = _attach_scip_to_response(response, _FakeStore(db), "local", "proj")
        first = copy.deepcopy(once)
        twice = _attach_scip_to_response(once, _FakeStore(db), "local", "proj")
        assert twice == first  # no duplicate rows, counts unchanged

    def test_batch_mode_annotated(self, tmp_path):
        db = self._ingested_db(tmp_path)
        response = {
            "repo": "local/proj",
            "results": [
                {"identifier": "callee", "reference_count": 0, "references": []},
                {"identifier": "unrelated", "reference_count": 0, "references": []},
            ],
            "_meta": {},
        }
        out = _attach_scip_to_response(response, _FakeStore(db), "local", "proj")
        callee_entry = out["results"][0]
        assert callee_entry["references"][0]["file"] == "src/a.py"
        assert callee_entry["references"][0]["source"] == "scip"
        assert out["results"][1]["references"] == []

    def test_staleness_flagged_after_reindex(self, tmp_path):
        db = self._ingested_db(tmp_path)
        conn = sqlite3.connect(str(db))
        conn.execute("UPDATE meta SET value = 'newhead' WHERE key = 'git_head'")
        conn.commit()
        conn.close()
        response = _singular_response(["src/x.py"])
        out = _attach_scip_to_response(response, _FakeStore(db), "local", "proj")
        assert out["_meta"]["scip"]["stale"] is True
        assert "import-scip" in out["_meta"]["scip"]["note"]


# ── Migration ───────────────────────────────────────────────────────────


class TestV17Migration:
    def test_v16_database_gains_scip_tables_on_connect(self, tmp_path):
        from jcodemunch_mcp.storage.sqlite_store import SQLiteIndexStore

        db_path = tmp_path / "local-migr.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO meta VALUES ('index_version', '16')")
        conn.commit()
        conn.close()

        store = SQLiteIndexStore(base_path=str(tmp_path))
        migrated = store._connect(db_path)
        try:
            version = migrated.execute(
                "SELECT value FROM meta WHERE key = 'index_version'"
            ).fetchone()[0]
            assert version == "17"
            for table in ("scip_edges", "scip_unmapped", "scip_meta"):
                assert migrated.execute(f"PRAGMA table_info({table})").fetchall()
        finally:
            migrated.close()

    def test_index_version_constant_is_17(self):
        from jcodemunch_mcp.storage.index_store import INDEX_VERSION

        assert INDEX_VERSION == 17
