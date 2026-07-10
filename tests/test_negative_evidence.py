"""Tests for negative evidence in search_symbols (Feature 1)."""

from pathlib import Path

from tests.conftest_helpers import create_mini_index


class TestNegativeEvidence:
    """Tests for negative_evidence field in search results."""

    def test_negative_evidence_on_empty_results(self, tmp_path: Path):
        """When no matches found, negative_evidence is present with verdict."""
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        repo, storage_path = create_mini_index(tmp_path)
        result = search_symbols(
            repo=repo,
            query="nonexistent_xyz_redis_cache",
            storage_path=storage_path,
        )
        assert "negative_evidence" in result
        ne = result["negative_evidence"]
        assert ne["verdict"] == "no_implementation_found"
        assert ne["scanned_symbols"] > 0
        assert ne["scanned_files"] > 0
        assert "best_match_score" in ne

    def test_no_negative_evidence_on_strong_match(self, tmp_path: Path):
        """When strong match found, negative_evidence is NOT present."""
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        repo, storage_path = create_mini_index(tmp_path)
        result = search_symbols(repo=repo, query="my_func", storage_path=storage_path)
        assert result["result_count"] >= 1
        assert "negative_evidence" not in result

    def test_related_existing_files(self, tmp_path: Path):
        """When query matches file name but not symbol, related_existing shows nearby files."""
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        repo, storage_path = create_mini_index(tmp_path, filename="auth_handler.py")
        result = search_symbols(
            repo=repo,
            query="auth_nonexistent",
            storage_path=storage_path,
        )
        # Should have negative_evidence since no symbol matches
        assert "negative_evidence" in result
        # related_existing should mention the auth file
        ne = result["negative_evidence"]
        assert "related_existing" in ne
        # The auth_handler.py file should be mentioned
        related = ne["related_existing"]
        assert any("auth" in f.lower() for f in related)

    def test_threshold_constant_importable(self):
        """_NEGATIVE_EVIDENCE_THRESHOLD constant exists and is > 0."""
        from jcodemunch_mcp.tools.search_symbols import _NEGATIVE_EVIDENCE_THRESHOLD
        assert _NEGATIVE_EVIDENCE_THRESHOLD > 0
        assert _NEGATIVE_EVIDENCE_THRESHOLD < 10  # Reasonable range

    def test_low_confidence_match_shows_negative_evidence(self, tmp_path: Path):
        """When match score is below threshold, negative_evidence is present."""
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        repo, storage_path = create_mini_index(tmp_path)
        # Search for something that partially matches but shouldn't be a strong match
        result = search_symbols(
            repo=repo,
            query="xyz_my_func_abc",  # Contains my_func but not exact
            storage_path=storage_path,
        )
        # If the score is below threshold, should have negative_evidence
        # If above threshold (fuzzy match worked), should have results
        if result.get("result_count", 0) == 0 or result.get("negative_evidence"):
            ne = result.get("negative_evidence", {})
            assert "verdict" in ne


class TestGetRankedContextNegativeEvidence:
    """Tests for negative_evidence field in get_ranked_context."""

    def test_negative_evidence_on_empty_result(self, tmp_path: Path):
        """get_ranked_context returns negative_evidence when query doesn't match."""
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def real_function(): pass\n")

        storage = str(tmp_path / "idx")
        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)
        repo = idx["repo"]

        result = get_ranked_context(
            repo=repo,
            query="completely_nonexistent_feature_xyz",
            token_budget=4000,
            storage_path=storage,
        )

        assert result["items_included"] == 0
        assert "negative_evidence" in result
        assert result["negative_evidence"]["verdict"] == "no_implementation_found"
        assert result["negative_evidence"]["scanned_symbols"] >= 0
        assert result["negative_evidence"]["best_match_score"] == 0.0

    def test_warning_string_present_on_negative_evidence(self, tmp_path: Path):
        """Verify warning string at top level when negative evidence triggers."""
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def real_function(): pass\n")

        storage = str(tmp_path / "idx")
        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)
        repo = idx["repo"]

        result = get_ranked_context(
            repo=repo,
            query="nonexistent_xyz",
            token_budget=4000,
            storage_path=storage,
        )

        assert "\u26a0 warning" in result
        assert "nonexistent_xyz" in result["\u26a0 warning"]

    def test_no_negative_evidence_on_strong_match(self, tmp_path: Path):
        """No negative_evidence when query matches well."""
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def my_special_function():\n    return 42\n")

        storage = str(tmp_path / "idx")
        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)
        repo = idx["repo"]

        result = get_ranked_context(
            repo=repo,
            query="my_special_function",
            token_budget=4000,
            storage_path=storage,
        )

        assert result["items_included"] > 0
        assert "negative_evidence" not in result
        assert "\u26a0 warning" not in result

    def test_related_existing_in_negative_evidence(self, tmp_path: Path):
        """related_existing lists files with partial name match."""
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context

        src = tmp_path / "src"
        src.mkdir()
        (src / "auth_handler.py").write_text("def login(): pass\n")

        storage = str(tmp_path / "idx")
        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)
        repo = idx["repo"]

        result = get_ranked_context(
            repo=repo,
            query="auth_nonexistent_feature",
            token_budget=4000,
            storage_path=storage,
        )

        ne = result.get("negative_evidence", {})
        # "auth" is in query and in filename "auth_handler.py"
        if ne.get("related_existing"):
            assert any("auth" in f for f in ne["related_existing"])


class TestSearchSymbolsWarningString:
    """Tests for warning string alongside negative_evidence in search_symbols."""

    def test_warning_on_negative_evidence(self, tmp_path: Path):
        """search_symbols includes warning string when negative evidence triggers."""
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.search_symbols import search_symbols

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def real_function(): pass\n")

        storage = str(tmp_path / "idx")
        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)
        repo = idx["repo"]

        result = search_symbols(
            repo=repo,
            query="nonexistent_xyz_feature",
            storage_path=storage,
        )

        assert "negative_evidence" in result
        assert "\u26a0 warning" in result
        assert "nonexistent_xyz" in result["\u26a0 warning"]

    def test_no_warning_on_good_match(self, tmp_path: Path):
        """No warning string when results are strong."""
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.search_symbols import search_symbols

        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def real_function(): pass\n")

        storage = str(tmp_path / "idx")
        idx = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)
        repo = idx["repo"]

        result = search_symbols(
            repo=repo,
            query="real_function",
            storage_path=storage,
        )

        assert "\u26a0 warning" not in result


class TestBuildVerdictUnit:
    """Pure-function coverage of the unified retrieval verdict (retrieval/verdict.py)."""

    def _bv(self, **kw):
        from jcodemunch_mcp.retrieval.verdict import build_verdict
        return build_verdict(**kw)

    def test_state_ok(self):
        v = self._bv(result_count=3, best_score=2.0, threshold=0.5,
                     scanned_symbols=100, scanned_files=10)["verdict"]
        assert v["state"] == "ok"
        assert v["scanned"] == {"symbols": 100, "files": 10}
        assert v["channels"]["index"] == "fresh"

    def test_state_absent_and_legacy_parity(self):
        res = self._bv(result_count=0, best_score=0.0, threshold=0.5,
                       scanned_symbols=100, scanned_files=10)
        assert res["verdict"]["state"] == "absent"
        # legacy negative_evidence still fires with the historical verdict name
        assert res["negative_evidence"]["verdict"] == "no_implementation_found"
        assert res["negative_evidence"]["scanned_symbols"] == 100

    def test_state_low_confidence(self):
        res = self._bv(result_count=2, best_score=0.2, threshold=0.5,
                       scanned_symbols=50, scanned_files=5)
        assert res["verdict"]["state"] == "low_confidence"
        assert res["negative_evidence"]["verdict"] == "low_confidence_matches"

    def test_state_degraded_on_timeout(self):
        # A cut-short scan cannot prove absence, even with zero results.
        v = self._bv(result_count=0, scanned_files=3, timed_out=True)["verdict"]
        assert v["state"] == "degraded"

    def test_state_degraded_on_semantic_unavailable(self, monkeypatch):
        import jcodemunch_mcp.retrieval.verdict as vmod
        monkeypatch.setattr(vmod, "_semantic_provider_available", lambda: False)
        v = self._bv(result_count=4, best_score=1.0, threshold=0.5,
                     semantic_requested=True)["verdict"]
        assert v["state"] == "degraded"
        assert v["channels"]["semantic"] == "unavailable"

    def test_semantic_channel_off_when_not_requested(self):
        v = self._bv(result_count=1, best_score=1.0, threshold=0.5)["verdict"]
        assert v["channels"]["semantic"] == "off"

    def test_did_you_mean_from_source_files(self):
        v = self._bv(result_count=0, threshold=0.5, best_score=0.0,
                     query_terms=["auth"],
                     source_files=["src/auth_handler.py", "src/other.py"])["verdict"]
        assert v.get("did_you_mean") == ["src/auth_handler.py"]

    def test_index_stale_reflected(self):
        v = self._bv(result_count=1, best_score=1.0, threshold=0.5,
                     index_stale=True)["verdict"]
        assert v["channels"]["index"] == "stale"


class TestVerdictOnTools:
    """The unified _meta.verdict is present across the retrieval tools."""

    def test_search_symbols_meta_verdict(self, tmp_path: Path):
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.search_symbols import search_symbols
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def my_special_function():\n    return 42\n")
        storage = str(tmp_path / "idx")
        repo = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)["repo"]

        hit = search_symbols(repo=repo, query="my_special_function", storage_path=storage)
        assert hit["_meta"]["verdict"]["state"] == "ok"

        miss = search_symbols(repo=repo, query="totally_absent_xyz", storage_path=storage)
        assert miss["_meta"]["verdict"]["state"] == "absent"

    def test_search_text_meta_verdict(self, tmp_path: Path):
        from jcodemunch_mcp.tools.index_folder import index_folder
        from jcodemunch_mcp.tools.search_text import search_text
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("MAGIC_TOKEN = 'present'\n")
        storage = str(tmp_path / "idx")
        repo = index_folder(path=str(tmp_path), use_ai_summaries=False, storage_path=storage)["repo"]

        hit = search_text(repo=repo, query="MAGIC_TOKEN", storage_path=storage)
        assert hit["_meta"]["verdict"]["state"] == "ok"

        miss = search_text(repo=repo, query="no_such_string_zzz", storage_path=storage)
        assert miss["_meta"]["verdict"]["state"] == "absent"


class TestFileToolVerdictUnit:
    """Pure-function coverage of the file/symbol verdict helpers (Phase 2)."""

    def test_suggest_paths_exact_basename_first(self):
        from jcodemunch_mcp.retrieval.verdict import suggest_paths
        # Right filename, wrong directory -> exact basename ranks first.
        out = suggest_paths(
            "lib/auth.py",
            ["src/auth.py", "src/authz_helper.py", "src/other.py"],
        )
        assert out[0] == "src/auth.py"
        assert "src/authz_helper.py" in out  # stem substring, ranked after

    def test_suggest_paths_normalizes_separators_and_excludes_self(self):
        from jcodemunch_mcp.retrieval.verdict import suggest_paths
        out = suggest_paths("src\\auth.py", ["src/auth.py"])
        assert out == []  # identical path is never its own suggestion

    def test_suggest_paths_empty_inputs(self):
        from jcodemunch_mcp.retrieval.verdict import suggest_paths
        assert suggest_paths(None, ["a.py"]) == []
        assert suggest_paths("a.py", None) == []

    def test_suggest_symbol_ids_same_name_first(self):
        from jcodemunch_mcp.retrieval.verdict import suggest_symbol_ids
        symbols = [
            {"id": "src/a.py::login#function", "name": "login"},
            {"id": "src/b.py::login_user#function", "name": "login_user"},
            {"id": "src/c.py::logout#function", "name": "logout"},
        ]
        out = suggest_symbol_ids("wrong/path.py::login#function", symbols)
        assert out[0] == "src/a.py::login#function"
        assert "src/b.py::login_user#function" in out
        assert "src/c.py::logout#function" not in out

    def test_build_file_verdict_absent_with_suggestions(self):
        from jcodemunch_mcp.retrieval.verdict import build_file_verdict
        v = build_file_verdict(
            present=False,
            requested_path="lib/auth.py",
            source_files=["src/auth.py"],
        )
        assert v["state"] == "absent"
        assert v["did_you_mean"] == ["src/auth.py"]

    def test_build_file_verdict_empty_symbols_is_absent_no_suggestions(self):
        from jcodemunch_mcp.retrieval.verdict import build_file_verdict
        v = build_file_verdict(present=True, empty_symbols=True)
        assert v["state"] == "absent"
        assert "did_you_mean" not in v

    def test_build_file_verdict_ok(self):
        from jcodemunch_mcp.retrieval.verdict import build_file_verdict
        v = build_file_verdict(present=True)
        assert v["state"] == "ok"

    def test_build_symbol_verdict_absent_and_ok(self):
        from jcodemunch_mcp.retrieval.verdict import build_symbol_verdict
        symbols = [{"id": "src/a.py::login#function", "name": "login"}]
        absent = build_symbol_verdict(
            found_count=0, requested_id="x::login#function", symbols=symbols
        )
        assert absent["state"] == "absent"
        assert absent["did_you_mean"] == ["src/a.py::login#function"]
        ok = build_symbol_verdict(found_count=2)
        assert ok["state"] == "ok"
        assert "did_you_mean" not in ok


class TestFileVerdictOnTools:
    """The _meta.verdict is emitted by the file/symbol read tools (Phase 2)."""

    def _index(self, tmp_path: Path):
        from jcodemunch_mcp.tools.index_folder import index_folder
        src = tmp_path / "src"
        src.mkdir()
        (src / "auth_handler.py").write_text(
            "def login():\n    return True\n"
        )
        (src / "empty.json").write_text('{"k": 1}\n')
        storage = str(tmp_path / "idx")
        repo = index_folder(
            path=str(tmp_path), use_ai_summaries=False, storage_path=storage
        )["repo"]
        return repo, storage

    def test_get_file_content_absent_suggests_path(self, tmp_path: Path):
        from jcodemunch_mcp.tools.get_file_content import get_file_content
        repo, storage = self._index(tmp_path)
        res = get_file_content(
            repo=repo, file_path="lib/auth_handler.py", storage_path=storage
        )
        assert "error" in res
        v = res["_meta"]["verdict"]
        assert v["state"] == "absent"
        assert any("auth_handler.py" in p for p in v.get("did_you_mean", []))

    def test_get_file_content_ok(self, tmp_path: Path):
        from jcodemunch_mcp.tools.get_file_content import get_file_content
        repo, storage = self._index(tmp_path)
        res = get_file_content(
            repo=repo, file_path="src/auth_handler.py", storage_path=storage
        )
        assert res["_meta"]["verdict"]["state"] == "ok"

    def test_get_file_outline_absent_path(self, tmp_path: Path):
        from jcodemunch_mcp.tools.get_file_outline import get_file_outline
        repo, storage = self._index(tmp_path)
        res = get_file_outline(
            repo=repo, file_path="nope/auth_handler.py", storage_path=storage
        )
        assert res["_meta"]["verdict"]["state"] == "absent"
        assert any("auth_handler.py" in p for p in res["_meta"]["verdict"].get("did_you_mean", []))

    def test_get_file_outline_present_ok(self, tmp_path: Path):
        from jcodemunch_mcp.tools.get_file_outline import get_file_outline
        repo, storage = self._index(tmp_path)
        res = get_file_outline(
            repo=repo, file_path="src/auth_handler.py", storage_path=storage
        )
        assert res["_meta"]["verdict"]["state"] == "ok"

    def test_get_symbol_source_absent_suggests_id(self, tmp_path: Path):
        from jcodemunch_mcp.tools.get_symbol import get_symbol_source
        repo, storage = self._index(tmp_path)
        res = get_symbol_source(
            repo=repo, symbol_id="wrong/file.py::login#function", storage_path=storage
        )
        assert "error" in res
        v = res["_meta"]["verdict"]
        assert v["state"] == "absent"
        assert any("login" in sid for sid in v.get("did_you_mean", []))

    def test_get_symbol_source_ok(self, tmp_path: Path):
        from jcodemunch_mcp.tools.get_file_outline import get_file_outline
        from jcodemunch_mcp.tools.get_symbol import get_symbol_source
        repo, storage = self._index(tmp_path)
        outline = get_file_outline(
            repo=repo, file_path="src/auth_handler.py", storage_path=storage
        )
        sid = outline["symbols"][0]["id"]
        res = get_symbol_source(repo=repo, symbol_id=sid, storage_path=storage)
        assert res["_meta"]["verdict"]["state"] == "ok"