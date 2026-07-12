"""max_folder_files truncation is loud, persisted, and self-healing (#366).

Before this, hitting the max_folder_files walk cap silently dropped files: the
index looked healthy while whole subdirectories were missing from search, with
nothing in the index result or query responses to indicate it.
"""

import pytest

import jcodemunch_mcp.config as config_module
from jcodemunch_mcp.tools.index_folder import (
    _file_cap_report,
    _attach_cap_report,
    index_folder,
)
from jcodemunch_mcp.tools.resolve_repo import resolve_repo
from jcodemunch_mcp.tools.search_text import search_text
from jcodemunch_mcp.tools.search_symbols import search_symbols
from jcodemunch_mcp.retrieval.verdict import index_truncation_meta


def test_file_cap_report_shape():
    assert _file_cap_report({}, 2000) == {"truncated": False}
    assert _file_cap_report({"file_limit": 0}, 2000) == {"truncated": False}
    rep = _file_cap_report({"file_limit": 150}, 2000)
    assert rep == {
        "truncated": True,
        "files_discovered": 2150,
        "files_indexed": 2000,
        "files_skipped_cap": 150,
        "max_folder_files": 2000,
    }


def test_attach_cap_report_noop_when_not_truncated():
    result = {"success": True}
    _attach_cap_report(result, {"truncated": False})
    assert "truncated" not in result
    assert "warnings" not in result


def test_attach_cap_report_surfaces_when_truncated():
    result = {"success": True}
    _attach_cap_report(
        result,
        {
            "truncated": True,
            "files_discovered": 2150,
            "files_indexed": 2000,
            "files_skipped_cap": 150,
            "max_folder_files": 2000,
        },
    )
    assert result["truncated"] is True
    assert result["files_discovered"] == 2150
    assert result["files_indexed"] == 2000
    assert result["files_skipped_cap"] == 150
    assert any("File cap reached" in w for w in result["warnings"])


def test_index_truncation_meta_helper():
    assert index_truncation_meta(None) is None
    assert index_truncation_meta({"truncated": False}) is None
    block = index_truncation_meta(
        {
            "truncated": True,
            "files_discovered": 2150,
            "files_indexed": 2000,
            "files_skipped_cap": 150,
            "max_folder_files": 2000,
        }
    )
    assert block["truncated"] is True
    assert block["files_skipped_cap"] == 150
    assert "max_folder_files" in block["note"]


@pytest.fixture
def _capped_repo(tmp_path, monkeypatch):
    """Index 5 files under a cap of 3 so 2 are dropped."""
    monkeypatch.setenv("CODE_INDEX_PATH", str(tmp_path / ".idx"))
    monkeypatch.setenv("JCODEMUNCH_USE_AI_SUMMARIES", "false")
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        (src / f"mod{i}.py").write_text(f"def fn{i}():\n    return {i}\n")
    saved = config_module._GLOBAL_CONFIG.get("max_folder_files")
    config_module._GLOBAL_CONFIG["max_folder_files"] = 3
    try:
        res = index_folder(str(src))
        yield str(src), res
    finally:
        if saved is None:
            config_module._GLOBAL_CONFIG.pop("max_folder_files", None)
        else:
            config_module._GLOBAL_CONFIG["max_folder_files"] = saved


def test_index_result_surfaces_truncation(_capped_repo):
    _folder, res = _capped_repo
    assert res.get("truncated") is True
    assert res["files_discovered"] == 5
    assert res["files_indexed"] == 3
    assert res["files_skipped_cap"] == 2
    assert res["file_count"] == 3
    assert any("File cap reached" in w for w in res.get("warnings", []))


def test_resolve_repo_surfaces_truncation(_capped_repo):
    folder, _res = _capped_repo
    rr = resolve_repo(folder)
    assert rr.get("truncated") is True
    assert rr["files_discovered"] == 5
    assert rr["files_indexed"] == 3
    assert "truncation_warning" in rr


def test_search_meta_surfaces_truncation(_capped_repo):
    folder, _res = _capped_repo
    repo = resolve_repo(folder)["repo"]

    st = search_text(repo, "def")
    st_block = st["_meta"].get("index_truncated")
    assert st_block and st_block["truncated"] is True
    assert st_block["files_skipped_cap"] == 2

    ss = search_symbols(repo, "fn")
    ss_block = ss["_meta"].get("index_truncated")
    assert ss_block and ss_block["truncated"] is True


def test_reindex_under_raised_cap_self_heals(_capped_repo):
    folder, _res = _capped_repo
    repo = resolve_repo(folder)["repo"]
    # Raise the cap and re-index: truncation must clear everywhere.
    config_module._GLOBAL_CONFIG["max_folder_files"] = 100
    res2 = index_folder(folder)
    assert not res2.get("truncated")

    rr2 = resolve_repo(folder)
    assert not rr2.get("truncated")
    assert "truncation_warning" not in rr2

    st = search_text(repo, "def")
    assert "index_truncated" not in st["_meta"]
