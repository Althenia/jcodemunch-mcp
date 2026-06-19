"""Regression tests for v1.108.63 (issues #338, #339).

#338 — `check_edit_safe` / `check_delete_safe` called `check_references` in
singular mode (`identifier=...`, flat response) but iterated the batch-only
`results` key, so identifier content-references were silently counted as zero
and a still-referenced symbol could come back `safe_to_delete`.

#339 — `find_importers(file_paths=[...], cross_repo=true)` passed an empty path
into the package-level cross-repo helper for multi-file batches, producing
misleading/empty cross-repo evidence. The combination now fails closed.
"""

from pathlib import Path

from jcodemunch_mcp.tools.check_delete_safe import check_delete_safe
from jcodemunch_mcp.tools.check_edit_safe import check_edit_safe
from jcodemunch_mcp.tools.find_importers import find_importers
from jcodemunch_mcp.tools.index_folder import index_folder


def _make_repo(tmp_path: Path, files: dict) -> tuple[str, str]:
    for rel, content in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    storage = str(tmp_path / ".index")
    result = index_folder(str(tmp_path), use_ai_summaries=False, storage_path=storage)
    repo_id = result.get("repo", str(tmp_path))
    return repo_id, storage


# render_widget is referenced by NAME in registry.py (a string-dispatch table),
# but registry.py does NOT import widget.py — so the only evidence of use is a
# content reference, exactly the signal the singular/batch shape bug dropped.
_DUCK_TYPED_REPO = {
    "widget.py": "def render_widget(payload):\n    return {'html': payload}\n",
    "registry.py": (
        "# Handlers dispatched by string name (no import of widget.py).\n"
        "HANDLERS = ['render_widget', 'render_footer']\n\n"
        "def dispatch(name, payload):\n"
        "    if name == 'render_widget':\n"
        "        return _invoke('render_widget', payload)\n"
        "    return None\n\n"
        "def _invoke(name, payload):\n"
        "    return name\n"
    ),
}


class TestContentReferencesConsumed338:
    """#338: composite preflights must consume singular content-references."""

    def test_delete_safe_blocks_on_content_reference(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _DUCK_TYPED_REPO)
        result = check_delete_safe(
            repo, symbol="render_widget", cross_repo=False,
            include_runtime=False, storage_path=storage,
        )
        assert "error" not in result, result
        # registry.py references the name → not safe to delete.
        assert result["verdict"] == "internal_uses_blocking", result
        assert result["signals"]["internal_ref_count"] >= 1, result
        assert any(b.get("kind") == "internal_reference" for b in result["blockers"]), result

    def test_edit_safe_counts_content_reference(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _DUCK_TYPED_REPO)
        result = check_edit_safe(repo, symbol="render_widget", storage_path=storage)
        assert "error" not in result, result
        assert result["signals"]["internal_ref_count"] >= 1, result
        assert result["verdict"] != "safe_to_edit", result

    def test_truly_orphan_symbol_still_safe(self, tmp_path):
        # Guard against false positives: a symbol nobody references stays safe.
        repo, storage = _make_repo(tmp_path, {
            "lonely.py": "def nobody_calls_me():\n    return 1\n",
        })
        result = check_delete_safe(
            repo, symbol="nobody_calls_me", cross_repo=False,
            include_runtime=False, storage_path=storage,
        )
        assert "error" not in result, result
        assert result["verdict"] == "safe_to_delete", result
        assert result["signals"]["internal_ref_count"] == 0, result


_CROSS_REPO_REPO = {
    "alpha.py": "def alpha():\n    return 1\n",
    "beta.py": "def beta():\n    return 2\n",
    "consumer.py": "from alpha import alpha\n\ndef use():\n    return alpha()\n",
}


class TestFindImportersCrossRepoBatch339:
    """#339: cross_repo + multi-file batch is unsupported (package-level scope)."""

    def test_multi_file_batch_with_cross_repo_fails_closed(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _CROSS_REPO_REPO)
        result = find_importers(
            repo, file_paths=["alpha.py", "beta.py"],
            cross_repo=True, storage_path=storage,
        )
        assert "error" in result, result
        assert "cross_repo" in result["error"], result
        assert result["_meta"]["cross_repo_scope"] == "package"
        assert result["_meta"]["file_count"] == 2

    def test_single_file_batch_with_cross_repo_is_allowed(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _CROSS_REPO_REPO)
        result = find_importers(
            repo, file_paths=["alpha.py"],
            cross_repo=True, storage_path=storage,
        )
        # Equivalent to the singular cross-repo path — must not error.
        assert "error" not in result, result
        assert "results" in result, result

    def test_multi_file_batch_without_cross_repo_unaffected(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _CROSS_REPO_REPO)
        result = find_importers(
            repo, file_paths=["alpha.py", "beta.py"],
            cross_repo=False, storage_path=storage,
        )
        assert "error" not in result, result
        assert "results" in result, result

    def test_singular_cross_repo_unaffected(self, tmp_path):
        repo, storage = _make_repo(tmp_path, _CROSS_REPO_REPO)
        result = find_importers(
            repo, file_path="alpha.py", cross_repo=True, storage_path=storage,
        )
        assert "error" not in result, result
        assert "importers" in result, result
