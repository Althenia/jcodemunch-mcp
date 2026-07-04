"""import-scip CLI backend: ingest a SCIP index file for a repo.

The compile-time sibling of ``import_runtime_signal``. Resolves the target
repo the same way, then hands off to ``evidence.scip_ingest.ingest_scip_file``.
"""

import os
from pathlib import Path
from typing import Any, Optional

from ..evidence.scip_ingest import DEFAULT_SCIP_MAX_ROWS, ingest_scip_file
from ..storage import IndexStore
from .resolve_repo import resolve_repo


def _scip_max_rows() -> int:
    raw = os.environ.get("JCODEMUNCH_SCIP_MAX_ROWS", "")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_SCIP_MAX_ROWS
    return value if value != 0 else DEFAULT_SCIP_MAX_ROWS


def import_scip(
    *,
    path: str,
    repo: Optional[str] = None,
    storage_path: Optional[str] = None,
) -> dict[str, Any]:
    """Import a SCIP index file into the scip_* tables for a repo.

    Args:
        path: Path to the ``.scip`` file (``.gz``-compressed accepted).
        repo: Repo identifier as ``owner/name`` or just ``name``. If
            omitted, defaults to resolving the current working directory
            via ``resolve_repo``.
        storage_path: Custom storage path (matches other tools).

    Returns:
        ``{'success': bool, 'repo': '<owner>/<name>', 'records': N,
        'edges_written': E, 'unique_edges': U, 'reference_edges': R,
        'implementation_edges': I, 'unmapped': K, 'unmapped_reasons': {...},
        'skipped_local': L, 'skipped_import': M, 'evicted': X, 'tool': '...'}``
        or ``{'success': False, 'error': '...'}``.
    """
    if repo:
        if "/" in repo:
            owner, name = repo.split("/", 1)
        else:
            owner, name = "local", repo
    else:
        resolved = resolve_repo(path=str(Path.cwd()), storage_path=storage_path)
        if not resolved.get("indexed"):
            return {
                "success": False,
                "error": (
                    "could not resolve current directory to an indexed repo. "
                    "Pass --repo <owner/name> or run `jcodemunch-mcp index .` first."
                ),
            }
        repo_id = resolved["repo"]
        owner, name = repo_id.split("/", 1)

    store = IndexStore(base_path=storage_path)
    db_path = store._sqlite._db_path(owner, name)  # type: ignore[attr-defined]
    if not db_path.exists():
        return {
            "success": False,
            "error": f"index database not found for {owner}/{name}; run `jcodemunch-mcp index` first.",
        }

    try:
        result = ingest_scip_file(
            db_path=str(db_path),
            file_path=path,
            max_rows=_scip_max_rows(),
        )
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}
    except ValueError as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "repo": f"{owner}/{name}",
        **result,
    }
