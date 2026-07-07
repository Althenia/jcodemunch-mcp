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
    sqlite_store = store._sqlite  # type: ignore[attr-defined]
    db_path = sqlite_store._db_path(owner, name)
    if not db_path.exists():
        return {
            "success": False,
            "error": f"index database not found for {owner}/{name}; run `jcodemunch-mcp index` first.",
        }

    # Serialise the scip_edges write against concurrent full/incremental
    # reindexes of the same .db via the indexwrite lock (audit V9). Uses the
    # exact lock_target/storage_root save_index uses so they actually block
    # each other.
    from ..storage import process_locks
    lock_target = f"{owner}/{name}"
    storage_root = str(sqlite_store.base_path)
    try:
        with process_locks.held(
            "indexwrite", lock_target, storage_root, wait_seconds=60.0
        ) as got_lock:
            if not got_lock:
                detail = process_locks.current_holder_diagnostic(
                    "indexwrite", lock_target, storage_root,
                )
                return {
                    "success": False,
                    "error": (
                        f"could not acquire index-write lock for {lock_target} "
                        f"after 60s{detail}"
                    ),
                }
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
