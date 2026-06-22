"""Tests for get_delivery_metrics — durable-change delivery over a window.

Builds real git repos with date-controlled commits (via GIT_*_DATE) so each
bucket — durable / reworked / reverted / revert_authored — and the hub-file
exclusion are deterministically exercised. Same-second commits would have
delta==0 and dodge the `0 < delta` rework check, so dates are set explicitly.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from jcodemunch_mcp.tools.get_delivery_metrics import get_delivery_metrics
from jcodemunch_mcp.tools.index_folder import index_folder


def _git(cwd, *args, date=None):
    env = dict(os.environ)
    if date is not None:
        iso = date.isoformat()
        env["GIT_AUTHOR_DATE"] = iso
        env["GIT_COMMITTER_DATE"] = iso
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, env=env)


def _init(src):
    _git(src, "init")
    _git(src, "config", "user.email", "t@t.com")
    _git(src, "config", "user.name", "T")
    _git(src, "config", "commit.gpgsign", "false")


def _commit(src, files: dict, msg, days_ago):
    date = datetime.now(timezone.utc) - timedelta(days=days_ago)
    for name, content in files.items():
        (src / name).write_text(content)
    _git(src, "add", "-A")
    r = _git(src, "commit", "-m", msg, date=date)
    rc = _git(src, "rev-parse", "HEAD")
    return rc.stdout.strip()


def _index(tmp_path, src):
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    r = index_folder(str(src), use_ai_summaries=False, storage_path=str(store))
    assert r["success"] is True
    return r["repo"], str(store)


def _have_git() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Error / honesty paths (no git needed)
# --------------------------------------------------------------------------- #

def test_missing_repo_errors(tmp_path):
    out = get_delivery_metrics(repo="nope", storage_path=str(tmp_path))
    assert "error" in out


def test_non_git_repo_errors(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text("x = 1\n")
    repo, store = _index(tmp_path, src)
    out = get_delivery_metrics(repo=repo, storage_path=store)
    # Indexed folder exists but has no git tree -> honest error, not zeros.
    assert "error" in out and "git" in out["error"].lower()


# --------------------------------------------------------------------------- #
# Bucket classification
# --------------------------------------------------------------------------- #

@pytest.fixture
def _buckets_repo(tmp_path):
    if not _have_git():
        pytest.skip("git not available")
    src = tmp_path / "src"
    src.mkdir()
    _init(src)
    # c1 durable (f_durable touched once)
    _commit(src, {"f_durable.py": "a = 1\n"}, "add durable thing", days_ago=9)
    # c2 will be reverted
    rev_target = _commit(src, {"f_revert.py": "b = 1\n"}, "add doomed thing", days_ago=8)
    # c3 reverts c2  -> c3 revert_authored, c2 reverted
    date = datetime.now(timezone.utc) - timedelta(days=7)
    r = _git(src, "revert", "--no-edit", rev_target, date=date)
    assert r.returncode == 0, r.stderr
    # c4 add f_rework, then c5 modify it 1 day later -> c4 reworked
    _commit(src, {"f_rework.py": "c = 1\n"}, "add reworkable thing", days_ago=6)
    _commit(src, {"f_rework.py": "c = 2\n"}, "tweak it again", days_ago=5)
    repo, store = _index(tmp_path, src)
    return repo, store


def test_buckets_are_mutually_exclusive_and_correct(_buckets_repo):
    repo, store = _buckets_repo
    out = get_delivery_metrics(repo=repo, window_days=30, rework_horizon_days=14,
                               storage_path=store)
    assert "error" not in out, out
    assert out["commits_total"] == 5
    assert out["commits_reverted"] == 1
    assert out["commits_revert_authored"] == 1
    assert out["commits_reworked"] == 1
    # durable = c1 + c5 (the later rework commit isn't itself re-touched)
    assert out["commits_durable"] == 2
    # buckets partition the total
    assert (out["commits_durable"] + out["commits_reworked"]
            + out["commits_reverted"] + out["commits_revert_authored"]
            == out["commits_total"])


def test_rates_and_assessment(_buckets_repo):
    repo, store = _buckets_repo
    out = get_delivery_metrics(repo=repo, storage_path=store)
    assert 0.0 <= out["durable_rate"] <= 1.0
    assert 0.0 <= out["rework_rate"] <= 1.0
    assert "durable change" in out["assessment"]
    assert out["_meta"]["timing_ms"] >= 0


def test_horizon_zero_disables_rework(_buckets_repo):
    repo, store = _buckets_repo
    out = get_delivery_metrics(repo=repo, window_days=30, rework_horizon_days=0,
                               storage_path=store)
    # With no horizon, the churn-back commit can't be flagged reworked.
    assert out["commits_reworked"] == 0


# --------------------------------------------------------------------------- #
# Hub-file exclusion (the honesty fix)
# --------------------------------------------------------------------------- #

def test_hub_file_does_not_inflate_rework(tmp_path):
    if not _have_git():
        pytest.skip("git not available")
    src = tmp_path / "src"
    src.mkdir()
    _init(src)
    changelog = "# changelog\n"
    # 6 commits, each appends to a shared CHANGELOG (a hub) + a unique file.
    for i in range(6):
        changelog += f"- entry {i}\n"
        _commit(src, {"CHANGELOG.md": changelog, f"feat_{i}.py": f"v = {i}\n"},
                f"ship feature {i}", days_ago=12 - i)
    repo, store = _index(tmp_path, src)
    out = get_delivery_metrics(repo=repo, window_days=30, rework_horizon_days=14,
                               storage_path=store)
    assert "error" not in out, out
    assert out["commits_total"] == 6
    # CHANGELOG is co-touched by all 6 -> hub -> excluded from rework. Each
    # unique feat file is touched once, so nothing is real churn-back.
    assert out["_meta"]["hub_files_excluded"] >= 1
    assert out["commits_durable"] == 6
    assert out["commits_reworked"] == 0


# --------------------------------------------------------------------------- #
# Empty window
# --------------------------------------------------------------------------- #

def test_empty_window_is_honest(tmp_path):
    if not _have_git():
        pytest.skip("git not available")
    src = tmp_path / "src"
    src.mkdir()
    _init(src)
    _commit(src, {"old.py": "x = 1\n"}, "ancient", days_ago=400)
    repo, store = _index(tmp_path, src)
    out = get_delivery_metrics(repo=repo, window_days=30, storage_path=store)
    assert out["commits_total"] == 0
    assert out["commits_durable"] == 0
    assert "No non-merge commits" in out["assessment"]
