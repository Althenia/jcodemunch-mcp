"""Regression test for #364: the ``license`` CLI must read a config-file key.

The ``license`` subcommand handler returns before the shared ``load_config()``
call further down in ``main()``. Before v1.108.121 that meant a ``license_key``
persisted in ``~/.code-index/config.jsonc`` was never loaded, so
``jcodemunch-mcp license`` reported "unlicensed evaluation (no license key)"
despite a valid key on disk — only ``JCODEMUNCH_LICENSE_KEY`` / ``--key`` worked.

This exercises the CLI end-to-end against an isolated ``CODE_INDEX_PATH`` store
containing a config.jsonc with a ``license_key``, and asserts the key is read
(surfaced as ``key_masked`` in the ``--json`` gate). The fake key won't validate
server-side, so the mode is ``grace`` — the point is that the key is *seen*, not
that it licenses.
"""
import json
import os
import subprocess
import sys


def _run(args, storage):
    env = {
        **os.environ,
        "CODE_INDEX_PATH": str(storage),
    }
    # Never let a real env key bleed into the isolated run.
    env.pop("JCODEMUNCH_LICENSE_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "jcodemunch_mcp", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


def test_license_cli_reads_config_key(tmp_path):
    storage = tmp_path / "store"
    storage.mkdir()
    (storage / "config.jsonc").write_text(
        '{\n  // #364 regression\n  "license_key": "TEST-KEY-1234-ABCD"\n}\n',
        encoding="utf-8",
    )

    res = _run(["license", "--json"], storage)
    assert res.returncode == 0, res.stderr
    gate = json.loads(res.stdout)
    # The key from config.jsonc must be visible (masked) — the bug returned no key.
    assert gate.get("key_masked"), f"config-file license_key was not read: {gate}"
    assert gate["key_masked"].startswith("TEST")


def test_license_cli_no_key_reports_no_key(tmp_path):
    """Control: an empty store still honestly reports no key."""
    storage = tmp_path / "store"
    storage.mkdir()

    res = _run(["license", "--json"], storage)
    assert res.returncode == 0, res.stderr
    gate = json.loads(res.stdout)
    assert not gate.get("key_masked"), gate
