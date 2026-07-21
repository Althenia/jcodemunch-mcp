"""CLI tests for the ``surface`` subcommand (v1.108.154).

``surface`` prints the tool-surface schema receipt (the same block
get_session_stats reports as ``tool_surface``) so consumers with no MCP
session — the jMunch Console in particular — can shell it. Covers routing
(``known_commands`` / prepend-serve guard), the ``--json`` body shape, and
that the env-selected surface/profile shape the receipt.
"""
import json
import os
import subprocess
import sys


def _run(args, extra_env=None):
    env = {
        **os.environ,
        "JCODEMUNCH_USE_AI_SUMMARIES": "false",
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "jcodemunch_mcp", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )


def test_surface_json_shape():
    proc = _run(["surface", "--json"])
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["visible_tools"] > 0
    assert body["catalog_tools"] >= body["visible_tools"]
    assert body["schema_tokens_avoided"] == (
        body["schema_tokens_catalog"] - body["schema_tokens_visible"]
    )
    assert body["estimator"] == "bytes/4"
    assert body["heaviest_tools"]


def test_surface_human_output():
    proc = _run(["surface"])
    assert proc.returncode == 0, proc.stderr
    assert "Schema tokens avoided:" in proc.stdout
    assert "Heaviest tool schemas:" in proc.stdout


def test_surface_honors_env_surface():
    proc = _run(["surface", "--json"], {"JCODEMUNCH_TOOL_SURFACE": "counter"})
    assert proc.returncode == 0, proc.stderr
    body = json.loads(proc.stdout)
    assert body["surface"] == "counter"
    assert body["visible_tools"] < body["catalog_tools"] / 2
    assert body["schema_tokens_avoided"] > 0
