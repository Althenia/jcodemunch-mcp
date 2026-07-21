"""v1.108.153 — tool-surface schema receipt in get_session_stats.

`_tool_surface_stats()` reports the schema token weight of the currently
visible tool surface vs the raw catalog (bytes/4 over the same serialization
the schema-budget baseline uses). Attached to get_session_stats as an
advisory `tool_surface` block; a helper failure never breaks the stats tool.
"""

import json

import pytest

from jcodemunch_mcp import server


class TestToolSurfaceStats:
    def test_shape_and_invariants(self):
        stats = server._tool_surface_stats()
        assert stats["visible_tools"] > 0
        assert stats["catalog_tools"] >= stats["visible_tools"]
        assert stats["schema_tokens_visible"] > 0
        assert stats["schema_tokens_catalog"] >= stats["schema_tokens_visible"]
        assert stats["schema_tokens_avoided"] == (
            stats["schema_tokens_catalog"] - stats["schema_tokens_visible"]
        )
        assert stats["estimator"] == "bytes/4"
        assert stats["surface"]
        assert stats["profile"]

    def test_heaviest_tools_capped_and_sorted(self):
        stats = server._tool_surface_stats(top_n=5)
        heaviest = stats["heaviest_tools"]
        assert 0 < len(heaviest) <= 5
        weights = list(heaviest.values())
        assert weights == sorted(weights, reverse=True)
        assert all(isinstance(w, int) and w > 0 for w in weights)

    def test_counter_surface_avoids_most_of_catalog(self, monkeypatch):
        monkeypatch.setenv("JCODEMUNCH_TOOL_SURFACE", "counter")
        stats = server._tool_surface_stats()
        assert stats["surface"] == "counter"
        # The front door + always-present controls are a small fraction of
        # the ~90-tool catalog.
        assert stats["visible_tools"] < stats["catalog_tools"] / 2
        assert stats["schema_tokens_avoided"] > 0


class TestServerWiring:
    @pytest.mark.asyncio
    async def test_get_session_stats_carries_tool_surface(self):
        res = await server.call_tool("get_session_stats", {"format": "json"})
        body = json.loads(res[0].text)
        ts = body["tool_surface"]
        assert ts["visible_tools"] > 0
        assert ts["schema_tokens_visible"] > 0
        assert ts["estimator"] == "bytes/4"

    @pytest.mark.asyncio
    async def test_helper_failure_never_breaks_stats(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("surface probe failed")

        monkeypatch.setattr(server, "_tool_surface_stats", _boom)
        res = await server.call_tool("get_session_stats", {"format": "json"})
        body = json.loads(res[0].text)
        assert "tool_surface" not in body
        assert "session_tokens_saved" in body
