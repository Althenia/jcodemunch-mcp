"""v1.108.139: Phase B — measured provenance rides the reporting surfaces.

The measured-artifact block (benchmarks/provenance/measured.json via
retrieval/provenance.measured_provenance) attaches where a human reads the
numbers — receipt JSON export, receipt text methodology footer,
get_session_stats, jcodemunch_guide — and stays OFF the hot retrieval path.
"""

import json

from jcodemunch_mcp.retrieval.provenance import (
    MEASURED,
    measured_provenance,
)


class TestMeasuredProvenanceHelper:
    def test_block_mirrors_measured_registry(self):
        block = measured_provenance()
        for key, entry in MEASURED.items():
            assert block[key] == entry, key
        assert "contract" in block and "measured" in block["contract"]

    def test_block_is_a_fresh_copy(self):
        block = measured_provenance()
        block["token_reduction"]["average_pct"] = 0
        assert MEASURED["token_reduction"]["average_pct"] != 0


class TestReceiptSurfaces:
    def test_json_export_carries_provenance(self):
        from jcodemunch_mcp.cli.receipt import aggregate, render_json
        agg = aggregate([])
        payload = json.loads(render_json(agg, model="opus"))
        prov = payload["provenance"]
        assert prov["token_reduction"]["basis"] == "measured"
        assert prov["replay_retrieval_quality"]["basis"] == "measured"
        assert prov["token_reduction"]["source"] == "benchmarks/provenance/measured.json"

    def test_text_render_cites_the_artifact(self):
        from jcodemunch_mcp.cli.receipt import aggregate, render_text
        agg = aggregate([{"tool": "search_symbols", "result_tokens": 100}])
        text = render_text(agg, days=30, model="opus")
        assert "benchmarks/provenance/measured.json" in text
        assert "drift-guarded" in text


class TestSessionStatsSurface:
    def test_session_stats_carries_savings_provenance(self, tmp_path):
        from jcodemunch_mcp.tools.get_session_stats import get_session_stats
        result = get_session_stats(storage_path=str(tmp_path))
        prov = result["savings_provenance"]
        assert prov["token_reduction"]["basis"] == "measured"
        assert "contract" in prov


class TestGuideSurface:
    def test_guide_result_carries_provenance(self):
        import asyncio
        from jcodemunch_mcp import server as srv
        result = asyncio.run(srv.call_tool("jcodemunch_guide", {}))
        text = result[0].text if isinstance(result, list) else result.content[0].text
        payload = json.loads(text)
        prov = payload["provenance"]
        assert prov["token_reduction"]["basis"] == "measured"
        assert prov["replay_retrieval_quality"]["ci_gated"] is True
