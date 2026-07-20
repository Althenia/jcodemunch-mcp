"""Self-attesting retrieval contract, Phase A (v1.108.138).

Every confidence number the suite emits states its basis — ``declared``
(engineering prior) or ``measured`` (committed benchmark artifact) — and a
prior is never presented as a measurement. These tests are the drift guards:

- registry ``value`` == the live constant in the emitting module (code and
  registry can't drift apart)
- registry ``measured`` entries == benchmarks/provenance/measured.json
- measured.json == the underlying artifacts it cites (replay golden, the
  methodology doc)
- live responses validate against the published JSON Schemas in schemas/
"""

import json
from pathlib import Path

import jsonschema
import pytest

from jcodemunch_mcp.retrieval import provenance
from jcodemunch_mcp.retrieval.provenance import (
    BASIS_DECLARED,
    BASIS_MEASURED,
    CONFIDENCE_PROVENANCE,
    MEASURED,
    channel_provenance,
)

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMAS = _ROOT / "schemas"
_MEASURED_JSON = _ROOT / "benchmarks" / "provenance" / "measured.json"


def _schema(name: str) -> dict:
    return json.loads((_SCHEMAS / name).read_text(encoding="utf-8"))


class TestRegistryHygiene:
    def test_every_entry_has_value_and_valid_basis(self):
        for key, entry in CONFIDENCE_PROVENANCE.items():
            assert "value" in entry, key
            assert entry["basis"] in (BASIS_DECLARED, BASIS_MEASURED), key

    def test_measured_entries_carry_source(self):
        for key, entry in MEASURED.items():
            assert entry["basis"] == BASIS_MEASURED, key
            assert entry["source"], key


class TestDeclaredConstantsMatchCode:
    """A registry value that drifts from the live constant fails loudly."""

    def test_find_implementations_channels(self):
        from jcodemunch_mcp.tools import find_implementations as fi
        expected = {
            "find_implementations.lsp": fi._CONF_LSP,
            "find_implementations.scip": fi._CONF_SCIP,
            "find_implementations.ast": fi._CONF_AST,
            "find_implementations.duck": fi._CONF_DUCK,
            "find_implementations.decorator": fi._CONF_DECORATOR,
        }
        for key, live in expected.items():
            assert CONFIDENCE_PROVENANCE[key]["value"] == live, key

    def test_negative_evidence_threshold(self):
        from jcodemunch_mcp.tools.search_symbols import _NEGATIVE_EVIDENCE_THRESHOLD
        assert (
            CONFIDENCE_PROVENANCE["retrieval.negative_evidence_threshold"]["value"]
            == _NEGATIVE_EVIDENCE_THRESHOLD
        )

    def test_exact_seed_verdict_floor(self):
        from jcodemunch_mcp.tools.get_ranked_context import _EXACT_SEED_VERDICT_SCORE
        assert (
            CONFIDENCE_PROVENANCE["get_ranked_context.exact_seed_verdict_floor"]["value"]
            == _EXACT_SEED_VERDICT_SCORE
        )


class TestMeasuredArtifactChain:
    """Registry -> measured.json -> underlying benchmark artifacts."""

    def test_registry_matches_measured_json(self):
        artifact = json.loads(_MEASURED_JSON.read_text(encoding="utf-8"))
        tr = artifact["token_reduction"]
        reg = MEASURED["token_reduction"]
        assert reg["average_pct"] == tr["average_pct"]
        assert reg["task_runs"] == tr["task_runs"]
        assert reg["tokenizer"] == tr["tokenizer"]
        rq = artifact["replay_retrieval_quality"]
        reg = MEASURED["replay_retrieval_quality"]
        for k in ("fixture", "k", "ndcg", "mrr", "recall"):
            assert reg[k] == rq[k], k

    def test_measured_json_matches_replay_golden(self):
        artifact = json.loads(_MEASURED_JSON.read_text(encoding="utf-8"))
        rq = artifact["replay_retrieval_quality"]
        golden = json.loads((_ROOT / rq["source"]).read_text(encoding="utf-8"))
        assert golden["fixture"] == rq["fixture"]
        assert golden["k"] == rq["k"]
        assert golden["captured_at"] == rq["captured_at"]
        assert golden["version"] == rq["generator_version"]
        overall = golden["overall"]
        for k in ("ndcg", "mrr", "recall"):
            assert overall[k] == rq[k], k
        assert len(golden["per_query"]) == rq["queries"]

    def test_measured_json_figures_appear_in_methodology(self):
        artifact = json.loads(_MEASURED_JSON.read_text(encoding="utf-8"))
        tr = artifact["token_reduction"]
        doc = (_ROOT / tr["methodology"]).read_text(encoding="utf-8")
        assert str(tr["average_pct"]) in doc
        assert f"{tr['baseline_tokens']:,}" in doc
        assert f"{tr['jcodemunch_tokens']:,}" in doc


class TestPublishedSchemas:
    """Live responses validate against the committed schemas/ contracts."""

    @staticmethod
    def _make_repo(tmp_path, files):
        from jcodemunch_mcp.tools.index_folder import index_folder
        for rel, content in files.items():
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        storage = str(tmp_path / ".index")
        result = index_folder(str(tmp_path), use_ai_summaries=False, storage_path=storage)
        return result.get("repo", str(tmp_path)), storage

    _REPO = {
        "shapes.py": (
            "class Shape:\n"
            "    def area(self):\n"
            "        raise NotImplementedError\n\n"
            "class Circle(Shape):\n"
            "    def area(self):\n"
            "        return 3\n"
        ),
        "util.py": "def area_report(s):\n    return s.area()\n",
    }

    def test_confidence_provenance_block_validates(self, tmp_path):
        from jcodemunch_mcp.tools.find_implementations import find_implementations
        repo, storage = self._make_repo(tmp_path, self._REPO)
        result = find_implementations(repo, "Shape", storage_path=storage)
        assert "error" not in result
        block = result["_meta"]["confidence_provenance"]
        jsonschema.validate(block, _schema("confidence-provenance.schema.json"))
        assert block["channels"]["ast"]["basis"] == BASIS_DECLARED

    def test_channel_provenance_helper_validates(self):
        jsonschema.validate(
            channel_provenance("find_implementations"),
            _schema("confidence-provenance.schema.json"),
        )

    def test_ranked_context_response_validates(self, tmp_path):
        from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context
        repo, storage = self._make_repo(tmp_path, self._REPO)
        schema = _schema("ranked-context-response.schema.json")
        verdict_schema = _schema("retrieval-verdict.schema.json")
        for query in ("Shape.area", "how are areas reported"):
            result = get_ranked_context(repo, query, token_budget=4000, storage_path=storage)
            assert "error" not in result
            jsonschema.validate(result, schema)
            jsonschema.validate(result["_meta"]["verdict"], verdict_schema)

    def test_absent_response_validates(self, tmp_path):
        from jcodemunch_mcp.tools.get_ranked_context import get_ranked_context
        repo, storage = self._make_repo(tmp_path, self._REPO)
        result = get_ranked_context(
            repo, "quantum flux capacitor telemetry", token_budget=4000, storage_path=storage,
        )
        jsonschema.validate(result, _schema("ranked-context-response.schema.json"))

    def test_schemas_are_valid_jsonschema(self):
        validator = jsonschema.Draft202012Validator
        for f in sorted(_SCHEMAS.glob("*.schema.json")):
            validator.check_schema(json.loads(f.read_text(encoding="utf-8")))
        assert len(list(_SCHEMAS.glob("*.schema.json"))) >= 3


class TestContractCulture:
    def test_no_measured_basis_without_artifact_entry(self):
        """A registry constant may claim 'measured' only if measured.json backs it.

        This is the load-bearing rule of the whole contract: a prior is never
        presented as a measurement. Flipping a channel's basis to 'measured'
        requires adding its gold-corpus results to the committed artifact
        first, or this test fails the build.
        """
        artifact = json.loads(_MEASURED_JSON.read_text(encoding="utf-8"))
        offenders = [
            key for key, entry in CONFIDENCE_PROVENANCE.items()
            if entry["basis"] == BASIS_MEASURED
            and key.replace(".", "_") not in artifact
            and entry.get("source_key", "") not in artifact
        ]
        assert not offenders, (
            f"claim basis=measured without a backing entry in "
            f"{provenance.MEASURED_ARTIFACT}: {offenders}"
        )
