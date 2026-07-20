"""Phase C drift guards: the channel-accuracy measurement is re-run in CI.

The committed artifact (benchmarks/provenance/channel_accuracy.json) cannot
diverge from the reproducible measurement — this suite re-executes
benchmarks/goldset/measure.py against the authored corpus and asserts
equality, then chains the registry's measured references to the artifact.
"""

import importlib.util
import json
from pathlib import Path

from jcodemunch_mcp.retrieval.provenance import (
    CONFIDENCE_PROVENANCE,
    MEASURED,
)

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "benchmarks" / "provenance" / "channel_accuracy.json"


def _load_measure_module():
    spec = importlib.util.spec_from_file_location(
        "goldset_measure", _ROOT / "benchmarks" / "goldset" / "measure.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _artifact() -> dict:
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


class TestMeasurementReproduces:
    def test_live_measurement_equals_committed_artifact(self):
        """The strongest guard: CI recomputes the numbers from the corpus."""
        mod = _load_measure_module()
        live = mod.measure()
        committed = _artifact()
        assert live["channels"] == committed["channels"]
        assert live["corpus_version"] == committed["corpus_version"]

    def test_corpus_hash_matches_artifact(self):
        """Corpus bytes changed -> artifact must be regenerated deliberately."""
        mod = _load_measure_module()
        assert mod.corpus_sha256() == _artifact()["corpus_sha256"]


class TestRegistryChainsToArtifact:
    def test_measured_refs_equal_artifact_channels(self):
        committed = _artifact()["channels"]
        for ch in ("ast", "duck", "decorator"):
            ref = CONFIDENCE_PROVENANCE[f"find_implementations.{ch}"]["measured_ref"]
            assert ref["precision"] == committed[ch]["precision"], ch
            assert ref["recall"] == committed[ch]["recall"], ch

    def test_measured_registry_entry_equals_artifact(self):
        committed = _artifact()
        reg = MEASURED["implementation_channel_accuracy"]
        assert reg["corpus"] == committed["corpus_version"]
        for ch, vals in reg["channels"].items():
            assert vals["precision"] == committed["channels"][ch]["precision"], ch
            assert vals["recall"] == committed["channels"][ch]["recall"], ch

    def test_compiler_grade_channels_stay_declared(self):
        """LSP/SCIP are the ground-truth side; they carry no measured_ref."""
        for ch in ("lsp", "scip"):
            entry = CONFIDENCE_PROVENANCE[f"find_implementations.{ch}"]
            assert entry["basis"] == "declared"
            assert "measured_ref" not in entry


class TestResponseCarriesMeasuredRef:
    def test_channel_provenance_includes_measured_ref(self):
        from jcodemunch_mcp.retrieval.provenance import channel_provenance
        block = channel_provenance("find_implementations")
        assert block["channels"]["ast"]["measured_ref"]["precision"] == 0.818
        assert "measured_ref" not in block["channels"]["lsp"]

    def test_block_still_validates_against_published_schema(self):
        import jsonschema
        from jcodemunch_mcp.retrieval.provenance import channel_provenance
        schema = json.loads(
            (_ROOT / "schemas" / "confidence-provenance.schema.json").read_text(encoding="utf-8")
        )
        jsonschema.validate(channel_provenance("find_implementations"), schema)
