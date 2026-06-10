"""Tests for `config set` / `config unset` typed JSONC write-back (v1.108.51).

Covers the general value writer (set_key), the comment/bracket-aware value
scanner, per-type coercion/validation, and the persist-with-rollback paths.
"""
import json

import pytest

from jcodemunch_mcp import config as c


def _seed(tmp_path):
    p = tmp_path / "config.jsonc"
    p.write_text(c.generate_template(), encoding="utf-8")
    return p


def _read(p):
    return json.loads(c._strip_jsonc(p.read_text(encoding="utf-8")))


@pytest.mark.parametrize("key,raw,expect", [
    ("watch", "true", True),
    ("context_providers", "false", False),
    ("staleness_days", "14", 14),
    ("server_output_threshold", "0.25", 0.25),
    ("server_output", "encoded", "encoded"),
    ("path_map", "/old=/new", "/old=/new"),                  # str key, bare string
    ("extra_extensions", '{".mpl":"cpp"}', {".mpl": "cpp"}),   # dict
    ("trusted_folders", '["/srv/a","/srv/b"]', ["/srv/a", "/srv/b"]),  # list
    ("languages", '["python","go"]', ["python", "go"]),        # active multi-line array
    ("meta_fields", "null", None),                             # (list, None) -> null
    ("use_ai_summaries", "auto", "auto"),                      # (bool, str) -> str
    ("use_ai_summaries", "false", False),                      # (bool, str) -> bool wins
])
def test_set_roundtrip(tmp_path, key, raw, expect):
    p = _seed(tmp_path)
    written = c.set_config_value(key, raw, storage_path=str(tmp_path))
    assert written == expect
    assert _read(p).get(key) == expect


def test_set_preserves_following_comments(tmp_path):
    p = _seed(tmp_path)
    c.set_config_value("staleness_days", "30", storage_path=str(tmp_path))
    text = p.read_text(encoding="utf-8")
    assert '"staleness_days": 30,' in text
    # the descriptive comment that follows the template line survives
    assert "considered stale" in text


def test_set_typed_value_from_api_caller(tmp_path):
    # raw may already be a typed Python object (JSON API path), not a string
    _seed(tmp_path)
    assert c.set_config_value("watch", True, storage_path=str(tmp_path)) is True
    assert c.set_config_value("staleness_days", 9, storage_path=str(tmp_path)) == 9


def test_coerce_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config key"):
        c.coerce_config_value("nope_key", "1")


def test_coerce_rejects_wrong_type():
    with pytest.raises(ValueError, match="expects int"):
        c.coerce_config_value("staleness_days", "notanumber")


def test_set_rejects_readonly(tmp_path):
    _seed(tmp_path)
    with pytest.raises(ValueError, match="read-only"):
        c.set_config_value("version", "9.9.9", storage_path=str(tmp_path))


def test_unset_clears_and_reports(tmp_path):
    p = _seed(tmp_path)
    c.set_config_value("watch", "true", storage_path=str(tmp_path))
    assert "watch" in _read(p)
    assert c.unset_config_value("watch", storage_path=str(tmp_path)) is True
    assert "watch" not in _read(p)
    # idempotent: clearing an unset key is a no-op
    assert c.unset_config_value("watch", storage_path=str(tmp_path)) is False


def test_set_keeps_file_valid_jsonc(tmp_path):
    p = _seed(tmp_path)
    for key, raw in [("watch", "true"), ("languages", '["python"]'),
                     ("extra_extensions", '{".x":"go"}'), ("meta_fields", "null")]:
        c.set_config_value(key, raw, storage_path=str(tmp_path))
    issues = [i for i in c.validate_config(str(p)) if "invalid" in i or "parse" in i]
    assert issues == []


def test_set_creates_file_from_template_when_absent(tmp_path):
    # no seed: file should be created
    assert not (tmp_path / "config.jsonc").exists()
    c.set_config_value("watch", "true", storage_path=str(tmp_path))
    assert (tmp_path / "config.jsonc").exists()
    assert _read(tmp_path / "config.jsonc")["watch"] is True


def test_scan_value_end_handles_embedded_comments(tmp_path):
    # languages/meta_fields arrays carry `// "field",` comment lines inside the
    # brackets; the scanner must span to the real closing bracket.
    p = _seed(tmp_path)
    c.set_config_value("languages", '["rust"]', storage_path=str(tmp_path))
    # the replacement must not have left a dangling bracket or duplicate key
    text = p.read_text(encoding="utf-8")
    assert text.count('"languages":') == 1
    assert _read(p)["languages"] == ["rust"]
