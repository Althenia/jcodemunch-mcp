"""Tests for the org-rollup store (team SKU telemetry aggregation)."""

from __future__ import annotations

import pytest

from jcodemunch_mcp.org.store import record_seat_report, org_rollup


def test_rollup_aggregates_seats(tmp_path):
    d = str(tmp_path)
    record_seat_report("acme", "laptop-1", 700000, 10.5, 200, storage_path=d)
    record_seat_report("acme", "seat-2", 500000, 7.5, 120, storage_path=d)
    r = org_rollup("acme", storage_path=d)
    assert r["org_id"] == "acme"
    assert r["totals"]["seat_count"] == 2
    assert r["totals"]["tokens_saved"] == 1200000
    assert r["totals"]["calls"] == 320
    # Sorted by tokens desc.
    assert r["seats"][0]["seat_id"] == "laptop-1"


def test_same_seat_same_day_upserts(tmp_path):
    d = str(tmp_path)
    record_seat_report("acme", "s1", 100, 1.0, 1, storage_path=d, date="2026-06-07")
    record_seat_report("acme", "s1", 999, 9.0, 9, storage_path=d, date="2026-06-07")
    r = org_rollup("acme", storage_path=d)
    assert r["totals"]["seat_count"] == 1
    assert r["seats"][0]["tokens_saved"] == 999  # overwrote, not summed


def test_same_seat_multiple_days_sum(tmp_path):
    d = str(tmp_path)
    record_seat_report("acme", "s1", 100, 1.0, 1, storage_path=d, date="2026-06-06")
    record_seat_report("acme", "s1", 200, 2.0, 2, storage_path=d, date="2026-06-07")
    r = org_rollup("acme", storage_path=d)
    assert r["seats"][0]["tokens_saved"] == 300


def test_orgs_are_isolated(tmp_path):
    d = str(tmp_path)
    record_seat_report("acme", "s1", 100, 1.0, 1, storage_path=d)
    record_seat_report("other", "s1", 999, 9.0, 9, storage_path=d)
    assert org_rollup("acme", storage_path=d)["totals"]["tokens_saved"] == 100
    assert org_rollup("other", storage_path=d)["totals"]["tokens_saved"] == 999


def test_record_requires_ids(tmp_path):
    with pytest.raises(ValueError):
        record_seat_report("", "s1", 1, 1.0, 1, storage_path=str(tmp_path))
