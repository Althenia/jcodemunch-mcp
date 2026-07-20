"""Calendar windows + per-day buckets for the receipt CLI (v1.108.134).

``--days N`` is a rolling window back from now, so it can't express a
calendar day ("today", "yesterday") at all. ``--since``/``--until`` can,
and ``--by-day`` turns one transcript scan into a real per-day series
instead of a caller re-scanning per range.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import pytest

from jcodemunch_mcp.cli.receipt import (
    _MODEL_PRICES_USD_PER_MTOK,
    aggregate,
    aggregate_by_day,
    iter_calls,
    main as receipt_main,
    parse_window_bound,
    render_json,
    render_rates,
)

from .test_receipt import _make_call, _write_session


def _local(day: str, hour: int = 12) -> str:
    """A UTC transcript timestamp landing at `hour` local time on `day`."""
    naive = _dt.datetime.fromisoformat(f"{day}T{hour:02d}:00:00")
    return naive.astimezone().astimezone(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _corpus(root: Path) -> None:
    events: list[dict] = []
    for i, day in enumerate(("2026-07-14", "2026-07-15", "2026-07-17")):
        events += _make_call("mcp__jcodemunch__search_symbols", f"tu{i}", "x" * 400, ts=_local(day))
    _write_session(root / "s.jsonl", events)


class TestParseWindowBound:
    def test_bare_date_is_local_midnight(self):
        got = parse_window_bound("2026-07-16")
        assert (got.year, got.month, got.day) == (2026, 7, 16)
        assert (got.hour, got.minute) == (0, 0)
        assert got.tzinfo is not None

    def test_iso_datetime_with_z_is_utc(self):
        got = parse_window_bound("2026-07-16T09:30:00Z")
        assert got.utcoffset() == _dt.timedelta(0)
        assert got.hour == 9

    def test_naive_iso_datetime_reads_as_local(self):
        assert parse_window_bound("2026-07-16T09:30").tzinfo is not None

    def test_garbage_rejected(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_window_bound("notadate")


class TestCalendarWindow:
    def test_single_day_window_isolates_that_day(self, tmp_path: Path):
        _corpus(tmp_path)
        calls = list(
            iter_calls(
                tmp_path,
                since=parse_window_bound("2026-07-15"),
                until=parse_window_bound("2026-07-16"),
            )
        )
        assert len(calls) == 1

    def test_until_is_exclusive_so_windows_dont_double_count(self, tmp_path: Path):
        """A call at the shared boundary belongs to exactly one of two adjacent windows."""
        boundary = parse_window_bound("2026-07-15")
        _write_session(
            tmp_path / "s.jsonl",
            _make_call("mcp__jcodemunch__search_symbols", "tu0", "x" * 400, ts=boundary.isoformat()),
        )
        earlier = list(iter_calls(tmp_path, since=parse_window_bound("2026-07-14"), until=boundary))
        later = list(iter_calls(tmp_path, since=boundary, until=parse_window_bound("2026-07-16")))
        assert (len(earlier), len(later)) == (0, 1)

    def test_since_only_leaves_upper_bound_open(self, tmp_path: Path):
        _corpus(tmp_path)
        calls = list(iter_calls(tmp_path, since=parse_window_bound("2026-07-15")))
        assert len(calls) == 2

    def test_no_bounds_is_unchanged_all_time(self, tmp_path: Path):
        _corpus(tmp_path)
        assert len(list(iter_calls(tmp_path))) == 3


class TestAggregateByDay:
    def test_buckets_by_local_calendar_day(self, tmp_path: Path):
        _corpus(tmp_path)
        rows = aggregate_by_day(iter_calls(tmp_path), model="opus")
        assert [r["date"] for r in rows] == ["2026-07-14", "2026-07-15", "2026-07-17"]

    def test_days_sum_to_the_window_total(self, tmp_path: Path):
        """The series must reconcile with the headline figure, or one of them is lying."""
        _corpus(tmp_path)
        calls = list(iter_calls(tmp_path))
        rows = aggregate_by_day(calls, model="opus")
        assert sum(r["savings_tokens"] for r in rows) == aggregate(calls)["totals"]["savings_tokens"]
        assert sum(r["calls"] for r in rows) == aggregate(calls)["totals"]["calls"]

    def test_empty_days_are_absent_not_zero_rows(self, tmp_path: Path):
        _corpus(tmp_path)
        rows = aggregate_by_day(iter_calls(tmp_path), model="opus")
        assert "2026-07-16" not in {r["date"] for r in rows}

    def test_no_calls_yields_empty_series(self, tmp_path: Path):
        assert aggregate_by_day([], model="opus") == []

    def test_usd_priced_at_selected_model_rate(self, tmp_path: Path):
        _corpus(tmp_path)
        opus = aggregate_by_day(iter_calls(tmp_path), model="opus")
        haiku = aggregate_by_day(iter_calls(tmp_path), model="haiku")
        assert opus[0]["savings_usd"] == pytest.approx(haiku[0]["savings_usd"] * 5.0)


class TestRenderRates:
    """`--rates` exists so consumers stop keeping their own copy of the prices."""

    def test_reports_the_live_price_table(self):
        payload = json.loads(render_rates())
        assert payload["rates_usd_per_mtok"] == _MODEL_PRICES_USD_PER_MTOK
        assert payload["default_model"] in payload["rates_usd_per_mtok"]

    def test_scans_no_transcripts(self, tmp_path: Path, monkeypatch):
        """It must not walk the projects tree — a consumer polls this cheaply."""
        import jcodemunch_mcp.cli.receipt as mod

        def explode(*a, **k):  # pragma: no cover - fails the test if reached
            raise AssertionError("--rates must not scan transcripts")

        monkeypatch.setattr(mod, "iter_calls", explode)
        assert receipt_main(["--rates", "--projects-root", str(tmp_path)]) == 0

    def test_every_priced_model_is_selectable(self):
        """The table drives --model's choices, so the two can't disagree."""
        rates = json.loads(render_rates())["rates_usd_per_mtok"]
        for model in rates:
            assert receipt_main(["--rates", "--model", model]) == 0


class TestJsonExport:
    def test_by_day_omitted_by_default(self, tmp_path: Path):
        _corpus(tmp_path)
        payload = json.loads(render_json(aggregate(iter_calls(tmp_path)), model="opus"))
        assert "by_day" not in payload and "window" not in payload

    def test_export_carries_window_and_series(self, tmp_path: Path):
        _corpus(tmp_path)
        out = tmp_path / "r.json"
        rc = receipt_main(
            [
                "--projects-root", str(tmp_path),
                "--since", "2026-07-14",
                "--until", "2026-07-16",
                "--by-day",
                "--export", str(out),
            ]
        )
        assert rc == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["totals"]["calls"] == 2
        assert [r["date"] for r in payload["by_day"]] == ["2026-07-14", "2026-07-15"]
        assert payload["window"]["since"].startswith("2026-07-14")
        assert "days" not in payload["window"]

    def test_rolling_window_export_reports_days(self, tmp_path: Path):
        _corpus(tmp_path)
        out = tmp_path / "r.json"
        receipt_main(["--projects-root", str(tmp_path), "--days", "30", "--export", str(out)])
        assert json.loads(out.read_text(encoding="utf-8"))["window"]["days"] == 30

    def test_inverted_window_rejected(self, tmp_path: Path):
        assert receipt_main(
            ["--projects-root", str(tmp_path), "--since", "2026-07-18", "--until", "2026-07-10"]
        ) == 2
