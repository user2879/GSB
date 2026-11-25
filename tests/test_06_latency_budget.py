"""Requirement 6: latency-budget response semantics and stale/fresh split."""

from __future__ import annotations

from vad_diag.engine import VADEngine
from vad_diag.errors import E_STALE_RESULT
from vad_diag.injection import InjectionHarness
from vad_diag.ledger import Ledger
from vad_diag.cache import ProductCache
from vad_diag.snapshot import SnapshotStore
from vad_diag.loader import Command
from vad_diag.segments import build_in_memory_segment
from vad_diag.samplegen import build_segments


def _engine(config, tmp_path):
    ledger = Ledger(str(tmp_path / "l.jsonl"))
    cache = ProductCache(str(tmp_path / "c.json")).load()
    store = SnapshotStore(str(tmp_path / "ss"))
    store.ensure_dirs()
    engine = VADEngine(config, ledger, cache, store,
                       harness=InjectionHarness([]))
    engine.auto_flush = False
    return engine


def _segs(n=12):
    entries, _ = build_segments(segment_count=n, segment_samples=3200)
    return {e["segment_id"]: build_in_memory_segment(
        e["segment_id"], e["samples"], e["capture_ms"], e["arrival_ms"],
        e["process_ms"]) for e in entries}


def test_generous_budget_returns_fresh_authoritative(
        config_factory, tmp_path):
    config = config_factory(latency_budget_ms=1_000_000)
    engine = _engine(config, tmp_path)
    segs = _segs()
    resp = engine.handle_command(
        Command(0, "append",
                {"segment_id": "S0000", "segment": segs["S0000"]}))
    summary = resp["summary"]
    assert summary["status"] == "fresh"
    assert summary["code"] is None
    # An authoritative summary is recorded in the ledger.
    assert engine.ledger.by_kind("authoritative_summary")


def test_tight_budget_returns_stale_with_code_and_watermark(
        config_factory, tmp_path):
    # First establish an authoritative baseline under a generous budget.
    base = config_factory(latency_budget_ms=1_000_000)
    engine = _engine(base, tmp_path)
    segs = _segs()
    engine.handle_command(
        Command(0, "append",
                {"segment_id": "S0000", "segment": segs["S0000"]}))
    assert engine.ledger.by_kind("authoritative_summary")
    # Now tighten the budget so the next recompute cannot finish in time.
    engine.config = config_factory(latency_budget_ms=0, segment_cost_ms=5,
                                   window_cost_ms=5)
    second = engine.handle_command(
        Command(1, "append",
                {"segment_id": "S0001", "segment": segs["S0001"]}))
    summary = second["summary"]
    assert summary["status"] == "stale"
    assert summary["code"] == E_STALE_RESULT
    assert "stale_watermark" in summary or summary["watermark"]
    # Stale summary must not be ledgered as authoritative again.
    auth = engine.ledger.by_kind("authoritative_summary")
    assert len(auth) == 1


def test_stale_does_not_advance_completed_cursor(config_factory, tmp_path):
    base = config_factory(latency_budget_ms=1_000_000)
    engine = _engine(base, tmp_path)
    segs = _segs(4)
    engine.handle_command(
        Command(0, "append",
                {"segment_id": "S0000", "segment": segs["S0000"]}))
    baseline_cursor = list(engine.completed_segments)
    engine.config = config_factory(latency_budget_ms=0, segment_cost_ms=10,
                                   window_cost_ms=10)
    engine.handle_command(
        Command(1, "append",
                {"segment_id": "S0001", "segment": segs["S0001"]}))
    # Cursor must remain at the last authoritative position.
    assert engine.completed_segments == baseline_cursor


def test_summary_contains_progress_and_affected_windows(
        config_factory, tmp_path):
    config = config_factory(latency_budget_ms=1_000_000)
    engine = _engine(config, tmp_path)
    segs = _segs(6)
    for i, sid in enumerate(sorted(segs)):
        resp = engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    summary = resp["summary"]
    assert summary["progress"]["windows_total"] >= summary[
        "progress"]["windows_done"]
    assert "endpoint_intervals" in summary
    assert "affected_windows" in summary
