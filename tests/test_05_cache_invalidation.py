"""Requirement 5: content-fingerprint invalidation and reuse ledger."""

from __future__ import annotations

from vad_diag.engine import VADEngine
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


def _segments(n=12):
    entries, _ = build_segments(segment_count=n, segment_samples=3200)
    return {e["segment_id"]: build_in_memory_segment(
        e["segment_id"], e["samples"], e["capture_ms"], e["arrival_ms"],
        e["process_ms"]) for e in entries}


def _feed(engine, segs):
    for i, sid in enumerate(sorted(segs)):
        engine.handle_command(
            Command(i, "append", {"segment_id": sid, "segment": segs[sid]}))


def test_unchanged_windows_are_reused_after_rerun(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    segs = _segments()
    _feed(engine, segs)
    ledger_len = len(engine.ledger.events)
    # Re-running the pipeline on identical content must yield reuse hits.
    result = engine.run_pipeline(changed_segments=[])
    reuse = [e for e in engine.ledger.by_kind("reuse_hit")
             if e["seq"] >= ledger_len]
    assert reuse, "expected cache reuse hits on unchanged content"
    assert set(result.reused_windows) == {w.window_id for w in result.windows}


def test_replacement_only_invalidates_referencing_windows(
        config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    segs = _segments()
    _feed(engine, segs)

    target = "S0006"
    before = engine._last_result
    affected_before = set(before.affected_windows)

    # Build replacement content with a different seed-derived waveform.
    entries, _ = build_segments(seed=999, segment_count=1,
                                segment_samples=3200)
    replacement = build_in_memory_segment(
        target, entries[0]["samples"],
        segs[target].capture_ms, segs[target].arrival_ms + 50,
        segs[target].process_ms)
    engine.handle_command(
        Command(99, "replace",
                {"segment_id": target, "segment": replacement}))
    after = engine._last_result

    # Only windows referencing S0006 may be recomputed.
    referencing = {w.window_id for w in after.windows
                   if target in w.segment_ids}
    assert set(after.recomputed_windows) <= referencing
    # Windows not referencing the replaced segment are reused.
    non_ref = [w.window_id for w in after.windows
               if target not in w.segment_ids]
    assert set(non_ref) <= set(after.reused_windows)
    # Ledger contains explicit invalidation events naming the segment.
    invalidations = [e for e in engine.ledger.events
                     if e["kind"] == "invalidation"]
    assert any(e.get("caused_by_segment") == target for e in invalidations)


def test_cache_keys_change_with_quantization(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    segs = _segments()
    _feed(engine, segs)
    key_q15 = engine._last_result.features["W0003"].cache_key
    engine.handle_command(Command(50, "set_quant",
                                  {"frac_bits": 8, "quant_mode": "TRUNC"}))
    key_q8 = engine._last_result.features["W0003"].cache_key
    assert key_q15 != key_q8


def test_every_endpoint_records_referencing_windows(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    _feed(engine, _segments())
    result = engine._last_result
    for endpoint in result.endpoints:
        assert endpoint.window_ids
        for wid in endpoint.window_ids:
            assert wid in result.posteriors


def test_ledger_records_recompute_and_reuse(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    _feed(engine, _segments(8))
    kinds = {e["kind"] for e in engine.ledger.events}
    assert "recompute" in kinds
    # Trigger a no-change rerun to produce reuse hits.
    engine.run_pipeline(changed_segments=[])
    assert "reuse_hit" in {e["kind"] for e in engine.ledger.events}
