"""Requirement 7: incremental edits == full rebuild, field by field.

The same *final* segment set and configuration must produce identical window
posteriors, frame endpoints, boundary lists, cache keys, reuse/recompute lists
and failure ordering regardless of delivery order or edit sequence, and under
any thread-pool concurrency up to the configured cap.
"""

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


def _segs(n=14, seed=20261126):
    entries, _ = build_segments(seed=seed, segment_count=n,
                                segment_samples=3200)
    return {e["segment_id"]: build_in_memory_segment(
        e["segment_id"], e["samples"], e["capture_ms"], e["arrival_ms"],
        e["process_ms"]) for e in entries}


def _signature(result):
    """Field-level comparable signature of every deterministic artifact."""
    return {
        "timeline_fp": result.timeline.fingerprint(),
        "posteriors": {
            wid: (p.posterior_q, p.is_speech, p.boundary_window,
                  p.tie_kind, p.cache_key)
            for wid, p in sorted(result.posteriors.items())},
        "feature_ints": {
            wid: tuple(f.int_features)
            for wid, f in sorted(result.features.items())},
        "endpoints": [
            (e.frame_start, e.frame_end, tuple(e.frame_indices),
             tuple(sorted(e.window_ids)), tuple(e.boundary_frames))
            for e in result.endpoints],
        "regions": [
            (r.frame_start, r.frame_end, tuple(r.frame_indices),
             tuple(sorted(r.segment_ids))) for r in result.regions],
        "boundary_windows": tuple(sorted(
            p.window_id for p in result.posteriors.values()
            if p.boundary_window)),
        "boundary_frames": tuple(sorted(
            d.frame_index for d in result.decisions
            if d.boundary_frame)),
        "keys": result.keys,
    }


def _full_rebuild(config, tmp_path, segs):
    engine = _engine(config, tmp_path / "full")
    for i, sid in enumerate(sorted(segs)):
        engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    return engine, engine._last_result


def test_full_rebuild_is_stable_across_runs(config_factory, tmp_path):
    config = config_factory()
    segs = _segs()
    _, r1 = _full_rebuild(config, tmp_path / "a", segs)
    _, r2 = _full_rebuild(config, tmp_path / "b", segs)
    assert _signature(r1) == _signature(r2)


def test_out_of_order_delivery_matches_full_rebuild(config_factory, tmp_path):
    config = config_factory()
    segs = _segs()
    _, full = _full_rebuild(config, tmp_path / "full", segs)

    engine = _engine(config, tmp_path / "inc")
    order = sorted(segs, key=lambda s: int(s[1:]), reverse=True)
    for i, sid in enumerate(order):
        engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    assert _signature(engine._last_result) == _signature(full)


def test_edit_series_matches_full_rebuild(config_factory, tmp_path):
    config = config_factory()
    segs = _segs()

    # Incremental: append, replace, delete, then restore to the final set.
    engine = _engine(config, tmp_path / "inc")
    ids = sorted(segs)
    for i, sid in enumerate(ids[:8]):
        engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    # Replace S0003 with fresh content, then delete S0007, then re-append
    # originals so the final set equals the clean full rebuild.
    alt_entries, _ = build_segments(seed=4242, segment_count=1,
                                    segment_samples=3200)
    alt = build_in_memory_segment(
        "S0003", alt_entries[0]["samples"],
        segs["S0003"].capture_ms, segs["S0003"].arrival_ms, 5)
    engine.handle_command(
        Command(100, "replace", {"segment_id": "S0003", "segment": alt}))
    engine.handle_command(
        Command(101, "delete", {"segment_id": "S0007"}))
    # Restore canonical content for S0003 and re-admit S0007.
    engine.handle_command(
        Command(102, "replace",
                {"segment_id": "S0003", "segment": segs["S0003"]}))
    engine.handle_command(
        Command(103, "append",
                {"segment_id": "S0007", "segment": segs["S0007"]}))
    for j, sid in enumerate(ids[8:]):
        engine.handle_command(
            Command(200 + j, "append",
                    {"segment_id": sid, "segment": segs[sid]}))

    _, full = _full_rebuild(config, tmp_path / "full", segs)
    assert _signature(engine._last_result) == _signature(full)


def test_parameter_then_quant_switch_matches_rebuild(config_factory, tmp_path):
    config = config_factory()
    segs = _segs()
    engine = _engine(config, tmp_path / "inc")
    for i, sid in enumerate(sorted(segs)):
        engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    engine.handle_command(Command(300, "set_params",
                                  {"params": {"energy_high": 0.06}}))
    engine.handle_command(Command(301, "set_quant",
                                  {"frac_bits": 8, "quant_mode": "TRUNC"}))
    engine.handle_command(Command(302, "set_quant",
                                  {"frac_bits": 15,
                                   "quant_mode": "ROUND_HALF_EVEN"}))
    engine.handle_command(Command(303, "set_params",
                                  {"params": {"energy_high": 0.05}}))

    _, full = _full_rebuild(config, tmp_path / "full", segs)
    assert _signature(engine._last_result) == _signature(full)


def test_concurrent_and_serial_features_match(config_factory, tmp_path):
    config = config_factory(max_workers=4)
    segs = _segs(10)
    serial = _engine(config, tmp_path / "serial")
    for i, sid in enumerate(sorted(segs)):
        serial.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    serial_result = serial.run_parallel_features(max_workers=1)

    parallel = _engine(config, tmp_path / "parallel")
    for i, sid in enumerate(sorted(segs)):
        parallel.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    parallel_result = parallel.run_parallel_features(max_workers=4)

    for wid in serial_result.features:
        assert (serial_result.features[wid].int_features
                == parallel_result.features[wid].int_features)
    assert {w: (p.posterior_q, p.is_speech)
            for w, p in serial_result.posteriors.items()} == \
        {w: (p.posterior_q, p.is_speech)
         for w, p in parallel_result.posteriors.items()}


def test_worker_count_never_exceeds_config_cap(config_factory, tmp_path,
                                               monkeypatch):
    import vad_diag.windows as win_mod
    config = config_factory(max_workers=3)
    observed = []
    real = win_mod.ThreadPoolExecutor

    class RecordingPool(real):
        def __init__(self, max_workers=None, *args, **kwargs):
            observed.append(max_workers)
            super().__init__(max_workers=max_workers, *args, **kwargs)

    monkeypatch.setattr(win_mod, "ThreadPoolExecutor", RecordingPool)
    engine = _engine(config, tmp_path)
    segs = _segs(8)
    for i, sid in enumerate(sorted(segs)):
        engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))
    engine.run_parallel_features(max_workers=16)  # must be clamped to 3
    assert observed
    assert max(observed) <= 3
