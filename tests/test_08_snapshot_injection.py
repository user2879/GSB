"""Requirement 8: atomic snapshots, crash recovery, injection isolation."""

from __future__ import annotations

import json
import os

import pytest

from vad_diag.config import build_config
from vad_diag.engine import VADEngine
from vad_diag.config import InjectionRule
from vad_diag.injection import InjectionHarness, assert_finite
from vad_diag.ledger import Ledger
from vad_diag.cache import ProductCache
from vad_diag.snapshot import SnapshotStore, config_hash
from vad_diag.loader import Command
from vad_diag.segments import build_in_memory_segment
from vad_diag.samplegen import build_segments
from vad_diag.errors import (
    E_AUDIO_FORMAT, E_FIELD_INVALID, E_INJECTION, VADInjectionError,
)


def _engine(config, tmp_path, harness=None):
    ledger = Ledger(str(tmp_path / "l.jsonl"))
    cache = ProductCache(str(tmp_path / "c.json")).load()
    store = SnapshotStore(str(tmp_path / "ss"))
    store.ensure_dirs()
    engine = VADEngine(config, ledger, cache, store,
                       harness=harness or InjectionHarness([]))
    engine.auto_flush = False
    return engine


def _segs(n=12):
    entries, _ = build_segments(segment_count=n, segment_samples=3200)
    return {e["segment_id"]: build_in_memory_segment(
        e["segment_id"], e["samples"], e["capture_ms"], e["arrival_ms"],
        e["process_ms"]) for e in entries}


def _feed(engine, segs):
    for i, sid in enumerate(sorted(segs)):
        engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))


# ------------------------------------------------------------- snapshots
def test_snapshot_is_atomic_and_manifest_versioned(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    _feed(engine, _segs(6))
    out = engine.handle_command(Command(0, "snapshot", {}))
    current = os.path.join(engine.snapshots.current_dir, "manifest.json")
    assert os.path.isfile(current)
    with open(current, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["version"] == engine.global_version
    assert manifest["config_hash"] == config_hash(config.to_dict())
    assert set(manifest["stage_versions"]) == {
        "load", "vad", "window", "posterior"}
    assert out["snapshot"]["quant_level"].startswith("Q")


def test_recovery_uses_config_hash_not_file_overlays(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    _feed(engine, _segs(6))
    engine.handle_command(Command(99, "snapshot", {}))

    # Same config -> recovery accepted.
    same = VADEngine(
        config, Ledger(str(tmp_path / "l2.jsonl")),
        ProductCache(str(tmp_path / "c2.json")).load(),
        engine.snapshots, harness=InjectionHarness([]))
    state = same.restore_from_snapshot()
    assert state is not None
    assert state.config_hash == config_hash(config.to_dict())

    # Different config -> snapshot rejected, full rebuild required.
    other_cfg = config_factory(energy_high=0.07)
    other = VADEngine(
        other_cfg, Ledger(str(tmp_path / "l3.jsonl")),
        ProductCache(str(tmp_path / "c3.json")).load(),
        engine.snapshots, harness=InjectionHarness([]))
    assert other.restore_from_snapshot() is None


def test_forced_termination_leaves_recoverable_current(config_factory,
                                                        tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    _feed(engine, _segs(6))
    # Simulate SIGKILL before the atomic swap: temp dir exists, current valid.
    engine._take_snapshot(Command(1, "snapshot", {}),
                          crash_point="before_swap")
    tmp_entries = [n for n in os.listdir(engine.snapshots.root)
                   if n.startswith(".tmp-snap-")]
    assert tmp_entries  # the abandoned temp dir is present
    # A subsequent clean commit establishes a recoverable current snapshot.
    engine.handle_command(Command(2, "snapshot", {}))
    state = engine.snapshots.recover(config_hash(config.to_dict()))
    # Recovery discards stale temp and still loads the last good snapshot.
    assert state is not None
    leftover = [n for n in os.listdir(engine.snapshots.root)
                if n.startswith(".tmp-snap-")]
    assert not leftover


def test_graceful_shutdown_emits_exit_report(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    _feed(engine, _segs(6))
    out = engine.handle_command(Command(0, "shutdown", {}))
    report = out["exit_report"]
    assert report["reason"] == "graceful_shutdown"
    assert report["endpoint_count"] >= 0
    assert "global_version" in report


# ------------------------------------------------------------- injections
def test_truncate_injection_isolates_one_segment(config_factory, tmp_path):
    rules = [InjectionRule(
        "INJ-001", "truncate_data", "load", "S0005",
        {"keep_samples": 100})]
    harness = InjectionHarness(rules)
    config = config_factory()
    engine = _engine(config, tmp_path, harness=harness)
    segs = _segs(10)
    for i, sid in enumerate(sorted(segs)):
        engine._admit_segment(segs[sid])
    # The injected segment is quarantined, others accepted.
    assert "S0005" not in engine.segments
    assert len(engine.segments) == 9
    assert harness.report.ids() == ["INJ-001"]
    failure = engine.failures[-1]
    assert failure.code == E_AUDIO_FORMAT
    assert failure.injection_id == "INJ-001"


def test_window_feature_dim_mismatch_is_isolated(config_factory, tmp_path):
    rules = [InjectionRule(
        "INJ-002", "feature_dim_mismatch", "window", "W0002", {"dim": 7})]
    harness = InjectionHarness(rules)
    config = config_factory()
    engine = _engine(config, tmp_path, harness=harness)
    _feed(engine, _segs(8))
    # The fault fires through the window hook on the target window.
    feat = engine._last_result.features.get("W0002")
    # The hook raises; the engine path records a failure for that window.
    assert any(f.injection_id == "INJ-002"
               for f in engine.failures) or feat is None
    assert "INJ-002" in harness.report.ids()
    # Surviving windows still produced features.
    assert len(engine._last_result.features) > 0


def test_invalid_quant_mode_injection_rejected(config_factory, tmp_path):
    rules = [InjectionRule(
        "INJ-003", "quant_mode_invalid", "load", "config",
        {"mode": "BOGUS"})]
    harness = InjectionHarness(rules)
    with pytest.raises(VADInjectionError) as exc:
        harness.validate_quant_mode("BOGUS", "config")
    assert exc.value.code == E_FIELD_INVALID
    assert harness.report.ids() == ["INJ-003"]


def test_non_finite_sample_detected():
    with pytest.raises(Exception) as exc:
        assert_finite([1.0, 2.0, float("nan"), 3.0], "S0001")
    assert exc.value.code == E_AUDIO_FORMAT


def test_range_overflow_injection_raises(config_factory, tmp_path):
    rules = [InjectionRule(
        "INJ-004", "range_overflow", "window", "W0001", {})]
    harness = InjectionHarness(rules)
    config = config_factory()
    engine = _engine(config, tmp_path, harness=harness)
    _feed(engine, _segs(6))
    # Use a clean engine (no rule attached) to obtain a valid feature, then
    # prove the injection rule fires when applied to the target window.
    clean = _engine(config, tmp_path / "cleanref")
    _feed(clean, _segs(6))
    target = clean._last_result.windows[1]
    feature = clean._last_result.features[target.window_id]
    fresh_harness = InjectionHarness(rules)
    with pytest.raises(VADInjectionError) as exc:
        fresh_harness.window_hook("W0001", feature)
    assert exc.value.code == E_AUDIO_FORMAT
    assert fresh_harness.report.ids() == ["INJ-004"]


def test_isolated_failure_keeps_others_consistent(config_factory, tmp_path):
    # With S0005 failed, conclusions over the surviving set must equal a clean
    # run built from the same surviving segments.
    rules = [InjectionRule(
        "INJ-001", "truncate_data", "load", "S0005",
        {"keep_samples": 100})]
    harness = InjectionHarness(rules)
    config = config_factory()
    injected = _engine(config, tmp_path / "inj", harness=harness)
    segs = _segs(10)
    for sid in sorted(segs):
        injected._admit_segment(segs[sid])
    result_inj = injected.run_pipeline()

    clean = _engine(config, tmp_path / "clean")
    surviving = {sid: s for sid, s in segs.items() if sid != "S0005"}
    for sid in sorted(surviving):
        clean.segments[sid] = surviving[sid]
        clean.arrival_order.append(sid)
    result_clean = clean.run_pipeline()

    assert result_inj.timeline.fingerprint() == \
        result_clean.timeline.fingerprint()
    assert {w: p.posterior_q for w, p in
            result_inj.posteriors.items()} == \
        {w: p.posterior_q for w, p in result_clean.posteriors.items()}
