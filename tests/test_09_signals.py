"""Requirement 8 (signal aspect): handleable exit produces an exit report."""

from __future__ import annotations

import json
import os
import signal

from vad_diag.engine import VADEngine
from vad_diag.injection import InjectionHarness
from vad_diag.ledger import Ledger
from vad_diag.cache import ProductCache
from vad_diag.snapshot import SnapshotStore
from vad_diag.signals import GracefulExit, install_handlers
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


def _segs(n=8):
    entries, _ = build_segments(segment_count=n, segment_samples=3200)
    return {e["segment_id"]: build_in_memory_segment(
        e["segment_id"], e["samples"], e["capture_ms"], e["arrival_ms"],
        e["process_ms"]) for e in entries}


def test_sigterm_produces_exit_report_and_snapshot(config_factory, tmp_path):
    config = config_factory()
    engine = _engine(config, tmp_path)
    segs = _segs()
    for i, sid in enumerate(sorted(segs)):
        engine.handle_command(
            Command(i, "append",
                    {"segment_id": sid, "segment": segs[sid]}))

    captured = {}
    install_handlers(engine, on_exit=lambda report: captured.update(report))
    # Force auto_flush so the handler persists state.
    engine.auto_flush = True
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except GracefulExit:
        pass
    assert captured["reason"].startswith("signal_")
    assert "global_version" in captured
    # A snapshot manifest must have been written by the handler.
    manifest = os.path.join(engine.snapshots.current_dir, "manifest.json")
    assert os.path.isfile(manifest)
    with open(manifest) as handle:
        data = json.load(handle)
    assert data["version"] == engine.global_version
