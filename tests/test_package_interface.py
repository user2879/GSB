"""Public package surface and end-to-end engine interface coverage."""
from __future__ import annotations

import obsflow


def test_public_interface_exposes_required_operations():
    required = [
        "load_source_set", "load_stream", "decode_stream",
        "ObsFlowEngine", "BoundedScheduler", "LeaseManager", "Ledger",
        "generate_samples", "quantize_value", "aggregate_window",
        "FormatError", "IntegrityError", "OrderError", "SnapshotError",
        "LeaseError", "ReplayError",
    ]
    for name in required:
        assert hasattr(obsflow, name)


def test_engine_interface_members(sample_dir, workdir):
    engine = obsflow.ObsFlowEngine.from_files(
        sample_dir["declaration_path"], sample_dir["window_path"],
        f"{workdir}/iface.db")
    for method in ("read_records", "advance", "query_windows",
                   "save_snapshot", "recover", "replay_verify",
                   "apply_source_set", "apply_window_params", "close",
                   "verify_snapshot"):
        assert callable(getattr(engine, method))
    engine.read_records()
    report = engine.advance()
    assert report.windows
    assert engine.query_windows()
    identity = engine.save_snapshot()
    assert len(identity.content_digest) == 64
    engine.close()
