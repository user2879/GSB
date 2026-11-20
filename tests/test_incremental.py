import json
import os

import pytest

from hdr_sdr_pipeline.config import ProcessingState, STAGES
from hdr_sdr_pipeline.engine import IncrementalEngine
from hdr_sdr_pipeline.ledger import new_ledger


def _engine(work_dir, state=None):
    manifest = json.load(open(os.path.join(work_dir, "truth.json")))
    ledger = new_ledger(state or ProcessingState(),
                        manifest["generator"]["frame_count"],
                        manifest["manifest_hash"])
    engine = IncrementalEngine(work_dir, ledger, manifest,
                               state or ProcessingState())
    engine.build_all()
    return engine, ledger, manifest


def _hashes(ledger):
    return {k: r["output_hash"] for k, r in ledger["outputs"].items()}


def test_exposure_closure_starts_at_exposure(tiny_workdir):
    engine, ledger, _ = _engine(tiny_workdir)
    before = _hashes(ledger)
    new_state = ProcessingState(exposure_ev=1.0)
    actual = engine.transition(new_state, target_frames=[2, 3])
    assert actual["stages"] == list(STAGES[STAGES.index("exposure"):])
    assert actual["frames"] == [2, 3]
    after = _hashes(ledger)
    for f in range(8):
        if f in (2, 3):
            assert after[str(f)] != before[str(f)]
        else:
            assert after[str(f)] == before[str(f)]


def test_transfer_change_only_invalidates_transfer_onward(tiny_workdir):
    engine, ledger, _ = _engine(tiny_workdir)
    new_state = ProcessingState(transfer="gamma", transfer_gamma=2.0)
    actual = engine.transition(new_state, target_frames=[0])
    assert actual["stages"] == ["transfer", "quantize"]


def test_interval_and_per_frame_routes_match(tiny_workdir):
    # interval route
    eng_a, ledger_a, manifest = _engine(tiny_workdir)
    eng_a.transition(ProcessingState(exposure_ev=0.75),
                     target_frames=[0, 1, 2, 3], route="interval")
    eng_a.transition(ProcessingState(exposure_ev=0.75,
                                     tonemap_operator="aces"),
                     target_frames=[4, 5, 6, 7], route="interval")

    # per-frame route on a mirrored workdir
    mirror = tiny_workdir + "_mirror"
    import shutil

    shutil.copytree(tiny_workdir, mirror)
    manifest_b = json.load(open(os.path.join(mirror, "truth.json")))
    ledger_b = new_ledger(ProcessingState(), 8, manifest_b["manifest_hash"])
    eng_b = IncrementalEngine(mirror, ledger_b, manifest_b, ProcessingState())
    eng_b.build_all()
    eng_b.transition_per_frame(ProcessingState(exposure_ev=0.75),
                               target_frames=[0, 1, 2, 3])
    eng_b.transition_per_frame(
        ProcessingState(exposure_ev=0.75, tonemap_operator="aces"),
        target_frames=[4, 5, 6, 7])
    assert _hashes(ledger_a) == _hashes(ledger_b)


def test_recompute_set_equals_closure_and_intervals(tiny_workdir):
    engine, ledger, manifest = _engine(tiny_workdir)
    new_state = ProcessingState(tonemap_operator="aces")
    actual = engine.transition(new_state, target_frames=[1, 2])
    closure = engine.last_recompute()
    assert actual == closure
    assert set(closure["intervals"]) == set(
        name for name, (lo, hi) in engine.intervals.items()
        if any(lo <= f < hi for f in (1, 2))
    )


def test_no_full_rebuild_on_partial_change(tiny_workdir):
    engine, ledger, _ = _engine(tiny_workdir)
    op_count_before = len(ledger["op_log"])
    engine.transition(ProcessingState(exposure_ev=0.25), target_frames=[5])
    op = ledger["op_log"][-1]
    assert op["closure"]["frames"] == [5]
    assert op["actual"]["frames"] == [5]
    assert len(ledger["op_log"]) == op_count_before + 1


def test_metrics_recomputed_only_for_closure_intervals(tiny_workdir):
    engine, ledger, _ = _engine(tiny_workdir)
    first_q_seq = ledger["interval_metrics"]["first_quarter"]["seq"]
    engine.transition(ProcessingState(exposure_ev=0.5), target_frames=[6, 7])
    # frames 6,7 sit in second_half/whole; first_quarter must stay at old seq
    assert ledger["interval_metrics"]["first_quarter"]["seq"] == first_q_seq
    assert ledger["interval_metrics"]["second_half"]["seq"] != first_q_seq
