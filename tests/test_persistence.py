import json
import os

import numpy as np
import pytest

from hdr_sdr_pipeline.config import ProcessingState
from hdr_sdr_pipeline.engine import IncrementalEngine
from hdr_sdr_pipeline.errors import SnapshotStaleError
from hdr_sdr_pipeline.ledger import new_ledger
from hdr_sdr_pipeline.persistence import (
    atomic_write, is_closed, read_ledger, recover,
    assert_snapshot_fresh, write_closed_marker, write_ledger,
)


def test_atomic_write_leaves_no_temp_on_success(empty_workdir):
    path = os.path.join(empty_workdir, "a", "data.json")
    atomic_write(path, "{}")
    leftovers = [f for f in os.listdir(os.path.dirname(path))
                 if ".tmp" in f]
    assert leftovers == []
    assert open(path).read() == "{}"


def test_uncommitted_temp_is_orphaned_on_restart(tiny_workdir):
    manifest = json.load(open(os.path.join(tiny_workdir, "truth.json")))
    ledger = new_ledger(ProcessingState(), 8, manifest["manifest_hash"])
    engine = IncrementalEngine(tiny_workdir, ledger, manifest, ProcessingState())
    engine.build_all()
    write_ledger(tiny_workdir, ledger)
    committed_seq = ledger["committed_seq"]

    # simulated forced termination: orphan temp + missing closed marker
    tmp = os.path.join(tiny_workdir, "frames_out", "frame_0000.u8.tmp.crash")
    open(tmp, "wb").write(b"\x00" * 16)
    assert not is_closed(tiny_workdir)

    recovered, info = recover(tiny_workdir)
    assert recovered["committed_seq"] == committed_seq
    assert not os.path.exists(tmp)
    assert os.path.isdir(os.path.join(tiny_workdir, "orphans"))
    assert any(i["reason"] == "temp" for i in info["reclaimed"])
    assert os.path.exists(os.path.join(tiny_workdir, "recovery_log.json"))


def test_recovered_run_matches_uninterrupted_run(tiny_workdir):
    manifest = json.load(open(os.path.join(tiny_workdir, "truth.json")))
    ledger = new_ledger(ProcessingState(), 8, manifest["manifest_hash"])
    engine = IncrementalEngine(tiny_workdir, ledger, manifest, ProcessingState())
    engine.build_all()
    open(os.path.join(tiny_workdir, "frames_out", "x.u8.tmp"), "wb").write(b"0")
    write_ledger(tiny_workdir, ledger)

    recovered, _ = recover(tiny_workdir)
    engine2 = IncrementalEngine(tiny_workdir, recovered, manifest,
                                ProcessingState())
    engine2.transition(ProcessingState(exposure_ev=0.5), target_frames=[0, 1])
    write_ledger(tiny_workdir, recovered)
    hashes_after_recovery = {
        k: r["output_hash"] for k, r in recovered["outputs"].items()
    }

    # uninterrupted reference in a second workdir
    import shutil

    ref = tiny_workdir + "_ref"
    shutil.copytree(tiny_workdir, ref)
    ref_manifest = json.load(open(os.path.join(ref, "truth.json")))
    ref_ledger = new_ledger(ProcessingState(), 8,
                            ref_manifest["manifest_hash"])
    ref_engine = IncrementalEngine(ref, ref_ledger, ref_manifest,
                                   ProcessingState())
    ref_engine.build_all()
    ref_engine.transition(ProcessingState(exposure_ev=0.5),
                          target_frames=[0, 1])
    ref_hashes = {k: r["output_hash"] for k, r in ref_ledger["outputs"].items()}
    assert hashes_after_recovery == ref_hashes


def test_closed_marker_lifecycle(tiny_workdir):
    manifest = json.load(open(os.path.join(tiny_workdir, "truth.json")))
    ledger = new_ledger(ProcessingState(), 8, manifest["manifest_hash"])
    engine = IncrementalEngine(tiny_workdir, ledger, manifest, ProcessingState())
    engine.build_all()
    write_ledger(tiny_workdir, ledger)
    write_closed_marker(tiny_workdir, ledger)
    assert is_closed(tiny_workdir)
    recover(tiny_workdir)
    assert not is_closed(tiny_workdir)  # resume invalidates prior marker


def test_stale_snapshot_rejected(tiny_workdir):
    manifest = json.load(open(os.path.join(tiny_workdir, "truth.json")))
    ledger = new_ledger(ProcessingState(), 8, manifest["manifest_hash"])
    engine = IncrementalEngine(tiny_workdir, ledger, manifest, ProcessingState())
    engine.build_all()
    write_ledger(tiny_workdir, ledger)
    with pytest.raises(SnapshotStaleError):
        assert_snapshot_fresh(0, None, ledger)
