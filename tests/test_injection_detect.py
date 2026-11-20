import json
import os

import numpy as np
import pytest

from hdr_sdr_pipeline.config import ProcessingState
from hdr_sdr_pipeline.detector import (
    classify, run_detection, validate_manifest, validate_state,
)
from hdr_sdr_pipeline.engine import IncrementalEngine
from hdr_sdr_pipeline.injection import (
    InjectionConfig, apply_manifest_tamper, apply_state_tamper,
    illegal_transfer_param, peak_out_of_bounds, stage_replacement,
    tonemap_constant_tamper, truth_sample_tamper,
)
from hdr_sdr_pipeline.ledger import commit_frame_output, new_ledger
from hdr_sdr_pipeline.persistence import write_ledger
from hdr_sdr_pipeline.pipeline import InjectionContext, process_frame


def _clean_build(work_dir):
    manifest = json.load(open(os.path.join(work_dir, "truth.json")))
    state = ProcessingState(tonemap_operator="aces")
    ledger = new_ledger(state, 8, manifest["manifest_hash"])
    engine = IncrementalEngine(work_dir, ledger, manifest, state)
    engine.build_all()
    write_ledger(work_dir, ledger)
    return manifest, state, ledger


def test_peak_out_of_bounds_detected(tiny_workdir):
    manifest, state, ledger = _clean_build(tiny_workdir)
    item = peak_out_of_bounds()
    patched = apply_manifest_tamper(manifest, item)
    config = InjectionConfig("c1", [item])
    report = run_detection(tiny_workdir, patched, state, config)
    codes = [f["code"] for f in report["findings"]]
    assert "E_INJECT_BOUNDS" in codes
    assert report["counts"]["detected"] == 1
    assert report["counts"]["missed"] == 0


def test_illegal_transfer_param_detected(tiny_workdir):
    manifest, state, ledger = _clean_build(tiny_workdir)
    item = illegal_transfer_param()
    bad_state = apply_state_tamper(state, item)
    config = InjectionConfig("c2", [item])
    report = run_detection(tiny_workdir, manifest, bad_state, config)
    assert any(f["code"] == "E_INJECT_BOUNDS" for f in report["findings"])
    assert report["counts"]["detected"] == 1


def test_truth_sample_tamper_breaks_manifest_hash(tiny_workdir):
    manifest, state, ledger = _clean_build(tiny_workdir)
    point = manifest["frames"][0]["samples"][0]
    item = truth_sample_tamper(
        "ts", frame=0, row=point["row"], col=point["col"], delta=0.5)
    patched = apply_manifest_tamper(manifest, item)
    findings = validate_manifest(patched)
    assert any(f["code"] == "E_HASH_MISMATCH" for f in findings)
    config = InjectionConfig("c3", [item])
    report = classify(config, findings)
    assert report["counts"]["detected"] == 1


def test_tonemap_constant_tamper_changes_stage_hashes(tiny_workdir):
    manifest, state, ledger = _clean_build(tiny_workdir)
    item = tonemap_constant_tamper()
    bad_state = apply_state_tamper(state, item)
    config = InjectionConfig("c4", [item])
    report = run_detection(tiny_workdir, manifest, bad_state, config)
    # state stays legal, but clean recomputation no longer matches committed
    mismatch = [f for f in report["findings"]
                if f["code"] == "E_HASH_MISMATCH" and f["scope"] == "stage"]
    assert mismatch
    assert report["counts"]["detected"] == 1


def test_stage_replacement_detected_at_frame_and_stage(tiny_workdir):
    manifest, state, ledger = _clean_build(tiny_workdir)
    item = stage_replacement("sr", frame=3, stage="tonemap",
                             replacement={"value": 0.1})
    width, height = 96, 54
    replacement = np.full((height, width, 3), 0.1, dtype=np.float32)
    ctx = InjectionContext({(3, "tonemap"): replacement})
    u8, hashes, _ = process_frame(3, tiny_workdir + "/frames", state,
                                  context=ctx, width=width, height=height)
    commit_frame_output(tiny_workdir, ledger, 3, u8, hashes,
                        seq=ledger["current_seq"], recomputed_stages=list(hashes))
    write_ledger(tiny_workdir, ledger)
    config = InjectionConfig("c5", [item])
    report = run_detection(tiny_workdir, manifest, state, config,
                           context=ctx)
    stage_findings = [f for f in report["findings"] if f["scope"] == "stage"]
    assert any(f["frame"] == 3 and f["stage"] == "tonemap"
               for f in stage_findings)
    assert report["counts"]["detected"] == 1
    assert report["counts"]["missed"] == 0


def test_missed_and_false_positive_classification(tiny_workdir):
    manifest, state, ledger = _clean_build(tiny_workdir)
    # An item the oracle marks as not-expected-to-fire, plus a rogue finding
    # with no injection behind it.
    stealth = stage_replacement(
        "stealth", frame=2, stage="colorspace",
        replacement={"value": 0.2}, expect_detected=False)
    config = InjectionConfig("c6", [stealth])
    rogue_finding = {"code": "E_HASH_MISMATCH", "scope": "stage",
                     "frame": 5, "stage": "transfer",
                     "expected": "x", "observed": "y", "message": "rogue"}
    report = classify(config, [rogue_finding])
    assert report["counts"]["false_positive"] == 1
    fp = report["false_positive"][0]
    assert "no injection item maps" in fp["basis"]

    # A truly missed expected item is reported with a documented basis.
    expected = stage_replacement("gone", frame=2, stage="exposure",
                                 replacement={"value": 0.2})
    report2 = classify(InjectionConfig("c7", [expected]), [])
    assert report2["counts"]["missed"] == 1
    assert "basis" in report2["missed"][0]
