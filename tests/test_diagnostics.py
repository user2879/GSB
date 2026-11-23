import json

import numpy as np

from workbench import errors
from workbench.config import load_config
from workbench.fixedpoint import FixedPoint
from workbench.fusion import fuse_pair
from workbench.loader import load_manifest
from workbench.pipeline import Pipeline


def test_injection_isolation_and_hits(small_samples, tmp_path):
    inject = small_samples / "inject.json"
    report = Pipeline(small_samples, tmp_path / "w", inject_path=inject).run()
    hits = report["injections_hit"]
    assert {"INJ-001", "INJ-002", "INJ-003", "INJ-004"} <= set(hits)
    codes = {f["code"] for f in report["failed"]}
    assert errors.E_BAND_CORRUPT in codes        # truncated band bytes
    assert errors.E_BAND_NONFINITE in codes      # NaN injected
    assert errors.E_ADJ_UNKNOWN_REF in codes     # ghost adjacency
    assert errors.E_CONFIG_INVALID in codes      # negative min_connected
    # isolation: untouched tiles still classified
    assert "tile_r1_c0" in report["per_tile_class_counts"]
    # inj ids recorded on the failure entries
    by_code = {f["code"]: f for f in report["failed"]}
    assert by_code[errors.E_BAND_CORRUPT]["inj_id"] == "INJ-001"
    assert by_code[errors.E_BAND_NONFINITE]["inj_id"] == "INJ-002"


def test_report_reproducible(small_samples, tmp_path):
    r1 = Pipeline(small_samples, tmp_path / "w1").run()
    r2 = Pipeline(small_samples, tmp_path / "w2").run()
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_report_contents(small_samples, tmp_path):
    report = Pipeline(small_samples, tmp_path / "w").run()
    assert report["config_hash"]
    assert set(report["band_set_versions"].values()) == {"bs-v1"}
    assert len(report["confusion"]["matrix"]) == 4
    assert report["fusion_stats"]
    assert report["budget_trace"]
    for entry in report["budget_trace"]:
        assert "remaining_global" in entry and "consumed" in entry


def test_boundary_sensitivity_event(small_samples, config_path):
    manifest = load_manifest(small_samples)
    config = load_config(config_path)
    fp = FixedPoint(config.fixed_point_frac_bits, config.quantum)
    spec_a = manifest.tiles["tile_r0_c0"]
    spec_b = manifest.tiles["tile_r0_c1"]
    size = manifest.tile_size
    # craft a tie in fixed point whose float reference prefers class 0 while
    # the tie-break fallback (tile A's own class) is 2
    res_a = {"classes": np.full((size, size), 2, dtype=np.uint8),
             "probs": np.zeros((size, size, 4), dtype=np.int64),
             "prob_summary": {"hash": "a", "mean_probs": [0, 0, 0, 0]}}
    res_b = {"classes": np.full((size, size), 3, dtype=np.uint8),
             "probs": np.zeros((size, size, 4), dtype=np.int64),
             "prob_summary": {"hash": "b", "mean_probs": [0, 0, 0, 0]}}
    for res in (res_a, res_b):
        res["probs"][:, :, :] = 1024  # exact four-way tie -> gap 0 <= half quantum
    out = fuse_pair(res_a, res_b, spec_a, spec_b, manifest, config, fp, "va", "vb")
    events = [e for e in out["boundary_events"]
              if e["code"] == errors.E_BOUNDARY_SENSITIVITY]
    assert events, "expected sensitivity events for exact ties"
    flipped = [e for e in events if e["direction"] == "flipped"]
    assert flipped, "expected at least one fixed/float disagreement"
    ev = flipped[0]
    assert ev["flip_pair"] == sorted([ev["fixed_point_class"], ev["reference_class"]])
