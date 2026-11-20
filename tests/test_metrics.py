import json
import os

import numpy as np

from hdr_sdr_pipeline.config import ProcessingState
from hdr_sdr_pipeline.engine import IncrementalEngine
from hdr_sdr_pipeline.hashing import canonical_json
from hdr_sdr_pipeline.ledger import new_ledger
from hdr_sdr_pipeline.metrics import (
    compute_frame_metrics, compute_interval_metrics, default_intervals,
)
from hdr_sdr_pipeline.pipeline import process_frame


def _build(work_dir):
    manifest = json.load(open(os.path.join(work_dir, "truth.json")))
    state = ProcessingState(tonemap_operator="aces")
    ledger = new_ledger(state, 8, manifest["manifest_hash"])
    engine = IncrementalEngine(work_dir, ledger, manifest, state)
    engine.build_all()
    return manifest, state, ledger, engine


def test_metrics_derived_only_from_output_and_truth(tiny_workdir):
    manifest, state, ledger, engine = _build(tiny_workdir)
    u8 = engine.load_output(0)
    independent = compute_frame_metrics(
        u8, manifest["frames"][0], state, manifest["regions"]
    )
    committed = ledger["metrics"]["0"]
    for key in ("sample_luma_bias", "shadow_rel_error",
                "highlight_rel_error", "peak_luma_bias", "mean_luma_bias"):
        assert independent[key] == committed[key]


def test_metrics_byte_identical_on_repeated_calculation(tiny_workdir):
    manifest, state, ledger, engine = _build(tiny_workdir)
    u8 = engine.load_output(1)
    m1 = compute_frame_metrics(u8, manifest["frames"][1], state,
                               manifest["regions"])
    m2 = compute_frame_metrics(u8, manifest["frames"][1], state,
                               manifest["regions"])
    assert canonical_json(m1) == canonical_json(m2)


def test_interval_pooling_equal_weights(tiny_workdir):
    manifest, state, ledger, engine = _build(tiny_workdir)
    members = list(range(0, 4))
    pooled = compute_interval_metrics(
        [ledger["metrics"][str(f)] for f in members]
    )
    vals = [ledger["metrics"][str(f)]["sample_luma_bias"] for f in members]
    assert pooled["sample_luma_bias_mean"] == sum(vals) / len(vals)


def test_endpoint_bias_is_finite_in_shadows(tiny_workdir):
    manifest, state, ledger, engine = _build(tiny_workdir)
    m = ledger["metrics"]["0"]
    assert np.isfinite(m["shadow_rel_error"])
    assert m["shadow_rel_error"] >= 0.0
    assert m["highlight_rel_error"] >= 0.0
