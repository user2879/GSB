import json
import os

import numpy as np

from hdr_sdr_pipeline.config import ProcessingState
from hdr_sdr_pipeline.detector import run_detection
from hdr_sdr_pipeline.engine import IncrementalEngine
from hdr_sdr_pipeline.injection import InjectionConfig, stage_replacement
from hdr_sdr_pipeline.ledger import commit_frame_output, new_ledger
from hdr_sdr_pipeline.locate import locate_first_affected, minimal_counterexample
from hdr_sdr_pipeline.persistence import write_ledger
from hdr_sdr_pipeline.pipeline import InjectionContext, process_frame


def _prepare(tiny_workdir):
    manifest = json.load(open(os.path.join(tiny_workdir, "truth.json")))
    state = ProcessingState()
    ledger = new_ledger(state, 8, manifest["manifest_hash"])
    engine = IncrementalEngine(tiny_workdir, ledger, manifest, state)
    engine.build_all()
    write_ledger(tiny_workdir, ledger)
    return manifest, state, ledger


def _replace_and_commit(work_dir, ledger, state, frame, stage, value):
    width, height = 96, 54
    dtype = np.uint8 if stage == "quantize" else np.float32
    arr = np.full((height, width, 3), value, dtype=dtype)
    ctx = InjectionContext({(frame, stage): arr})
    u8, hashes, _ = process_frame(frame, work_dir + "/frames", state,
                                  context=ctx, width=width, height=height)
    commit_frame_output(work_dir, ledger, frame, u8, hashes,
                        seq=ledger["current_seq"],
                        recomputed_stages=list(hashes))
    write_ledger(work_dir, ledger)
    return ctx


def test_first_affected_pixel_ordering(tiny_workdir):
    manifest, state, ledger = _prepare(tiny_workdir)
    ctx = _replace_and_commit(tiny_workdir, ledger, state, 3, "tonemap", 0.1)
    config = InjectionConfig(
        "loc", [stage_replacement("s1", 3, "tonemap", {"value": 0.1})]
    )
    first, report, all_locs = locate_first_affected(
        tiny_workdir, manifest, state, config, context=ctx
    )
    assert first["frame"] == 3
    # row-major first pixel of the frame (constant replacement differs at 0,0)
    assert (first["row"], first["col"]) == (0, 0)
    assert first["error_code"] == "E_HASH_MISMATCH"
    assert first["injection_item"] == "s1"
    assert "error_interval" in first and "observed_value" in first


def test_frame_ordering_across_multiple_faults(tiny_workdir):
    manifest, state, ledger = _prepare(tiny_workdir)
    ctx1 = _replace_and_commit(tiny_workdir, ledger, state, 5, "transfer", 0.9)
    ctx2 = _replace_and_commit(tiny_workdir, ledger, state, 2, "transfer", 0.1)
    replacements = dict(ctx1.stage_replacements)
    replacements.update(ctx2.stage_replacements)
    ctx = InjectionContext(replacements)
    config = InjectionConfig(
        "multi",
        [
            stage_replacement("late", 5, "transfer", {"value": 0.9}),
            stage_replacement("early", 2, "transfer", {"value": 0.1}),
        ],
    )
    first, _r, all_locs = locate_first_affected(
        tiny_workdir, manifest, state, config, context=ctx
    )
    assert first["frame"] == 2  # ascending frame number wins


def test_minimal_counterexample_removable_elements(tiny_workdir):
    manifest, state, ledger = _prepare(tiny_workdir)
    ctx_keep = _replace_and_commit(tiny_workdir, ledger, state, 4,
                                   "tonemap", 0.2)
    ctx_a = _replace_and_commit(tiny_workdir, ledger, state, 1,
                                "exposure", 0.3)
    ctx_b = _replace_and_commit(tiny_workdir, ledger, state, 6,
                                "colorspace", 0.4)
    config = InjectionConfig(
        "mce_test",
        [
            stage_replacement("keep", 4, "tonemap", {"value": 0.2}),
            stage_replacement("drop_a", 1, "exposure", {"value": 0.3}),
            stage_replacement("drop_b", 6, "colorspace", {"value": 0.4}),
        ],
    )

    def builder(items):
        reps = {}
        for it in items:
            if it.kind != "stage_replacement":
                continue
            value = it.payload["replacement"]["value"]
            reps[(it.frame, it.stage)] = np.full(
                (54, 96, 3), value, dtype=np.float32
            )
        return InjectionContext(reps)

    mce = minimal_counterexample(
        tiny_workdir, manifest, state, config, context_builder=builder
    )
    # the MCE is a single irreducible element and still reproduces the code
    assert len(mce.items) == 1
    mce_report = run_detection(
        tiny_workdir, manifest, state, mce,
        context=builder(mce.items),
    )
    assert any(f["code"] == "E_HASH_MISMATCH"
               for f in mce_report["findings"])
    # determinism: repeated reduction yields the same result
    mce2 = minimal_counterexample(
        tiny_workdir, manifest, state, config, context_builder=builder
    )
    assert [it.item_id for it in mce2.items] == [it.item_id for it in mce.items]
    # minimality proof exercised inside minimal_counterexample: removing
    # the single surviving element leaves an empty config, which by contract
    # cannot be a counterexample (the helper asserts this before returning).
    assert mce.items and all(
        it.item_id in {"keep", "drop_a", "drop_b"} for it in mce.items
    )
