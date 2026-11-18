"""Incremental vs full-rebuild equivalence across widths and bucket counts."""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ab_workbench.config import load_config
from ab_workbench.edits import (
    append_batch,
    correct_one_record,
    read_jsonl,
    synthesize_new_batch,
    update_config,
)
from ab_workbench.generator import generate
from ab_workbench.loader import load_attributes, load_injections
from ab_workbench.pipeline import Workbench

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORK = os.path.join(ROOT, "tmp", "equiv")

COMPARED_FIELDS = (
    "group_stats",
    "estimation",
    "attribution",
    "failures",
)
MATRIX = [(6, 18), (4, 24)]


def _fresh_dirs(tag: str):
    base = os.path.join(WORK, tag)
    dirs = {
        "root": base,
        "snapshot": os.path.join(base, "snapshot"),
        "cache": os.path.join(base, "cache"),
        "tmp": os.path.join(base, "tmp"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _run(tag: str, sample_dir: str, concurrent: bool, injections=None):
    dirs = _fresh_dirs(tag)
    config = load_config(os.path.join(sample_dir, "experiment_config.json"))
    bench = Workbench(
        config=config,
        snapshot_dir=dirs["snapshot"],
        cache_dir=dirs["cache"],
        tmp_dir=dirs["tmp"],
        concurrent=concurrent,
    )
    result = bench.run(
        log_path=os.path.join(sample_dir, "exposures.jsonl"),
        attributes_path=os.path.join(sample_dir, "user_attributes.json"),
        injections=injections or [],
    )
    return result.to_dict()


def _run_persistent(tag: str, sample_dir: str):
    """Run against a stable cache/snapshot dir that survives across edits."""
    dirs = _fresh_dirs(tag)
    config = load_config(os.path.join(sample_dir, "experiment_config.json"))
    bench = Workbench(
        config=config,
        snapshot_dir=dirs["snapshot"],
        cache_dir=dirs["cache"],
        tmp_dir=dirs["tmp"],
        concurrent=True,
    )
    result = bench.run(
        log_path=os.path.join(sample_dir, "exposures.jsonl"),
        attributes_path=os.path.join(sample_dir, "user_attributes.json"),
        injections=[],
    )
    return result.to_dict()


def _comparable(report):
    return {field: copy.deepcopy(report[field]) for field in COMPARED_FIELDS}


def _assert_equal(left, right, label, diffs):
    if left != right:
        diffs.append(label)


def run_matrix():
    diffs = []
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK, exist_ok=True)
    for width, buckets in MATRIX:
        tag = f"w{width}-b{buckets}"
        sample_inc = os.path.join(WORK, f"{tag}-inc")
        sample_full = os.path.join(WORK, f"{tag}-full")
        generate(sample_inc, decimal_width=width, num_buckets=buckets)
        generate(sample_full, decimal_width=width, num_buckets=buckets)

        rng = np.random.default_rng(9000 + width + buckets)
        attributes = load_attributes(os.path.join(sample_inc, "user_attributes.json"))
        user_ids = sorted(attributes.keys())

        steps = []

        # Step 1: baseline run (incremental workspace, persistent cache)
        persistent_tag = f"{tag}-inc-persistent"
        baseline = _run_persistent(persistent_tag, sample_inc)
        steps.append("baseline")

        # Step 2: append a new exposure batch
        existing = read_jsonl(os.path.join(sample_inc, "exposures.jsonl"))
        next_index = max(int(r["request_id"][1:]) for r in existing) + 1
        new_rows = synthesize_new_batch(next_index, 12, 6, "control", user_ids, rng)
        append_batch(os.path.join(sample_inc, "exposures.jsonl"), new_rows)
        _run_persistent(persistent_tag, sample_inc)
        steps.append("append_batch")

        # Step 3: single record correction
        target = existing[0]
        correct_one_record(
            os.path.join(sample_inc, "exposures.jsonl"),
            target["request_id"],
            target["item_id"],
            {"clicked": 1 - target["clicked"], "score": 0.424242},
        )
        _run_persistent(persistent_tag, sample_inc)
        steps.append("record_correction")

        # Step 4: config change (min sample size)
        update_config(
            os.path.join(sample_inc, "experiment_config.json"),
            {"min_sample_size": 150},
        )
        _run_persistent(persistent_tag, sample_inc)
        steps.append("config_change")

        # Step 5: decimal width adjustment
        update_config(
            os.path.join(sample_inc, "experiment_config.json"),
            {"decimal_width": width + 2},
        )
        width_report = _run_persistent(persistent_tag, sample_inc)
        steps.append("width_change")

        # Mirror the identical final state into the full-rebuild workspace.
        shutil.copyfile(
            os.path.join(sample_inc, "exposures.jsonl"),
            os.path.join(sample_full, "exposures.jsonl"),
        )
        shutil.copyfile(
            os.path.join(sample_inc, "experiment_config.json"),
            os.path.join(sample_full, "experiment_config.json"),
        )
        full_report = _run(f"{tag}-full-final", sample_full, concurrent=True)
        serial_report = _run(f"{tag}-serial-final", sample_full, concurrent=False)

        inc_cmp = _comparable(width_report)
        full_cmp = _comparable(full_report)
        serial_cmp = _comparable(serial_report)
        for field in COMPARED_FIELDS:
            _assert_equal(inc_cmp[field], full_cmp[field], f"{tag}:{field}:inc-vs-full", diffs)
            _assert_equal(full_cmp[field], serial_cmp[field], f"{tag}:{field}:full-vs-serial", diffs)

        # Failure list ordering must be identical and sorted deterministically.
        inc_codes = [f["code"] for f in width_report["failures"]]
        full_codes = [f["code"] for f in full_report["failures"]]
        _assert_equal(inc_codes, full_codes, f"{tag}:failure_order", diffs)

        # Re-run on unchanged inputs over the same persistent cache: full reuse.
        reuse_report = _run_persistent(persistent_tag, sample_inc)
        if not reuse_report["cache"]["reused_keys"]:
            diffs.append(f"{tag}:expected_cache_reuse")
        if reuse_report["cache"]["computed_keys"]:
            diffs.append(f"{tag}:unexpected_recompute_on_identical_input")

    return diffs, MATRIX


if __name__ == "__main__":
    diffs, matrix = run_matrix()
    print("matrix:", matrix)
    if diffs:
        print("EQUIVALENCE FAILURES:")
        for diff in diffs:
            print(" -", diff)
        raise SystemExit(1)
    print("ALL INCREMENTAL/FULL/SERIAL REPORTS ARE FIELD-FOR-FIELD EQUIVALENT")
