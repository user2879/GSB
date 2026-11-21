"""End-to-end acceptance verification.

Checks:
  A. three-channel fixed-point fusion runs and produces confidences
  B. stage-by-stage error propagation is reported for every turn
  C. concurrent (4 workers) and serial (1 worker) runs are field-identical
  D. budget exhaustion marks pending turns with E_FUSE_BUDGET_PENDING and
     keeps a stable deduction sequence
  E. incremental runs after a five-class command sequence are
     field-identical to a from-scratch full rebuild (requirement 7)
  F. fault injection hits all four injection classes with their ids
  G. snapshot staleness is detected via E_SNAPSHOT_STALE
"""

from __future__ import annotations

import copy
import json
import os
import sys

from . import codes, injection, loader
from .cache import StageCache
from .cli import load_dataset
from .interactive import Workbench
from .pipeline import Pipeline
from .snapshot import Snapshotter

NEW_TURN = {
    "turn_id": "T005",
    "speaker": "caller",
    "terms": ["glad", "clear", "stable", "today", "meeting", "warm"],
    "start_ms": 100,
    "end_ms": 1900,
    "sentiment": "pos",
    "tool_calls": [{"tool": "sentiment_polarity",
                    "params": {"pos_count": 4, "neg_count": 0, "term_count": 6}}],
}

NEW_TOOL = {
    "name": "emphasis_score",
    "param_schema": {"peak": "number", "floor": "number"},
    "formula": "(peak - floor) / max(peak, 0.001)",
    "dimension": "sentiment_polarity",
}


def _strip(report, keys=("version",)):
    report = copy.deepcopy(report)
    for key in keys:
        report.pop(key, None)
    return report


def _apply_final_mutations(dataset):
    ds = copy.deepcopy(dataset)
    ds["config"]["weights"]["text"] = 0.65
    ds["config"]["temperature"] = 1.25
    ds["config"]["precision"]["frac_bits"] = 16
    ds["config"]["precision"]["quantum"] = 1
    for idx, turn in enumerate(ds["turns"]):
        if turn["turn_id"] == "T005":
            ds["turns"][idx] = copy.deepcopy(NEW_TURN)
    ds["tools"]["emphasis_score"] = copy.deepcopy(NEW_TOOL)
    ds["tools"].pop("speech_rate", None)
    return ds


def run_verification(work_dir):
    os.makedirs(work_dir, exist_ok=True)
    samples_dir = os.path.join(work_dir, "samples")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from generate_samples import generate
    generate(samples_dir)
    dataset = load_dataset(samples_dir)
    checks = {}

    # -- A/B: baseline fusion + error propagation -------------------------
    pipeline = Pipeline(dataset["turns"], dataset["acoustics"], dataset["tools"],
                        dataset["lexicon"], dataset["config"], cache=StageCache())
    baseline = pipeline.run()
    checks["fusion_completed"] = (
        baseline["summary"]["completed_turns"] == baseline["summary"]["total_turns"]
        and all(t["confidence_int"] is not None for t in baseline["turns"]))
    checks["error_propagation_present"] = all(
        t["error_propagation"] and set(t["error_propagation"]["stage_diffs"])
        == {"normalize", "quantize", "weighted_sum", "clip"}
        for t in baseline["turns"])

    # -- C: concurrency consistency ---------------------------------------
    cfg_serial = copy.deepcopy(dataset["config"])
    cfg_serial["concurrency"] = 1
    serial = Pipeline(dataset["turns"], dataset["acoustics"], dataset["tools"],
                      dataset["lexicon"], cfg_serial, cache=StageCache()).run()
    # concurrency is an execution parameter, so the config fingerprint is
    # excluded from this comparison; every computed field must match.
    checks["concurrency_consistent"] = (
        _strip(serial, ("config_fingerprint",))
        == _strip(baseline, ("config_fingerprint",)))

    # -- D: budget exhaustion ---------------------------------------------
    cfg_budget = copy.deepcopy(dataset["config"])
    cfg_budget["fusion_point_budget"] = 6  # room for exactly two turns
    budgeted = Pipeline(dataset["turns"], dataset["acoustics"], dataset["tools"],
                        dataset["lexicon"], cfg_budget, cache=StageCache()).run()
    pending = budgeted["budget"]["pending"]
    checks["budget_pending"] = (
        len(pending) == len(dataset["turns"]) - 2
        and all(p["code"] == codes.E_FUSE_BUDGET_PENDING for p in pending)
        and [d["turn_id"] for d in budgeted["budget"]["deductions"]]
        == sorted(t["turn_id"] for t in dataset["turns"])[:2])

    # -- E: incremental vs full rebuild ------------------------------------
    run_dir_inc = os.path.join(work_dir, "run_incremental")
    workbench = Workbench(dataset["turns"], dataset["acoustics"], dataset["tools"],
                          dataset["lexicon"], copy.deepcopy(dataset["config"]),
                          dataset["load_failures"], run_dir_inc)
    workbench.apply("run")
    workbench.apply("set-weight text 0.65")
    workbench.apply("set-temperature 1.25")
    workbench.apply("set-precision 16 1")
    workbench.apply("replace-turn T005 " + json.dumps(NEW_TURN))
    workbench.apply("add-tool " + json.dumps(NEW_TOOL))
    mutation_response = workbench.apply("remove-tool speech_rate")
    incremental_mutation = mutation_response["report"]
    noop_response = workbench.apply("run")
    incremental_warm = noop_response["report"]

    rebuilt_dataset = _apply_final_mutations(dataset)
    run_dir_full = os.path.join(work_dir, "run_rebuild")
    rebuilder = Workbench(rebuilt_dataset["turns"], rebuilt_dataset["acoustics"],
                          rebuilt_dataset["tools"], rebuilt_dataset["lexicon"],
                          rebuilt_dataset["config"], rebuilt_dataset["load_failures"],
                          run_dir_full)
    rebuild_cold = rebuilder.apply("run")["report"]
    rebuild_warm = rebuilder.apply("run")["report"]

    data_keys = ("config_fingerprint", "precision_tiers", "normalize_stats", "turns",
                 "failures", "budget", "summary")
    incremental_data = {k: incremental_mutation[k] for k in data_keys}
    rebuild_data = {k: rebuild_cold[k] for k in data_keys}
    checks["incremental_equals_rebuild_data"] = incremental_data == rebuild_data
    checks["incremental_equals_rebuild_cache_lists"] = (
        incremental_warm["reuse"] == rebuild_warm["reuse"]
        and incremental_warm["recompute"] == rebuild_warm["recompute"]
        and len(incremental_warm["recompute"]) == 0)

    # -- F: fault injection -------------------------------------------------
    injection_path = os.path.join(os.path.dirname(__file__), "..", "config",
                                  "injection.yaml")
    injections = injection.load_injections(injection_path)
    i_turns, i_acoustics, i_tools, i_config, hits, config_failures = \
        injection.apply_injections(dataset["turns"], dataset["acoustics"],
                                   dataset["tools"], dataset["config"], injections)
    injected = Pipeline(i_turns, i_acoustics, i_tools, dataset["lexicon"],
                        i_config, cache=StageCache()).run()
    injected_failures = config_failures + injected["failures"]
    injected_codes = {f["code"] for f in injected_failures}
    turn_codes = {c for t in injected["turns"] for c in t["codes"]}
    checks["injection_hits"] = len(hits) == 4 and len({h["id"] for h in hits}) == 4
    checks["injection_codes"] = {
        "nonfinite_acoustic": codes.E_LOAD_ACOUSTIC_NONFINITE in injected_codes,
        "tool_dimension_mismatch": codes.E_FUSE_DIM_INCOMPAT in injected_codes
        or codes.E_FUSE_DIM_INCOMPAT in turn_codes,
        "precision_out_of_range": codes.E_LOAD_PRECISION in injected_codes,
        "weights_all_zero": codes.E_FUSE_WEIGHT_SUM_ZERO in turn_codes,
    }
    checks["injection_all_codes_hit"] = all(checks["injection_codes"].values())

    # -- G: snapshot staleness ----------------------------------------------
    snap_dir = os.path.join(work_dir, "run_snapshot")
    snapshotter = Snapshotter(snap_dir)
    snapshotter.commit(1, "hash-a", {"report": {"v": 1}})
    stale = snapshotter.check_stale("hash-b")
    fresh = snapshotter.check_stale("hash-a")
    checks["snapshot_stale_detected"] = (
        len(stale) == 1 and stale[0]["code"] == codes.E_SNAPSHOT_STALE and fresh == [])

    checks["all_passed"] = all(
        v for k, v in checks.items()
        if isinstance(v, bool) and k != "all_passed")
    return {
        "checks": checks,
        "all_passed": checks["all_passed"],
        "baseline_summary": baseline["summary"],
        "injection_hits": hits,
    }
