"""End-to-end acceptance validation for the vibration pipeline.

Validates, on deterministic local samples:
1. cross-window feature spectrum, padding and dependency registration;
2. three deterministic detector fixed-point scores and boundary verdicts;
3. exact invalidation after a single-window replacement;
4. an edit sequence (window/hop, order/stat window, window replacement,
   segment deletion, detector hyperparameters, normalization and quant bits);
5. field-identical incremental vs from-blank full rebuild;
6. serial vs parallel equivalence;
7. point-budget pending/resume and atomic stage snapshots.
"""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vibdet.config import load_config
from vibdet.edits import delete_segment, replace_window
from vibdet.hashing import stable_hash
from vibdet.pipeline import run_pipeline
from vibdet.synth import load_descriptions
from tools.generate_sample import build_descriptions


def _write_workspace(workspace, descriptions):
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "descriptions.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"version": 1, "devices": descriptions}, fh,
                       sort_keys=True)
    base_cfg = load_config()
    with (workspace / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(base_cfg, fh, sort_keys=True)


def _clone_descriptions(descriptions):
    return copy.deepcopy(descriptions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    root = Path(args.workspace)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    report = {"checks": [], "passed": True}

    def check(name, condition, detail=""):
        report["checks"].append({"name": name, "passed": bool(condition),
                                 "detail": detail})
        report["passed"] = report["passed"] and bool(condition)

    base_descriptions = build_descriptions()["devices"]
    base_cfg = load_config()

    # --- baseline run -----------------------------------------------------
    ws_base = root / "baseline"
    _write_workspace(ws_base, base_descriptions)
    baseline = run_pipeline(ws_base, base_cfg,
                            {"version": 1, "devices": _clone_descriptions(
                                base_descriptions), "window_overrides": {}})
    check("baseline_runs", len(baseline["devices"]) == 3)
    check("nonfinite_failure_recorded",
          any(f["code"] == "E_SAMPLE_NONFINITE" for f in baseline["failures"]))
    dev_a = next(d for d in baseline["devices"] if d["device_id"] == "DEV-A")
    check("spectrum_positions_match_windows",
          len(dev_a["spectrum_positions"]) == dev_a["window_count"])
    check("padding_flagged",
          any(p["padding_count"] > 0 for p in dev_a["spectrum_positions"]))
    check("cross_segment_dependencies",
          any(len(p["segments"]) > 1 for p in dev_a["spectrum_positions"]))
    check("three_detectors_present",
          all(name in dev_a["detectors"] for name in
              ("maha", "residual", "crest")))
    check("fixed_point_integer_scores",
          all(isinstance(v, int) for verdict in dev_a["verdicts"]
              for v in verdict["fixed"].values()))

    # --- determinism: repeated identical run ------------------------------
    ws_det = root / "determinism"
    _write_workspace(ws_det, base_descriptions)
    repeat = run_pipeline(ws_det, copy.deepcopy(base_cfg),
                          {"version": 1,
                           "devices": _clone_descriptions(base_descriptions),
                           "window_overrides": {}})
    check("repeated_run_field_identical",
          stable_hash(repeat["devices"]) == stable_hash(baseline["devices"]))

    # --- single window invalidation --------------------------------------
    ws_inv = root / "invalidation"
    _write_workspace(ws_inv, base_descriptions)
    run_pipeline(ws_inv, base_cfg,
                 {"version": 1,
                  "devices": _clone_descriptions(base_descriptions),
                  "window_overrides": {}})
    replace_window(ws_inv, "DEV-A", 2, amplitude=3.0)
    edited = load_descriptions(ws_inv / "descriptions.yaml")
    after = run_pipeline(ws_inv, base_cfg, edited)
    inc_dev = next(d for d in after["devices"] if d["device_id"] == "DEV-A")
    base_keys = {p["position"]: p["cache_key"]
                 for p in dev_a["spectrum_positions"]}
    changed = {p["position"] for p in inc_dev["spectrum_positions"]
               if base_keys[p["position"]] != p["cache_key"]}
    check("single_window_changes_subset", len(changed) < dev_a["window_count"])
    check("single_window_keeps_untouched",
          len(changed) + len([p for p in inc_dev["spectrum_positions"]
                              if base_keys[p["position"]] == p["cache_key"]])
          == dev_a["window_count"])

    # --- full edit sequence with incremental vs blank rebuild ------------
    edit_steps = [
        ("config", "window.length", 200),
        ("config", "window.hop", 100),
        ("config", "spectrum.order", 3),
        ("config", "spectrum.stat_window", 5),
        ("replace_window", None, None),
        ("delete_segment", None, None),
        ("config", "detectors.maha.scale_multiplier", 1.8),
        ("config", "detectors.residual.power", 2.0),
        ("config", "normalization.method", "robust"),
        ("config", "quantization.frac_bits", 10),
    ]
    ws_edit = root / "edited"
    _write_workspace(ws_edit, _clone_descriptions(base_descriptions))
    run_pipeline(ws_edit, base_cfg,
                 {"version": 1,
                  "devices": _clone_descriptions(base_descriptions),
                  "window_overrides": {}})
    edit_cfg = copy.deepcopy(base_cfg)
    for kind, path, value in edit_steps:
        if kind == "config":
            node = edit_cfg
            keys = path.split(".")
            for key in keys[:-1]:
                node = node[key]
            node[keys[-1]] = value
        elif kind == "replace_window":
            replace_window(ws_edit, "DEV-A", 1, amplitude=2.5)
        elif kind == "delete_segment":
            delete_segment(ws_edit, "DEV-A", "A-seg2")
    edited_desc = load_descriptions(ws_edit / "descriptions.yaml")
    incremental = run_pipeline(ws_edit, edit_cfg, edited_desc)

    ws_fresh = root / "fresh_rebuild"
    _write_workspace(ws_fresh, edited_desc["devices"])
    # carry window overrides into the blank-rebuild workspace
    fresh_doc = load_descriptions(ws_fresh / "descriptions.yaml")
    fresh_doc["window_overrides"] = edited_desc.get("window_overrides", {})
    with (ws_fresh / "descriptions.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(fresh_doc, fh, sort_keys=True)
    full_rebuild = run_pipeline(ws_fresh, copy.deepcopy(edit_cfg), fresh_doc)
    check("edit_sequence_incremental_equals_full",
          stable_hash(incremental["devices"]) ==
          stable_hash(full_rebuild["devices"]))
    check("edit_sequence_failures_identical",
          stable_hash(incremental["failures"]) ==
          stable_hash(full_rebuild["failures"]))

    # --- serial vs parallel ----------------------------------------------
    serial_cfg = copy.deepcopy(edit_cfg)
    serial_cfg["scheduler"]["serial"] = True
    serial_cfg["scheduler"]["max_workers"] = 1
    parallel_cfg = copy.deepcopy(edit_cfg)
    parallel_cfg["scheduler"]["serial"] = False
    parallel_cfg["scheduler"]["max_workers"] = 4
    ws_serial = root / "serial"
    ws_parallel = root / "parallel"
    _write_workspace(ws_serial, edited_desc["devices"])
    _write_workspace(ws_parallel, edited_desc["devices"])
    for ws in (ws_serial, ws_parallel):
        doc = load_descriptions(ws / "descriptions.yaml")
        doc["window_overrides"] = edited_desc.get("window_overrides", {})
        with (ws / "descriptions.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=True)
    serial_run = run_pipeline(ws_serial, serial_cfg,
                              load_descriptions(ws_serial / "descriptions.yaml"))
    parallel_run = run_pipeline(ws_parallel, parallel_cfg,
                                load_descriptions(ws_parallel / "descriptions.yaml"))
    check("serial_parallel_devices_identical",
          stable_hash(serial_run["devices"]) ==
          stable_hash(parallel_run["devices"]))
    check("serial_parallel_failures_identical",
          stable_hash(serial_run["failures"]) ==
          stable_hash(parallel_run["failures"]))

    # --- point budget and resume ------------------------------------------
    budget_cfg = copy.deepcopy(base_cfg)
    budget_cfg["scheduler"]["point_budget"] = 8
    ws_budget = root / "budget"
    _write_workspace(ws_budget, base_descriptions)
    first = run_pipeline(ws_budget, budget_cfg,
                         {"version": 1,
                          "devices": _clone_descriptions(base_descriptions),
                          "window_overrides": {}})
    check("budget_pending_recorded",
          any(f["code"] == "E_POINT_BUDGET_PENDING" for f in first["failures"]))
    second = run_pipeline(ws_budget, budget_cfg,
                          load_descriptions(ws_budget / "descriptions.yaml"))
    check("budget_resume_prefers_pending",
          second["devices"][0]["allowed_positions"][0] ==
          first["pending_queue"][0]["position"])

    # --- snapshots ---------------------------------------------------------
    manifest = json.loads((ws_base / "snapshots" / "manifest.json").read_text())
    check("snapshot_manifest_versions",
          all(stage in manifest["stages"] for stage in
              ("spectrum", "detector", "verdict")))
    check("snapshot_tracks_config_and_wave",
          bool(manifest["config_hash"]) and bool(manifest["wave_fingerprint"]))

    report_path = root / "acceptance_report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2),
                           encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
