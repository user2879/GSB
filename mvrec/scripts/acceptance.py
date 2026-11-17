"""End-to-end acceptance driver for the deterministic reconstruction pipeline.

Runs four acceptance checks against freshly generated local samples:
  1. grouping decision is auditable and concurrency-independent
  2. dual-layer budget suspension and resume produce the full cloud
  3. serial and parallel point clouds / ordering are field-for-field equal
  4. injection faults map to the expected business codes
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mvrec.cli import run_pipeline
from mvrec.samplegen import generate_sample


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def main():
    workspace = tempfile.mkdtemp(prefix="mvrec_accept_")
    sample = os.path.join(workspace, "sample")
    generate_sample(sample, 8)
    base_config = load_json(os.path.join(os.path.dirname(__file__), "..",
                                         "configs", "recon_config.json"))
    results = []

    def check(name, condition, detail):
        results.append({"check": name, "passed": bool(condition), "detail": detail})
        print("[%s] %s -- %s" % ("PASS" if condition else "FAIL", name, detail))

    # 1. grouping decision stability across concurrency levels
    cfg_serial = dict(base_config, thread_count=1)
    cfg_parallel = dict(base_config, thread_count=4)
    p_serial = os.path.join(workspace, "c_serial.json")
    p_parallel = os.path.join(workspace, "c_parallel.json")
    write_json(p_serial, cfg_serial)
    write_json(p_parallel, cfg_parallel)
    r_serial, _ = run_pipeline(sample, p_serial, None,
                               os.path.join(workspace, "snap_serial"),
                               os.path.join(workspace, "r_serial.json"))
    r_parallel, _ = run_pipeline(sample, p_parallel, None,
                                 os.path.join(workspace, "snap_parallel"),
                                 os.path.join(workspace, "r_parallel.json"))
    check(
        "grouping decision concurrency-independent",
        r_serial["grouping_decision"] == r_parallel["grouping_decision"],
        "selected=%s" % r_serial["grouping_decision"]["selected"],
    )

    # 2. dual-layer budget suspend + resume
    cfg_tight = dict(base_config, point_budget=120, thread_count=2)
    p_tight = os.path.join(workspace, "c_tight.json")
    write_json(p_tight, cfg_tight)
    snap_resume = os.path.join(workspace, "snap_resume")
    r_tight, _ = run_pipeline(sample, p_tight, None, snap_resume,
                              os.path.join(workspace, "r_tight.json"))
    suspended = "E_POINT_BUDGET_EXHAUSTED" in {f["code"] for f in r_tight["failures"]}
    cfg_loose = dict(base_config, point_budget=100000, thread_count=4)
    p_loose = os.path.join(workspace, "c_loose.json")
    write_json(p_loose, cfg_loose)
    r_resume, _ = run_pipeline(sample, p_loose, None, snap_resume,
                               os.path.join(workspace, "r_resume.json"))
    check(
        "point budget suspension recorded with stable pending queue",
        suspended and r_tight["pending_queue"] == sorted(r_tight["pending_queue"]),
        "pending=%s" % r_tight["pending_queue"],
    )
    check(
        "resume completes every group after budget increase",
        r_resume["pending_queue"] == [] and r_resume["point_cloud_size"] > 0,
        "cloud_size=%d committed=%s" % (
            r_resume["point_cloud_size"],
            r_resume["resume_point"]["committed_groups"],
        ),
    )

    # zero-slot budget business code
    cfg_no_slot = dict(base_config, slot_budget=0)
    p_no_slot = os.path.join(workspace, "c_no_slot.json")
    write_json(p_no_slot, cfg_no_slot)
    r_no_slot, _ = run_pipeline(sample, p_no_slot, None,
                                os.path.join(workspace, "snap_no_slot"),
                                os.path.join(workspace, "r_no_slot.json"))
    check(
        "zero slot budget maps to E_SLOT_BUDGET_EXHAUSTED",
        "E_SLOT_BUDGET_EXHAUSTED" in {f["code"] for f in r_no_slot["failures"]},
        "pending=%s" % r_no_slot["pending_queue"],
    )

    # 3. serial vs parallel field-for-field equality
    cloud_serial = load_json(os.path.join(
        workspace, "snap_serial", "merge_v000001", "point_cloud.json"))
    cloud_parallel = load_json(os.path.join(
        workspace, "snap_parallel", "merge_v000001", "point_cloud.json"))
    check(
        "serial and parallel point clouds are field-for-field equal",
        cloud_serial == cloud_parallel
        and r_serial["deduction_sequence"] == r_parallel["deduction_sequence"]
        and r_serial["budget_curve"] == r_parallel["budget_curve"],
        "points=%d" % len(cloud_serial),
    )

    # 4. injection business codes
    injection_path = os.path.join(os.path.dirname(__file__), "..",
                                  "configs", "injections.json")
    injections = load_json(injection_path)
    stage_entries = [
        entry for entry in injections["injections"]
        if not (entry["effects"].get("slot_budget_zero")
                or entry["effects"].get("point_budget_zero"))
    ]
    stage_injection_path = os.path.join(workspace, "injections_stage.json")
    write_json(stage_injection_path, {"injections": stage_entries})
    r_inject, _ = run_pipeline(sample, p_parallel, stage_injection_path,
                               os.path.join(workspace, "snap_inject"),
                               os.path.join(workspace, "r_inject.json"))
    codes = {failure["code"] for failure in r_inject["failures"]}
    slot_injection_path = os.path.join(workspace, "injections_slot.json")
    write_json(slot_injection_path, {"injections": [
        entry for entry in injections["injections"]
        if entry["effects"].get("slot_budget_zero")
    ]})
    r_slot_inject, _ = run_pipeline(
        sample, p_parallel, slot_injection_path,
        os.path.join(workspace, "snap_inject_slot"),
        os.path.join(workspace, "r_inject_slot.json"))
    codes |= {failure["code"] for failure in r_slot_inject["failures"]}
    point_injection_path = os.path.join(workspace, "injections_point.json")
    write_json(point_injection_path, {"injections": [
        entry for entry in injections["injections"]
        if entry["effects"].get("point_budget_zero")
    ]})
    r_point_inject, _ = run_pipeline(
        sample, p_parallel, point_injection_path,
        os.path.join(workspace, "snap_inject_point"),
        os.path.join(workspace, "r_inject_point.json"))
    codes |= {failure["code"] for failure in r_point_inject["failures"]}
    all_hits = sorted(set(r_inject["hit_injection_ids"])
                      | set(r_slot_inject["hit_injection_ids"])
                      | set(r_point_inject["hit_injection_ids"]))
    expected = {
        "E_PNG_INVALID", "E_CAMERA_INVALID", "E_PARALLAX_INVALID",
        "E_FEATURE_NONFINITE", "E_SLOT_BUDGET_EXHAUSTED",
        "E_POINT_BUDGET_EXHAUSTED",
    }
    check(
        "injection faults map to expected business codes",
        expected.issubset(codes),
        "hits=%s codes=%s" % (all_hits, sorted(codes)),
    )

    summary_path = os.path.join(workspace, "acceptance_summary.json")
    write_json(summary_path, results)
    all_passed = all(item["passed"] for item in results)
    print("\nacceptance summary written to %s" % summary_path)
    print("OVERALL: %s (%d/%d checks passed)"
          % ("PASS" if all_passed else "FAIL",
             sum(1 for item in results if item["passed"]), len(results)))
    shutil.rmtree(workspace, ignore_errors=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
