"""End-to-end diagnostic run: full -> edits -> incremental -> fresh full -> checks -> report."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rope_diag.pipeline import Pipeline
from rope_diag.pruning import select_policy
from rope_diag.report import COMPARE_SECTIONS, build_report, canonical, write_reports
from rope_diag.samples import apply_edits, generate_all
from rope_diag.sensitivity import diagnose


def load_inputs(sample_dir: Path) -> dict:
    config = json.loads((sample_dir / "config.json").read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in (sample_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    edits = json.loads((sample_dir / "edits.json").read_text(encoding="utf-8"))
    injections = json.loads((sample_dir / "injections.json").read_text(encoding="utf-8"))
    policy_table = json.loads((sample_dir / "policy_table.json").read_text(encoding="utf-8"))
    return {
        "config": config,
        "samples": samples,
        "edits": edits,
        "injections": injections,
        "policy_table": policy_table,
    }


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def main() -> int:
    sample_dir = ROOT / "samples"
    if not (sample_dir / "config.json").exists():
        generate_all(sample_dir)
    inputs = load_inputs(sample_dir)
    config, samples = inputs["config"], inputs["samples"]
    injections, policy_table = inputs["injections"], inputs["policy_table"]

    out_main = fresh_dir(ROOT / "output" / "run_main")
    out_full = fresh_dir(ROOT / "output" / "run_full_after_edits")
    out_serial = fresh_dir(ROOT / "output" / "run_serial")
    out_repeat = fresh_dir(ROOT / "output" / "run_repeat")

    # --- run 1: full with the original config -------------------------------
    policy, decision = select_policy(policy_table, config, len(samples))
    pipe = Pipeline(out_main)
    run1 = pipe.run(config, samples, injections, policy, decision, mode="full")

    # --- run 1b: serial repeat for concurrency-consistency ------------------
    serial_config = dict(config, concurrency=1)
    pipe_serial = Pipeline(out_serial)
    run1_serial = pipe_serial.run(config, samples, injections, policy, decision, mode="full")

    # --- run 1c: straight repeat for reproducibility ------------------------
    pipe_repeat = Pipeline(out_repeat)
    run1_repeat = pipe_repeat.run(config, samples, injections, policy, decision, mode="full")

    # --- run 2: apply edit stream, incremental recompute --------------------
    config2, samples2 = apply_edits(config, samples, inputs["edits"])
    policy2, decision2 = select_policy(policy_table, config2, len(samples2))
    run2 = pipe.run(config2, samples2, injections, policy2, decision2, mode="incremental")

    # --- run 3: fresh full recompute with the edited config -----------------
    pipe_full = Pipeline(out_full)
    run3 = pipe_full.run(config2, samples2, injections, policy2, decision2, mode="full")

    checks = {
        "serial_vs_concurrent": canonical(run1, COMPARE_SECTIONS)
        == canonical(run1_serial, COMPARE_SECTIONS),
        "reproducible_rerun": canonical(run1, COMPARE_SECTIONS)
        == canonical(run1_repeat, COMPARE_SECTIONS),
        "incremental_vs_full": canonical(run2, COMPARE_SECTIONS)
        == canonical(run3, COMPARE_SECTIONS),
        "incremental_reused_artifacts": len(run2["reused"]) > 0,
        "pending_marked": all(
            p["code"] == "E_POINT_BUDGET_PENDING" for p in run3["budget"]["pending"]
        )
        and len(run3["budget"]["pending"]) > 0,
        "injection_codes_present": {
            f["code"] for f in run3["failures"]
        }
        == {"E_NONFINITE_FEATURE", "E_KEY_POS_OOB", "E_BASE_ZERO", "E_SCALE_NONPOSITIVE"},
    }

    # sensitivity diagnosis on the final (edited) configuration
    valid_ids = {
        s["sample_id"]
        for s in samples2
        if s["sample_id"]
        not in {f["sample_id"] for f in run3["failures"]}
    }
    valid_samples = [s for s in samples2 if s["sample_id"] in valid_ids]
    sensitivity = diagnose(
        config2, valid_samples, tuple(policy2["dim_range"]), int(policy2["score_chunk"])
    )

    report = build_report(run3, sensitivity, checks, decision2)
    # invalidation/audit lists come from the incremental run (run2)
    report["invalidation"] = {
        "reused_count": len(run2["reused"]),
        "recomputed_count": len(run2["recomputed"]),
        "invalidated_key_count": len(run2["invalidated_keys"]),
        "reused": run2["reused"],
        "recomputed": run2["recomputed"],
    }
    json_path, md_path = write_reports(report, ROOT / "output" / "reports")

    print(json.dumps(checks, indent=2))
    print(f"report: {json_path}")
    print(f"report: {md_path}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
