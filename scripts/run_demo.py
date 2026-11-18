"""End-to-end acceptance demo runnable with ``python scripts/run_demo.py``.

The script never touches the network: it regenerates the deterministic input
set, exercises sample/layer/rank/base edits incrementally, reconciles every
step against a from-scratch rebuild, compares concurrency settings and finally
runs the fault-injection suite. All outputs are written under ``demo_out/``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lora_bench.deps import plan_invalidation  # noqa: E402
from lora_bench.generator import build_default_dataset  # noqa: E402
from lora_bench.injection import run_injection_suite  # noqa: E402
from lora_bench.pipeline import run_pipeline  # noqa: E402
from lora_bench.rebuild import compare_outcomes, full_rebuild  # noqa: E402
from lora_bench.utils import read_json, write_json_atomic  # noqa: E402

OUT = os.path.join(ROOT, "demo_out")


def reset():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)


def load_config(name):
    return read_json(os.path.join(ROOT, "configs", name))


def write_samples(input_dir, samples):
    path = os.path.join(input_dir, "samples.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for sample in sorted(samples, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(sample, sort_keys=True) + "\n")


def read_samples(input_dir):
    with open(os.path.join(input_dir, "samples.jsonl"), encoding="utf-8") as h:
        return [json.loads(line) for line in h if line.strip()]


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        raise SystemExit(f"demo assertion failed: {label}")


def main():
    reset()
    input_dir = os.path.join(OUT, "input")
    workspace = os.path.join(OUT, "workspace")
    rebuild_root = os.path.join(OUT, "rebuild")
    build_default_dataset(input_dir)
    config = load_config("base_config.json")
    report = {"steps": []}

    # Step 0: baseline run.
    baseline = run_pipeline(input_dir, workspace, config)
    previous_identities = dict(baseline["manifest"]["task_identities"])
    report["steps"].append(
        {
            "step": "baseline",
            "trained": baseline["trained_tasks"],
            "reused": baseline["reused_tasks"],
        }
    )
    print("baseline trained:", baseline["trained_tasks"])

    # Step 1: no edits -> every task reused, no retraining.
    rerun = run_pipeline(input_dir, workspace, config)
    check(
        "unchanged rerun reuses all adapters",
        rerun["trained_tasks"] == []
        and sorted(rerun["reused_tasks"])
        == sorted(baseline["manifest"]["completed_tasks"]),
    )

    # Step 2: edit one sample -> single task invalidation.
    samples = read_samples(input_dir)
    for sample in samples:
        if sample["sample_id"] == "cls_mod4-001":
            sample["input_tokens"].append(5)
    write_samples(input_dir, samples)
    inc = run_pipeline(input_dir, workspace, config)
    check("sample edit invalidates one task", inc["trained_tasks"] == ["cls_mod4"])
    rebuild = full_rebuild(
        input_dir,
        os.path.join(rebuild_root, "step2"),
        config,
        baseline_identities=previous_identities,
    )
    rebuild["invalidation"] = plan_invalidation(
        previous_identities, rebuild["manifest"]["task_identities"]
    )
    check(
        "step2 incremental equals full rebuild",
        compare_outcomes(inc, rebuild)["equal"],
    )
    previous_identities = dict(inc["manifest"]["task_identities"])

    # Step 3: rank/alpha override -> single task invalidation.
    config_rank = dict(
        config, task_overrides={"reg_cycle": {"rank": 3, "alpha": 7.0}}
    )
    inc = run_pipeline(input_dir, workspace, config_rank)
    check("rank override invalidates one task", inc["trained_tasks"] == ["reg_cycle"])
    rebuild = full_rebuild(
        input_dir,
        os.path.join(rebuild_root, "step3"),
        config_rank,
        baseline_identities=previous_identities,
    )
    rebuild["invalidation"] = plan_invalidation(
        previous_identities, rebuild["manifest"]["task_identities"]
    )
    check(
        "step3 incremental equals full rebuild",
        compare_outcomes(inc, rebuild)["equal"],
    )
    previous_identities = dict(inc["manifest"]["task_identities"])

    # Step 4: target layer whitelist change -> all tasks invalidated.
    config_layers = dict(config_rank, target_layers=[1, 3])
    inc = run_pipeline(input_dir, workspace, config_layers)
    check(
        "layer whitelist change fans out to all tasks",
        sorted(inc["trained_tasks"])
        == sorted(inc["manifest"]["completed_tasks"]),
    )
    rebuild = full_rebuild(
        input_dir,
        os.path.join(rebuild_root, "step4"),
        config_layers,
        baseline_identities=previous_identities,
    )
    rebuild["invalidation"] = plan_invalidation(
        previous_identities, rebuild["manifest"]["task_identities"]
    )
    check(
        "step4 incremental equals full rebuild",
        compare_outcomes(inc, rebuild)["equal"],
    )
    previous_identities = dict(inc["manifest"]["task_identities"])

    # Step 5: base version replacement -> all tasks invalidated.
    shutil.copytree(
        os.path.join(input_dir, "base_v2"),
        os.path.join(input_dir, "base_active"),
        dirs_exist_ok=True,
    )
    inc = run_pipeline(
        input_dir, workspace, config_layers, base_subdir="base_active"
    )
    check(
        "base replacement fans out to all tasks",
        all(
            e["reasons"] == ["base_version"]
            for e in inc["invalidation"]["invalidated"]
        ),
    )
    rebuild = full_rebuild(
        input_dir,
        os.path.join(rebuild_root, "step5"),
        config_layers,
        baseline_identities=previous_identities,
        base_subdir="base_active",
    )
    rebuild["invalidation"] = plan_invalidation(
        previous_identities, rebuild["manifest"]["task_identities"]
    )
    check(
        "step5 incremental equals full rebuild",
        compare_outcomes(inc, rebuild)["equal"],
    )

    # Step 6: concurrency variants must be numerically identical.
    serial_cfg = dict(config_layers, concurrency=1)
    parallel_cfg = dict(config_layers, concurrency=4)
    serial = run_pipeline(
        input_dir,
        os.path.join(OUT, "ws_serial"),
        serial_cfg,
        base_subdir="base_active",
    )
    parallel = run_pipeline(
        input_dir,
        os.path.join(OUT, "ws_parallel"),
        parallel_cfg,
        base_subdir="base_active",
    )
    check(
        "serial and parallel outputs match",
        serial["validation_summary"] == parallel["validation_summary"]
        and serial["invalidation"] == parallel["invalidation"],
    )

    # Step 7: fault injection suite.
    injection_report = run_injection_suite(os.path.join(OUT, "injection"))
    check(
        "all injection scenarios hit expected codes",
        injection_report["hit_count"] == injection_report["total"],
    )

    report["final_validation"] = inc["validation_summary"]
    report["injection"] = injection_report
    write_json_atomic(os.path.join(OUT, "demo_report.json"), report)
    print("\nall demo checks passed; report:", os.path.join(OUT, "demo_report.json"))


if __name__ == "__main__":
    main()
