"""Repeatable acceptance verification against real loopback shard processes."""

from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .canonical import write_json
from .config import load_project
from .edits import apply_edit_sequence, project_to_config
from .engine import equivalence_projection, run_build
from .sample import generate_sample


def _fresh(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _codes(result) -> dict[str, int]:
    codes = dict(result.report["business_code_counts"])
    arbitration = result.ledger["arbitration_stats"]
    if arbitration.get("accepted_compatible"):
        codes["E_RECEIPT_STALE_COMPATIBLE"] = arbitration["accepted_compatible"]
    return dict(sorted(codes.items()))


def run_verification(out_dir: str | Path = "artifacts/verification", *, seed: int = 17) -> dict[str, Any]:
    out_dir = Path(out_dir)
    _fresh(out_dir)
    sample_dir = out_dir / "sample"
    config = generate_sample(sample_dir, seed=seed, port_start=19500)
    project, issues = load_project(config)
    assert issues == []
    started_ns = time.monotonic_ns()

    checks: dict[str, Any] = {}

    baseline_work = out_dir / "baseline-incremental"
    blank_work = out_dir / "baseline-full"
    baseline_incremental = run_build(config, baseline_work, mode="incremental", workers=3, global_version=1)
    baseline_full = run_build(config, blank_work, mode="full", workers=3, global_version=1)
    checks["baseline_equivalence"] = equivalence_projection(baseline_incremental) == equivalence_projection(baseline_full)

    edits = json.loads((sample_dir / "edits.json").read_text(encoding="utf-8"))["edits"]
    current = project
    edit_results: list[dict[str, Any]] = []
    for index, edit in enumerate(edits, start=1):
        current, edit_issues = apply_edit_sequence(current, [edit])
        assert edit_issues == []
        edit_config = out_dir / "configs" / f"edit-{index:02d}.json"
        write_json(edit_config, project_to_config(current))
        incremental = run_build(
            edit_config,
            baseline_work,
            mode="incremental",
            workers=3,
            global_version=index + 1,
        )
        full = run_build(
            edit_config,
            out_dir / "edits-full" / f"edit-{index:02d}",
            mode="full",
            workers=3,
            global_version=index + 1,
        )
        equal = equivalence_projection(incremental) == equivalence_projection(full)
        edit_results.append(
            {
                "edit_id": edit["edit_id"],
                "kind": edit["type"],
                "incremental_equals_full": equal,
                "entries_count": incremental.ledger["entries_count"],
                "shard_entry_counts": incremental.ledger["shard_entry_counts"],
                "reuse_counts": incremental.report["reuse"]["counts"],
                "business_codes": _codes(incremental),
                "pending_shards": incremental.report["pending_shards"],
                "failure_count": len(incremental.failure_list),
            }
        )
        assert equal
    checks["edit_sequence_equivalence"] = all(item["incremental_equals_full"] for item in edit_results)
    checks["bucket_level_reuse_observed"] = sum(
        item["reuse_counts"]["bucket_reuse"] for item in edit_results
    ) > 0

    serial_work = out_dir / "serial-parallel" / "serial"
    parallel_work = out_dir / "serial-parallel" / "parallel"
    shutil.copytree(blank_work, serial_work)
    shutil.copytree(blank_work, parallel_work)
    append_config = out_dir / "configs" / "append-only.json"
    append_state, append_issues = apply_edit_sequence(project, edits[:1])
    assert append_issues == []
    write_json(append_config, project_to_config(append_state))
    serial = run_build(append_config, serial_work, mode="incremental", workers=1, global_version=2)
    parallel = run_build(append_config, parallel_work, mode="incremental", workers=8, global_version=2)
    checks["serial_parallel_equivalence"] = equivalence_projection(serial) == equivalence_projection(parallel)
    checks["serial_parallel_failure_order_equal"] = serial.failure_list == parallel.failure_list

    low_budget_raw = copy.deepcopy(project_to_config(project))
    low_budget_raw["budget"] = {"global_term_entry_budget": 30, "per_shard_term_entry_limit": 80}
    low_budget_config = out_dir / "configs" / "low-budget.json"
    write_json(low_budget_config, low_budget_raw)
    low_budget = run_build(low_budget_config, out_dir / "budget-low", mode="full", workers=3, global_version=1)
    checks["budget_pending_codes"] = _codes(low_budget)

    hard_limit_raw = copy.deepcopy(project_to_config(project))
    hard_limit_raw["budget"] = {"global_term_entry_budget": 240, "per_shard_term_entry_limit": 20}
    hard_limit_config = out_dir / "configs" / "hard-limit.json"
    write_json(hard_limit_config, hard_limit_raw)
    hard_limit = run_build(hard_limit_config, out_dir / "budget-hard-limit", mode="full", workers=3, global_version=1)
    checks["per_shard_abort_codes"] = _codes(hard_limit)

    stale_raw = copy.deepcopy(project_to_config(project))
    stale_raw["layout"]["shards"][0]["revision"] = 1
    stale_config = out_dir / "configs" / "stale-revision.json"
    write_json(stale_config, stale_raw)
    stale = run_build(stale_config, out_dir / "stale-revision", mode="full", workers=3, global_version=1)
    checks["stale_arbitration"] = stale.ledger["arbitration_stats"]
    checks["stale_arbitration_codes"] = _codes(stale)

    recovery_work = out_dir / "recovery"
    recovery_baseline = run_build(config, recovery_work, mode="full", workers=3, global_version=1)
    staged = run_build(config, recovery_work, mode="incremental", workers=3, global_version=2, commit=False)
    recovered = run_build(config, recovery_work, mode="incremental", workers=3, global_version=2)
    checks["recovery"] = {
        "baseline_snapshot": recovery_baseline.snapshot_id,
        "staged_path": staged.staging_path,
        "recovered_snapshot": recovered.snapshot_id,
        "decision": recovered.recovery["decision"],
        "codes": [issue["code"] for issue in recovered.recovery["issues"]],
        "orphan_directories": recovered.recovery["orphan_directories"],
    }

    report = {
        "schema_version": 1,
        "generated_sample": str(sample_dir),
        "checks": checks,
        "baseline": {
            "entries_count": baseline_incremental.ledger["entries_count"],
            "shard_entry_counts": baseline_incremental.ledger["shard_entry_counts"],
            "snapshot_id": baseline_incremental.snapshot_id,
            "arbitration": baseline_incremental.ledger["arbitration_stats"],
            "recall": [
                {"query_id": item["query_id"], "selected": [doc["doc_id"] for doc in item["selected"]]}
                for item in baseline_incremental.recall
            ],
        },
        "edits": edit_results,
        "low_budget": {
            "entries_count": low_budget.ledger["entries_count"],
            "pending_shards": low_budget.report["pending_shards"],
            "codes": _codes(low_budget),
        },
        "hard_limit": {
            "entries_count": hard_limit.ledger["entries_count"],
            "codes": _codes(hard_limit),
        },
        "stale_revision": {
            "shard_entry_counts": stale.ledger["shard_entry_counts"],
            "arbitration": stale.ledger["arbitration_stats"],
            "codes": _codes(stale),
            "rejected_receipts": [
                {
                    "shard_id": item["shard_id"],
                    "served_revision": item["served_revision"],
                    "current_revision": item["current_revision"],
                    "code": item["code"],
                    "covered_by": item["covered_by"],
                    "retained": bool(item["raw_receipt"]),
                }
                for item in stale.ledger["receipt_arbitrations"]
                if not item["counted"]
            ],
        },
        "end_to_end_ms_monotonic": (time.monotonic_ns() - started_ns) // 1_000_000,
        "all_checks_passed": all(
            value
            for key, value in checks.items()
            if key in {
                "baseline_equivalence",
                "edit_sequence_equivalence",
                "bucket_level_reuse_observed",
                "serial_parallel_equivalence",
                "serial_parallel_failure_order_equal",
            }
        ),
    }
    write_json(out_dir / "report.json", report)
    write_json(
        out_dir / "business-codes.json",
        {
            "baseline": _codes(baseline_incremental),
            "edits": {item["edit_id"]: item["business_codes"] for item in edit_results},
            "low_budget": _codes(low_budget),
            "hard_limit": _codes(hard_limit),
            "stale_revision": _codes(stale),
            "recovery": {issue["code"]: 1 for issue in recovered.recovery["issues"]},
        },
    )
    lines = [
        f"generated_sample={sample_dir}",
        f"baseline_equivalence={checks['baseline_equivalence']}",
        f"edit_sequence_equivalence={checks['edit_sequence_equivalence']}",
        f"bucket_level_reuse_observed={checks['bucket_level_reuse_observed']}",
        f"serial_parallel_equivalence={checks['serial_parallel_equivalence']}",
        f"serial_parallel_failure_order_equal={checks['serial_parallel_failure_order_equal']}",
        f"baseline_entries={report['baseline']['entries_count']}",
        f"stale_arbitration={json.dumps(checks['stale_arbitration'], sort_keys=True)}",
        f"low_budget_codes={json.dumps(checks['budget_pending_codes'], sort_keys=True)}",
        f"hard_limit_codes={json.dumps(checks['per_shard_abort_codes'], sort_keys=True)}",
        f"recovery_decision={checks['recovery']['decision']}",
        f"recovery_codes={json.dumps(checks['recovery']['codes'], sort_keys=True)}",
        f"end_to_end_ms_monotonic={report['end_to_end_ms_monotonic']}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
