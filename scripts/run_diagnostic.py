#!/usr/bin/env python3
"""端到端诊断入口：产出可复核的诊断报告。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxpdiag.convert_service import full_roundtrip
from fxpdiag.edits import run_command_sequence
from fxpdiag.errors import FailureList
from fxpdiag.fingerprints import config_hash
from fxpdiag.loader import (
    apply_injection,
    build_weight_states,
    load_commands,
    load_config,
    load_injections,
    load_package,
)
from fxpdiag.pipeline import PipelineRunner, PipelineState
from fxpdiag.samplegen import write_samples
from fxpdiag.snapshots import can_resume, commit_snapshot, load_manifest


def _new_state(package, config, v1_map, v2_map, version="v1"):
    state = PipelineState(copy.deepcopy(package), copy.deepcopy(config), version)
    state.weights.v1_rows = copy.deepcopy(v1_map)
    state.weights.v2_rows = copy.deepcopy(v2_map)
    return state


def _slim(artifacts: dict) -> dict:
    """去掉大体量中间数组，保留可逐字段比对的稳定结构。"""

    return {
        "active_version": artifacts["active_version"],
        "batches": artifacts["batches"],
        "ledger": artifacts["ledger"],
        "ledger_summary": artifacts["ledger_summary"],
        "failures": artifacts["failures"],
        "conversion_records": artifacts["conversion_records"],
        "row_signatures": [
            {
                "layer": row["layer"],
                "row_index": row["row_index"],
                "version": row["version"],
                "softmax_int": row["softmax"]["int"],
                "softmax_argmax": row["softmax"]["argmax"],
                "ln_int": row["layernorm"]["int"],
                "ln_stats": row["layernorm"]["stats"],
                "degenerate": row["layernorm"]["degenerate"],
                "diagnosis": row["diagnosis"],
                "cross_stage_error": row["cross_layer"]["stage_error"],
            }
            for row in artifacts["rows"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="定点化诊断管线")
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--snapshot-dir", default="reports/snapshot")
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)
    paths = write_samples(args.samples_dir)
    failures = FailureList()
    package = load_package(paths["package"], failures)
    config = load_config(paths["config"])
    commands = load_commands(paths["commands"], failures)
    v1_map, v2_map = build_weight_states(paths["v1"], paths["v2"], failures)
    injections = load_injections(paths["injections"])

    report: dict = {"sample_paths": paths, "load_failures": failures.to_dicts()}

    # 1) 双版本互转与往返损耗
    v2_rows = list(v2_map.values())
    report["roundtrip"] = full_roundtrip(v2_rows, os.path.join(args.reports_dir, "roundtrip"))

    # 2) 增量运行（串行）与全量重建（空白）一致性
    inc_state = _new_state(package, config, v1_map, v2_map, "v1")
    incremental = run_command_sequence(
        inc_state, commands, PipelineRunner, parallel=False
    )
    full_slim = []
    for snapshot in incremental:
        # 每一步都从空白全量重建
        blank = _new_state(package, config, v1_map, v2_map, "v1")
        blank_snapshots = run_command_sequence(
            blank, commands[: snapshot["seq"]], PipelineRunner, parallel=False
        )
        full_slim.append(_slim(blank_snapshots[-1]))
    incremental_slim = [_slim(item) for item in incremental]
    report["incremental_equals_full"] = incremental_slim == full_slim

    # 3) 并发与串行一致性（最终状态）
    serial_state = _new_state(package, config, v1_map, v2_map, "v1")
    serial_final = run_command_sequence(
        serial_state, commands, PipelineRunner, parallel=False
    )[-1]
    parallel_state = _new_state(package, config, v1_map, v2_map, "v1")
    parallel_final = run_command_sequence(
        parallel_state, commands, PipelineRunner, parallel=True
    )[-1]
    report["parallel_equals_serial"] = _slim(parallel_final) == _slim(serial_final)

    # 4) 快照整体替换 + 续跑判定
    manifest = {
        "config_version": config.config_version,
        "config": config.__dict__,
        "active_version": inc_state.active_version,
    }
    commit_snapshot(
        args.snapshot_dir,
        {"final": _slim(incremental[-1])},
        manifest,
    )
    report["snapshot"] = {
        "resume_same_config": can_resume(
            args.snapshot_dir, config.config_version, config.__dict__
        ),
        "resume_wrong_version": can_resume(
            args.snapshot_dir, config.config_version + 1, config.__dict__
        ),
        "manifest": load_manifest(args.snapshot_dir),
        "config_hash": config_hash(config.__dict__),
    }

    # 5) 故障注入隔离
    injection_failures = FailureList()
    injection_reports = [
        apply_injection(package, v1_map, v2_map, item, injection_failures)
        for item in injections
    ]
    report["injections"] = {
        "items": injection_reports,
        "all_detected": all(item["detected"] for item in injection_reports),
        "failure_codes": injection_failures.codes(),
    }

    # 6) 诊断明细（最终增量快照）
    report["final_diagnostic"] = _slim(incremental[-1])
    report["budget_pending_count"] = sum(
        1 for item in incremental[-1]["failures"]
        if item["code"] == "E_OP_BUDGET_PENDING"
    )

    report_path = os.path.join(args.reports_dir, "diagnostic_report.json")
    with open(report_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"report written: {report_path}")
    print(json.dumps({
        "incremental_equals_full": report["incremental_equals_full"],
        "parallel_equals_serial": report["parallel_equals_serial"],
        "all_injections_detected": report["injections"]["all_detected"],
        "budget_pending_count": report["budget_pending_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
