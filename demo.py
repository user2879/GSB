#!/usr/bin/env python3
"""Deterministic three-scenario demonstration plus three fault injections."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from emergency_drill.errors import BusinessError
from emergency_drill.workbench import Workbench


ROOT = Path(__file__).resolve().parent
FAULTS = ("plan-corrupt", "cycle", "snapshot-digest")


def _status_tuple(summary: dict[str, Any]) -> list[list[Any]]:
    return [
        [stage["seq"], stage["status"], stage["skip_reason"]]
        for stage in summary["stages"]
    ]


def _stage_status(summary: dict[str, Any]) -> str:
    return "|".join(f"{stage['seq']}:{stage['status']}" for stage in summary["stages"])


def _stage_status_from_tuple(summary: dict[str, Any]) -> str:
    return "|".join(f"{seq}:{status}" for seq, status, _ in summary["stage_statuses"])


def _run_fault_checks(
    workbench: Workbench, selected_faults: list[str], normal_snapshot: Path | None
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if "plan-corrupt" in selected_faults:
        try:
            workbench.load_plan_file(ROOT / "samples" / "plan_corrupt.json")
            results.append({"fault": "plan-corrupt", "reported": False})
        except BusinessError as exc:
            results.append(
                {
                    "fault": "plan-corrupt",
                    "reported": True,
                    "code": exc.code,
                    "stage_seq": exc.stage_seq,
                    "field_path": exc.field_path,
                }
            )
    if "cycle" in selected_faults:
        try:
            workbench.load_plan_file(ROOT / "samples" / "plan_cycle.json")
            results.append({"fault": "cycle", "reported": False})
        except BusinessError as exc:
            results.append(
                {
                    "fault": "cycle",
                    "reported": True,
                    "code": exc.code,
                    "stage_seq": exc.stage_seq,
                    "field_path": exc.field_path,
                }
            )
    if "snapshot-digest" in selected_faults and normal_snapshot is not None:
        original = json.loads(normal_snapshot.read_text(encoding="utf-8"))
        tampered = dict(original)
        phases = [dict(item) for item in original["phase_states"]]
        phases[0]["status"] = "failed"
        tampered["phase_states"] = phases
        tampered_path = workbench.snapshots_dir / "tampered-content-digest.json"
        tampered_path.write_text(
            json.dumps(tampered, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            workbench.load_snapshot(tampered_path)
            results.append({"fault": "snapshot-digest", "reported": False})
        except BusinessError as exc:
            results.append(
                {
                    "fault": "snapshot-digest",
                    "reported": True,
                    "code": exc.code,
                    "stage_seq": exc.stage_seq,
                    "field_path": exc.field_path,
                }
            )
    return results


def _scenario_normal(workbench: Workbench) -> tuple[dict[str, Any], Path]:
    session = "scenario-normal"
    workbench.create_session(session, ROOT / "samples" / "plan_valid.json")
    summary = workbench.run_session(session)
    snapshot = workbench.save_snapshot(session)
    restore = workbench.restore_snapshot(session, snapshot["file_path"])
    replay = workbench.replay_snapshot(session, snapshot["file_path"])
    result = {
        "name": "normal",
        "status": summary["status"],
        "logical_clock_ms": summary["logical_clock_ms"],
        "stage_statuses": _status_tuple(summary),
        "snapshot_id": snapshot["snapshot_id"],
        "restore_whole_replacement": restore["whole_replacement"],
        "restore_generation": restore["generation"],
        "replay_reproducible": replay.reproducible,
        "replay_compared_stages": replay.compared_stage_count,
    }
    return result, Path(snapshot["file_path"])


def _scenario_failure(workbench: Workbench) -> dict[str, Any]:
    session = "scenario-failure"
    workbench.create_session(session, ROOT / "samples" / "plan_valid.json")
    workbench.inject_event(
        session,
        {
            "event_id": "inject-switchover-failure-stage-3",
            "type": "switchover_failed",
            "offset_ms": 6000,
            "stage_seq": 3,
        },
    )
    summary = workbench.run_session(session)
    snapshot = workbench.save_snapshot(session)
    replay = workbench.replay_snapshot(session, snapshot["file_path"])
    return {
        "name": "failure-injection",
        "status": summary["status"],
        "logical_clock_ms": summary["logical_clock_ms"],
        "stage_statuses": _status_tuple(summary),
        "stage_status_line": _stage_status(summary),
        "snapshot_id": snapshot["snapshot_id"],
        "replay_reproducible": replay.reproducible,
    }


def _scenario_crash(workbench: Workbench, data_dir: Path) -> dict[str, Any]:
    session = "scenario-crash"
    command = [
        sys.executable,
        "-m",
        "emergency_drill.cli",
        "run",
        "--db",
        str(workbench.db_path),
        "--snapshots-dir",
        str(workbench.snapshots_dir),
        "--log",
        str(workbench.logger.path),
        "--session",
        session,
        "--plan",
        str(ROOT / "samples" / "plan_valid.json"),
        "--crash-after-stage",
        "2",
        "--hard-crash",
    ]
    child = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    recovery = workbench.recover(session).to_dict()
    resumed = workbench.resume(session)
    snapshot = workbench.save_snapshot(session)
    restore = workbench.restore_snapshot(session, snapshot["file_path"])
    replay = workbench.replay_snapshot(session, snapshot["file_path"])
    return {
        "name": "forced-termination-recovery",
        "child_exit_code": child.returncode,
        "child_hard_terminated": child.returncode == 137,
        "recovery": recovery,
        "resumed_status": resumed["run"]["status"],
        "resumed_stage_status_line": _stage_status(resumed["run"]),
        "snapshot_id": snapshot["snapshot_id"],
        "restore_whole_replacement": restore["whole_replacement"],
        "replay_reproducible": replay.reproducible,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="演练工作台确定性演示")
    parser.add_argument("--data-dir", default="artifacts/demo-data")
    parser.add_argument(
        "--fault",
        action="append",
        choices=list(FAULTS),
        help="可重复指定；默认注入全部三类异常",
    )
    parser.add_argument("--skip-faults", action="store_true")
    parser.add_argument("--keep-data", action="store_true")
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir)
    if data_dir.exists() and not args.keep_data:
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    selected_faults = [] if args.skip_faults else (args.fault or list(FAULTS))

    workbench = Workbench(
        data_dir / "session.db",
        data_dir / "snapshots",
        data_dir / "logs" / "drill.jsonl",
    )
    normal_result, normal_snapshot = _scenario_normal(workbench)
    faults = _run_fault_checks(workbench, selected_faults, normal_snapshot)
    failure_result = _scenario_failure(workbench)
    crash_result = _scenario_crash(workbench, data_dir)

    fault_ok = all(
        item["reported"]
        and item.get("code")
        == {
            "plan-corrupt": "E_PLAN",
            "cycle": "E_CYCLE",
            "snapshot-digest": "E_SNAPSHOT",
        }[item["fault"]]
        for item in faults
    )
    normal_ok = (
        normal_result["status"] == "completed"
        and normal_result["restore_whole_replacement"]
        and normal_result["replay_reproducible"]
    )
    failure_ok = (
        failure_result["status"] == "completed"
        and failure_result["stage_status_line"]
        == "1:succeeded|2:succeeded|3:failed|4:skipped|5:skipped"
        and failure_result["replay_reproducible"]
    )
    crash_ok = (
        crash_result["child_hard_terminated"]
        and crash_result["recovery"]["unknown_stage_seqs"] == [3]
        and crash_result["recovery"]["exit_report_available"] is False
        and crash_result["resumed_status"] == "completed"
        and crash_result["replay_reproducible"]
    )
    summary = {
        "faults": faults,
        "scenarios": [normal_result, failure_result, crash_result],
        "checks": {
            "faults_reported_with_expected_codes": fault_ok,
            "normal_completed_restored_and_replayed": normal_ok,
            "failure_injection_skipped_downstream_and_replayed": failure_ok,
            "forced_termination_unknown_then_recovered_and_replayed": crash_ok,
        },
        "all_checks_passed": fault_ok and normal_ok and failure_ok and crash_ok,
        "data_dir": str(data_dir.resolve()),
    }
    _write_summary(data_dir / "summary.json", summary)

    print("=== 异常注入 ===")
    for item in faults:
        if item["reported"]:
            print(
                f"{item['code']} stage={item['stage_seq']} path={item['field_path']} "
                f"fault={item['fault']}"
            )
        else:
            print(f"NOT_REPORTED fault={item['fault']}")
    print("=== 场景 1 正常演练 ===")
    print(
        f"status={normal_result['status']} stages={_stage_status_from_tuple(normal_result)} "
        f"restore_whole={normal_result['restore_whole_replacement']} "
        f"replay={normal_result['replay_reproducible']}"
    )
    print("=== 场景 2 失败事件注入 ===")
    print(
        f"status={failure_result['status']} stages={failure_result['stage_status_line']} "
        f"replay={failure_result['replay_reproducible']}"
    )
    print("=== 场景 3 强制终止后恢复 ===")
    print(
        f"child_exit={crash_result['child_exit_code']} "
        f"unknown={crash_result['recovery']['unknown_stage_seqs']} "
        f"tail_action={crash_result['recovery']['tail_action']}"
    )
    print(
        f"resumed={crash_result['resumed_status']} stages={crash_result['resumed_stage_status_line']} "
        f"replay={crash_result['replay_reproducible']}"
    )
    print(
        f"ALL_CHECKS_PASSED={str(summary['all_checks_passed']).lower()} "
        f"summary={data_dir / 'summary.json'}"
    )
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
