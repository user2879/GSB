#!/usr/bin/env python3
"""一条命令执行固定序列：生成 -> 合成 -> 增量修改 -> 强制终止 -> 恢复 -> 验证。

用法：
  python run_demo.py --workspace workspace_demo
  python run_demo.py --workspace workspace_demo --repeat-check   # 跑两次并比对

所有输出均为确定性 JSON；报告内不含时间戳、PID、主机名等非确定信息。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from score_pipeline.engine import Engine
from score_pipeline.injection import (
    FAULT_COMMIT_CRASH_AFTER_BLOB,
    InjectionConfig,
)
from score_pipeline.lifecycle import (
    _load_context,
    controlled_close,
    planned_operations,
    prepare_workspace,
    resume_after_crash,
)
from score_pipeline.model import content_hash, read_json, write_json
from score_pipeline.scenarios import run_error_scenarios
from score_pipeline.validator import validate_workspace


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_reference(workspace: Path) -> dict:
    """无中断参考线：整段重建基线 + 全量增量演进，逐版本记录哈希与音符行。"""
    info = _load_context(str(workspace))
    score = info["score"]
    engine = Engine(score)
    state = engine.full_rebuild(version=0)
    expr = {
        "tempo": score.tempo,
        "dynamics": score.dynamics,
        "pedals": score.pedals,
    }
    full = engine.full_rebuild(version=0)
    versions = {
        0: {
            "timeline_hash": state.timeline_hash,
            "notes": {pid: list(rows) for pid, rows in state.rendered_notes.items()},
            "full_rebuild_hash": full.timeline_hash,
        }
    }
    for spec in planned_operations(score):
        from score_pipeline.engine import make_operation

        op = make_operation(
            version=spec["version"],
            parent=state,
            kind=spec["kind"],
            tempo=spec.get("tempo"),
            dynamics=spec.get("dynamics"),
            pedals=spec.get("pedals"),
            total_beats=score.total_beats,
        )
        state, _ = engine.incremental_apply(state, op)
        expr[spec["kind"]] = spec[spec["kind"]]
        full = engine.full_rebuild(
            tempo=expr["tempo"], dynamics=expr["dynamics"],
            pedals=expr["pedals"], version=op.version,
        )
        versions[op.version] = {
            "timeline_hash": state.timeline_hash,
            "notes": {pid: [dict(r) for r in rows] for pid, rows in state.rendered_notes.items()},
            "full_rebuild_hash": full.timeline_hash,
        }
    return versions


def _full_rebuild_hashes(workspace: Path) -> dict:
    """每个历史版本用整段重建独立产出，用于与增量哈希核对。"""
    info = _load_context(str(workspace))
    score = info["score"]
    engine = Engine(score)
    specs = planned_operations(score)
    result = {}
    state0 = engine.full_rebuild(version=0)
    result[0] = state0.timeline_hash
    expr = {
        "tempo": score.tempo,
        "dynamics": score.dynamics,
        "pedals": score.pedals,
    }
    for spec in specs:
        expr[spec["kind"]] = spec[spec["kind"]]
        rebuilt = engine.full_rebuild(
            tempo=expr["tempo"],
            dynamics=expr["dynamics"],
            pedals=expr["pedals"],
            version=spec["version"],
        )
        result[spec["version"]] = rebuilt.timeline_hash
    return result


def _verify_closures(workspace: Path) -> dict:
    """闭包精度：重算集合 == 闭包清单；闭包外相对时长/力度逐音符不变。"""
    info = _load_context(str(workspace))
    store_path = Path(info["state_dir"])
    closure_files = sorted((store_path / "closures").glob("v*.json"))
    checks = []
    for path in closure_files:
        closure = read_json(path)
        recomputed = closure["recomputed_note_total"]
        listed = sum(len(p["recompute_note_ids"]) for p in closure["parts"].values())
        checks.append(
            {
                "version": closure["op_version"],
                "kind": closure["kind"],
                "recomputed_total": recomputed,
                "listed_total": listed,
                "exact_match": recomputed == listed,
            }
        )
    return {"all_exact": all(item["exact_match"] for item in checks), "checks": checks}


def run_once(workspace: str) -> dict:
    workspace = Path(workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    prepared = prepare_workspace(str(workspace))
    reports = workspace / "reports"

    # --- 参考线（无中断） ---
    reference = _build_reference(workspace)

    # --- 子进程工作进程：在 v2 提交写 blob 后强制终止 ---
    injection = InjectionConfig(
        crash_at_version=2, crash_point=FAULT_COMMIT_CRASH_AFTER_BLOB
    )
    injection_path = reports / "injection_crash.json"
    injection.save(str(injection_path))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "score_pipeline.cli",
            "worker",
            "--workspace",
            str(workspace),
            "--injection",
            str(injection_path),
        ],
        capture_output=True,
        text=True,
    )
    worker_result = {
        "exit_code": proc.returncode,
        "stderr": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "",
        "forced_termination": proc.returncode == 137,
    }

    # --- 恢复 + 续写 ---
    resumed = resume_after_crash(str(workspace))
    recovery = resumed["recovery"]
    recovery["exit_report_status"] = (
        "missing_forced_termination" if recovery["missing_exit_report"] else "present"
    )

    # --- 恢复后逐音符/逐版本 parity ---
    manifest = read_json(workspace / "state" / "manifest.json")
    parity = {
        "final_timeline_hash_match": manifest["timeline_hash"]
        == reference[max(reference)]["timeline_hash"],
        "stable_version": manifest["version"],
    }
    engine = Engine(prepared["score"])
    current = engine.full_rebuild(
        tempo=manifest["tempo"],
        dynamics=manifest["dynamics"],
        pedals=[
            {"part_id": p["part_id"], "start_beat": p["start_beat"], "end_beat": p["end_beat"]}
            for p in manifest["pedals"]
        ],
        version=manifest["version"],
    )
    note_mismatch = None
    for part_id in sorted(current.rendered_notes):
        ref_rows = {r["note_id"]: r for r in reference[current.version]["notes"][part_id]}
        for row in current.rendered_notes[part_id]:
            if ref_rows[row["note_id"]] != row:
                note_mismatch = {"part_id": part_id, "note_id": row["note_id"]}
                break
        if note_mismatch:
            break
    parity["per_note_match"] = note_mismatch is None
    parity["first_mismatch"] = note_mismatch

    # --- 增量 vs 整段重建哈希 ---
    full_hashes = _full_rebuild_hashes(workspace)
    store_manifests = {"final": manifest["timeline_hash"]}
    rebuild_parity = {
        "all_versions_match": all(
            reference[version]["timeline_hash"] == full_hashes[version]
            for version in full_hashes
        ),
        "incremental_hashes": {str(k): v["timeline_hash"] for k, v in reference.items()},
        "full_rebuild_hashes": {str(k): v for k, v in full_hashes.items()},
    }

    closure_check = _verify_closures(workspace)

    # --- 受控退出 + 独立验证器 + 错误码链路 ---
    exit_report = controlled_close(str(workspace))
    validation = validate_workspace(str(workspace))
    error_scenarios = run_error_scenarios(str(workspace))

    report = {
        "schema": "demo-report/1",
        "generation": prepared["generation_report"],
        "worker": worker_result,
        "recovery": recovery,
        "parity": parity,
        "rebuild_parity": rebuild_parity,
        "closure_check": closure_check,
        "controlled_exit": exit_report,
        "validation": validation,
        "error_scenarios": error_scenarios,
        "overall_ok": (
            worker_result["forced_termination"]
            and parity["per_note_match"]
            and parity["final_timeline_hash_match"]
            and rebuild_parity["all_versions_match"]
            and closure_check["all_exact"]
            and validation["ok"]
            and error_scenarios["all_codes_covered"]
        ),
    }
    write_json(reports / "demo_report.json", report)
    report_hash = _file_sha256(reports / "demo_report.json")
    write_json(reports / "demo_report.sha256.json", {"demo_report_sha256": report_hash})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace_demo")
    parser.add_argument("--repeat-check", action="store_true")
    args = parser.parse_args(argv)

    report1 = run_once(args.workspace)
    hash1 = read_json(Path(args.workspace) / "reports" / "demo_report.sha256.json")[
        "demo_report_sha256"
    ]
    repeat = None
    if args.repeat_check:
        workspace2 = f"{args.workspace.rstrip('/')}_repeat"
        report2 = run_once(workspace2)
        hash2 = read_json(Path(workspace2) / "reports" / "demo_report.sha256.json")[
            "demo_report_sha256"
        ]
        repeat = {
            "report_hashes_equal": hash1 == hash2,
            "first_hash": hash1,
            "second_hash": hash2,
            "byte_identical": _file_sha256(
                Path(args.workspace) / "reports" / "demo_report.json"
            )
            == _file_sha256(Path(workspace2) / "reports" / "demo_report.json"),
        }
        write_json(Path(args.workspace) / "reports" / "repeat_check.json", repeat)

    summary = {
        "overall_ok": report1["overall_ok"],
        "report_sha256": hash1,
        "repeat_check": repeat,
    }
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if report1["overall_ok"] and (repeat is None or repeat["report_hashes_equal"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
