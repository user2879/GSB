from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .bundle import load_bundle
from .errors import Failure
from .samples import default_config
from .session import (
    InteractiveSession,
    canonical_ledger,
    field_report,
    render_final_state,
)
from .snapshot import SnapshotStore
from .validation import (
    validate_config,
    validate_dimensions,
    validate_motion_table,
    validate_noise_table,
    validate_png_bytes,
    validate_prompt_text,
    write_failure_list,
    load_yaml_file,
)


def _read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_command_acceptance(root: Path, *, stream_path: str | None = None,
                           report_dir: Path | None = None, workers: int = 2,
                           budget_ms: float | None = None) -> dict[str, Any]:
    root = Path(root)
    report_dir = Path(report_dir or root / "artifacts/command-acceptance")
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle, load_failures = load_bundle(root)
    if load_failures:
        payload = write_failure_list(load_failures, report_dir / "load_failures.json")
        return {"ok": False, "summary": {"load_failure_count": payload["count"]}}

    if stream_path:
        stream = _read_yaml(Path(stream_path))
        commands = stream["commands"]
    else:
        commands = bundle["commands"]["commands"]

    snapshot_dir = report_dir / "snapshot"
    incremental = InteractiveSession(
        bundle, snapshot_dir=snapshot_dir,
        output_dir=report_dir / "incremental-frames", workers=workers,
        budget_ms=budget_ms,
    )
    incremental.finish()
    incremental.run_stream(commands, budget_ms=budget_ms)
    final_result = incremental.finish(output_dir=report_dir / "incremental-frames")
    incremental_report = field_report(incremental.last_engine, final_result)

    oracle_engine, oracle_result, _ = render_final_state(
        bundle, commands, workers=min(3, int(bundle["config"]["max_workers"])),
        output_dir=report_dir / "full-frames",
    )
    full_report = field_report(oracle_engine, oracle_result)

    equality = {
        "pixels": incremental_report["pixels"] == full_report["pixels"],
        "denoise_residuals": incremental_report["denoise_residuals"] == full_report["denoise_residuals"],
        "tile_completion_order": incremental_report["tile_completion_order"]
        == full_report["tile_completion_order"],
        "lineage": incremental_report["lineage"] == full_report["lineage"],
        "failures": incremental_report["failures"] == full_report["failures"],
    }

    ledger_runs: list[list[dict[str, Any]]] = []
    worker_checks: list[dict[str, Any]] = []
    for check_workers in (1, 2, int(bundle["config"]["max_workers"])):
        first = InteractiveSession(bundle, workers=check_workers)
        first.finish()
        first.run_stream(commands, budget_ms=None)
        second = InteractiveSession(bundle, workers=check_workers)
        second.finish()
        second.run_stream(commands, budget_ms=None)
        first_ledger = canonical_ledger(first.command_reports)
        second_ledger = canonical_ledger(second.command_reports)
        ledger_runs.append(first_ledger)
        check_engine, check_result, _ = render_final_state(
            bundle, commands, workers=check_workers
        )
        check_report = field_report(check_engine, check_result)
        worker_checks.append({
            "workers": check_workers,
            "canonical_ledger_repeat_equal": first_ledger == second_ledger,
            "pixels_equal_cold_full": check_report["pixels"] == full_report["pixels"],
            "residuals_equal_cold_full":
                check_report["denoise_residuals"] == full_report["denoise_residuals"],
            "tile_order_equal_cold_full":
                check_report["tile_completion_order"] == full_report["tile_completion_order"],
            "max_observed_concurrency": check_engine.max_observed_concurrency,
            "worker_limit": check_engine.workers,
            "concurrency_limit_respected":
                check_engine.max_observed_concurrency <= check_engine.workers
                <= int(bundle["config"]["max_workers"]),
        })

    per_command = [
        {
            "command_id": report.get("command_id"),
            "op": report.get("op"),
            "status": report.get("status"),
            "code": report.get("code"),
            "affected_frame_range": report.get("affected_frame_range"),
            "reused": sum(1 for e in report.get("entries", []) if e["status"] == "HIT"),
            "recomputed": sum(1 for e in report.get("entries", []) if e["status"] == "RECOMPUTE"),
        }
        for report in incremental.command_reports
    ]
    ok = all(equality.values()) and all(item["canonical_ledger_repeat_equal"]
                                        and item["pixels_equal_cold_full"]
                                        for item in worker_checks)
    summary = {
        "ok": ok,
        "equality": equality,
        "command_count": len(commands),
        "per_command": per_command,
        "worker_checks": worker_checks,
        "final_summary": final_result["summary"],
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "field_comparison.json").write_text(
        json.dumps({"incremental": incremental_report, "full": full_report,
                    "equality": equality}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "canonical_ledger.json").write_text(
        json.dumps(ledger_runs[0] if ledger_runs else [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": ok, "summary": summary, "report_dir": str(report_dir)}


def run_negative_validation(root: Path, report_dir: Path) -> dict[str, Any]:
    root = Path(root)
    cases_doc = _read_yaml(root / "samples/negative/cases.yaml")
    config = default_config(root)
    all_failures: list[Failure] = []
    case_results: list[dict[str, Any]] = []

    for case in cases_doc["cases"]:
        path = root / case["path"]
        kind = case["kind"]
        failures: list[Failure] = []
        if kind == "prompt":
            doc = _read_yaml(path)
            terms = []
            vocab = _read_yaml(root / "samples/controlled_vocab.yaml")
            terms = vocab["entities"] + vocab["attributes"]
            for frame, text in enumerate(doc.get("frame_prompts", [])):
                failures.extend(validate_prompt_text(text, terms, path=str(path), frame=frame))
        elif kind == "config":
            doc = _read_yaml(path)
            failures = validate_config(doc, str(path))
        elif kind == "bundle":
            doc = _read_yaml(path)
            vocab = _read_yaml(root / "samples/controlled_vocab.yaml")
            terms = vocab["entities"] + vocab["attributes"]
            failures = validate_config(doc, str(path))
            failures.extend(validate_dimensions(doc, terms))
        elif kind == "png":
            failures = validate_png_bytes(path.read_bytes(), config, str(path))
        elif kind == "motion":
            doc = _read_yaml(path)
            failures = validate_motion_table(doc, config, str(path))
        elif kind == "noise":
            data = json.loads(path.read_text(encoding="utf-8"))
            failures = validate_noise_table(data, config, str(path))

        all_failures.extend(failures)
        expected_codes = case.get("expected_codes") or [case.get("expected_code")]
        found_codes = [failure.code for failure in failures]
        matched = all(code in found_codes for code in expected_codes if code)
        case_results.append({
            "id": case["id"],
            "path": case["path"],
            "expected_codes": expected_codes,
            "found_codes": found_codes,
            "matched": matched,
        })

    payload = write_failure_list(
        all_failures, report_dir / "negative_failure_list.json",
        extra={"suite": "negative-validation", "cases": case_results},
    )
    ok = all(item["matched"] for item in case_results)
    return {"ok": ok, "case_results": case_results, "failure_count": payload["count"]}


def run_injection_acceptance(root: Path, report_dir: Path) -> dict[str, Any]:
    root = Path(root)
    report_dir.mkdir(parents=True, exist_ok=True)
    bundle, failures = load_bundle(root)
    if failures:
        return {"ok": False, "failures": [f.to_dict() for f in failures]}
    injections = bundle["injections"]["injections"]
    results: list[dict[str, Any]] = []
    ok = True

    for injection in injections:
        session = InteractiveSession(bundle, injections=[injection], workers=2)
        result = session.finish()
        expected_code = injection["expected_code"]
        matching = [
            f for f in result["failures"]
            if f.get("injection_id") == injection["id"] and f.get("code") == expected_code
            and f.get("frame") == injection["target_frame"]
            and f.get("stage") == injection["stage"]
        ]
        other_failures = [
            f for f in result["failures"]
            if f.get("frame") != injection["target_frame"]
        ]
        case_ok = bool(matching) and not other_failures and result["completed_frames"] == [
            f for f in range(4) if f != injection["target_frame"]
        ]
        ok = ok and case_ok
        results.append({
            "injection_id": injection["id"],
            "type": injection["type"],
            "target_frame": injection["target_frame"],
            "stage": injection["stage"],
            "expected_code": expected_code,
            "matched": bool(matching),
            "isolated": not other_failures,
            "completed_frames": result["completed_frames"],
            "failure": matching[0] if matching else None,
            "ok": case_ok,
        })

    combined = InteractiveSession(bundle, injections=injections, workers=4)
    combined_result = combined.finish()
    combined_codes = [f["code"] for f in combined_result["failures"]]
    expected_order = [item["expected_code"] for item in injections]
    combined_ok = combined_codes == expected_order
    ok = ok and combined_ok

    payload = {
        "ok": ok,
        "single_injections": results,
        "combined": {
            "failure_codes_in_order": combined_codes,
            "expected_order": expected_order,
            "order_matches": combined_ok,
        },
    }
    (report_dir / "injection_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_failure_list(
        combined_result["failures"], report_dir / "injection_failure_list.json",
        extra={"injection_ids": [i["id"] for i in injections]},
    )
    return payload


def run_preview_acceptance(root: Path, report_dir: Path) -> dict[str, Any]:
    bundle, failures = load_bundle(root)
    if failures:
        return {"ok": False, "failures": [f.to_dict() for f in failures]}
    session = InteractiveSession(bundle, workers=2)
    baseline = session.finish()
    old_hash = baseline["summary"]["config_hash"]
    old_condition = baseline["summary"]["condition_fingerprint"]

    stale_prompt = session.handle(
        {"id": "PREVIEW-1", "op": "replace_prompt", "frame": 0,
         "text": "green sphere dark rough"},
        budget_ms=0.0,
    )
    stale_weights = session.handle(
        {"id": "PREVIEW-2", "op": "set_weights",
         "weights": {"text": "0.80", "image": "0.10", "motion": "0.10"}},
        budget_ms=0.0,
    )
    final = session.finish()
    result = {
        "stale_prompt_status": stale_prompt["status"],
        "stale_prompt_code": stale_prompt["code"],
        "stale_prompt_marked_old_condition":
            stale_prompt["summary"].get("condition_fingerprint") == old_condition,
        "stale_weights_code": stale_weights["code"],
        "stale_weights_old_config_hash":
            stale_weights["summary"].get("config_hash") == old_hash,
        "fresh_final_status": final["status"],
        "fresh_final_code": final["summary"].get("fresh"),
        "ok": (
            stale_prompt["code"] == "E_STALE_PREVIEW"
            and stale_weights["code"] == "E_STALE_PREVIEW"
            and stale_weights["summary"].get("config_hash") == old_hash
            and final["status"] == "complete"
        ),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "preview_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _wait_for_committed_frame(snapshot_dir: Path, timeout_s: float = 10.0) -> dict[str, Any]:
    store = SnapshotStore(snapshot_dir)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = store.load_current()
        if payload and payload.get("completed_frames"):
            return payload
        time.sleep(0.02)
    raise TimeoutError("snapshot did not commit a completed frame before timeout")


def run_sigkill_recovery(root: Path, report_dir: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    project_root = Path(__file__).resolve().parent.parent
    report_dir = Path(report_dir)
    if report_dir.exists():
        shutil.rmtree(report_dir)
    snapshot_dir = report_dir / "snapshot"
    output_dir = report_dir / "recovered-frames"
    report_json = report_dir / "killed_render_report.json"
    pid_file = report_dir / "render.pid"
    report_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable, "-m", "condrender.cli", "render",
        "--root", str(root),
        "--snapshot-dir", str(snapshot_dir),
        "--output-dir", str(output_dir),
        "--report", str(report_json),
        "--workers", "4",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(project_root) + os.pathsep + str(root) + os.pathsep + env.get("PYTHONPATH", "")
    )
    killed_cmd = base_cmd + ["--task-delay-ms", "35", "--pid-file", str(pid_file)]
    process = subprocess.Popen(
        killed_cmd, cwd=str(project_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        committed = _wait_for_committed_frame(snapshot_dir)
        committed_before_kill = {
            "global_version": committed["global_version"],
            "completed_frames": committed["completed_frames"],
            "config_hash": committed["config_hash"],
        }
        time.sleep(0.05)
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGKILL)
        return_code = process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        stdout, stderr = process.communicate(timeout=5)

    after_kill = SnapshotStore(snapshot_dir).load_current()
    staging_left = list(snapshot_dir.glob(".tmp-*"))
    resume = subprocess.run(
        base_cmd + ["--report", str(report_dir / "resume_report.json")],
        cwd=str(project_root), env=env, text=True, capture_output=True, timeout=30,
    )
    resume_report = json.loads((report_dir / "resume_report.json").read_text(encoding="utf-8"))

    bundle, failures = load_bundle(root)
    oracle_engine, oracle_result, _ = render_final_state(
        bundle, bundle["commands"]["commands"][:0], workers=1,
        output_dir=report_dir / "cold-frames",
    )
    recovered_engine = RenderEngine = None
    recovered_pngs = sorted((output_dir / "frames").glob("frame_*.png"))
    cold_pngs = sorted((report_dir / "cold-frames" / "frames").glob("frame_*.png"))
    png_bytes_equal = (
        len(recovered_pngs) == len(cold_pngs) == 4
        and all(a.read_bytes() == b.read_bytes() for a, b in zip(recovered_pngs, cold_pngs))
    )
    result = {
        "sigkill_returncode": return_code,
        "sigkill_is_minus_9": return_code == -9,
        "committed_before_kill": committed_before_kill,
        "committed_snapshot_still_readable": after_kill is not None,
        "uncommitted_staging_directories_left_for_reaper": len(staging_left),
        "resume_returncode": resume.returncode,
        "resume_stdout_tail": resume.stdout[-500:],
        "resume_stderr_tail": resume.stderr[-1000:],
        "restore_decision": resume_report.get("restore_decision"),
        "resume_summary": {
            "completed_frames": resume_report["summary"]["completed_frames"],
            "reused_entries": resume_report["summary"]["reused_entries"],
            "recomputed_entries": resume_report["summary"]["recomputed_entries"],
        },
        "png_bytes_equal_cold_full": png_bytes_equal,
        "ok": (
            return_code == -9 and resume.returncode == 0 and png_bytes_equal
            and after_kill is not None
            and resume_report.get("restore_decision", {}).get("decision") == "resume_pending"
        ),
    }
    (report_dir / "kill_recovery_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def run_full_verification(root: Path, artifacts_dir: Path) -> dict[str, Any]:
    root = Path(root)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    from .samples import generate_samples
    generate_samples(root)

    command_acceptance = run_command_acceptance(
        root, report_dir=artifacts_dir / "commands", workers=4
    )
    negative = run_negative_validation(root, artifacts_dir / "validation")
    injections = run_injection_acceptance(root, artifacts_dir / "injections")
    preview = run_preview_acceptance(root, artifacts_dir / "preview")
    recovery = run_sigkill_recovery(root, artifacts_dir / "sigkill")
    tile_recovery = run_tile_cursor_kill_recovery(root, artifacts_dir / "tile-sigkill")

    checks = {
        "command_stream_incremental_equals_full": command_acceptance["ok"],
        "negative_validation_codes": negative["ok"],
        "injection_isolation": injections["ok"],
        "stale_preview": preview["ok"],
        "sigkill_snapshot_recovery": recovery["ok"],
        "tile_cursor_sigkill_recovery": tile_recovery["ok"],
    }
    ok = all(checks.values())
    report = {
        "ok": ok,
        "checks": checks,
        "summary": {
            "ok": ok,
            "checks": checks,
            "command_summary": command_acceptance.get("summary"),
            "negative": {"ok": negative["ok"], "case_count": len(negative["case_results"])},
            "injections": {"ok": injections["ok"]},
            "preview": preview,
            "recovery": recovery,
            "tile_recovery": tile_recovery,
        },
    }
    (artifacts_dir / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    markdown = _render_markdown(report)
    (artifacts_dir / "VERIFICATION_REPORT.md").write_text(markdown, encoding="utf-8")
    return report


def run_tile_cursor_kill_recovery(root: Path, report_dir: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    project_root = Path(__file__).resolve().parent.parent
    report_dir = Path(report_dir)
    if report_dir.exists():
        shutil.rmtree(report_dir)
    snapshot_dir = report_dir / "snapshot"
    output_dir = report_dir / "recovered-frames"
    pid_file = report_dir / "render.pid"
    report_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable, "-m", "condrender.cli", "render",
        "--root", str(root),
        "--snapshot-dir", str(snapshot_dir),
        "--output-dir", str(output_dir),
        "--report", str(report_dir / "killed_render_report.json"),
        "--workers", "4",
        "--tile-rows", "8",
        "--product-checkpoints",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root) + os.pathsep + str(root) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        base_cmd + ["--task-delay-ms", "70", "--pid-file", str(pid_file)],
        cwd=str(project_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    partial_cursor: dict[str, Any] | None = None
    try:
        store = SnapshotStore(snapshot_dir)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and partial_cursor is None:
            payload = store.load_current()
            if payload:
                for frame, cursor in payload.get("tile_cursors", {}).items():
                    count = int(cursor.get("completed_count", 0))
                    if 0 < count < 8:
                        partial_cursor = {"frame": int(frame), "cursor": cursor,
                                          "global_version": payload.get("global_version")}
                        break
            time.sleep(0.005)
        if partial_cursor is None:
            raise TimeoutError("no partial tile-row cursor was committed before timeout")

        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGKILL)
        return_code = process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        stdout, stderr = process.communicate(timeout=5)

    after_kill = SnapshotStore(snapshot_dir).load_current()
    resume = subprocess.run(
        base_cmd + ["--report", str(report_dir / "resume_report.json")],
        cwd=str(project_root), env=env, text=True, capture_output=True, timeout=30,
    )
    resume_report = json.loads((report_dir / "resume_report.json").read_text(encoding="utf-8"))

    bundle, failures = load_bundle(root)
    if failures:
        return {"ok": False, "failures": [f.to_dict() for f in failures]}
    render_final_state(
        bundle,
        [{"id": "COLD-TILE8", "op": "set_tile_rows", "tile_rows": 8}],
        workers=1, output_dir=report_dir / "cold-frames",
    )
    recovered_pngs = sorted((output_dir / "frames").glob("frame_*.png"))
    cold_pngs = sorted((report_dir / "cold-frames" / "frames").glob("frame_*.png"))
    png_bytes_equal = (
        len(recovered_pngs) == len(cold_pngs) == 4
        and all(a.read_bytes() == b.read_bytes() for a, b in zip(recovered_pngs, cold_pngs))
    )
    partial_count = int(partial_cursor["cursor"]["completed_count"])
    result = {
        "sigkill_returncode": return_code,
        "sigkill_is_minus_9": return_code == -9,
        "partial_cursor_before_kill": partial_cursor,
        "committed_snapshot_still_readable": after_kill is not None,
        "resume_returncode": resume.returncode,
        "resume_stderr_tail": resume.stderr[-1000:],
        "restore_decision": resume_report.get("restore_decision"),
        "resume_summary": {
            "reused_entries": resume_report["summary"]["reused_entries"],
            "recomputed_entries": resume_report["summary"]["recomputed_entries"],
        },
        "partial_tile_rows_reused_on_resume":
            resume_report["summary"]["reused_entries"] >= partial_count,
        "png_bytes_equal_cold_tile8": png_bytes_equal,
        "ok": (
            return_code == -9
            and resume.returncode == 0
            and png_bytes_equal
            and after_kill is not None
            and resume_report.get("restore_decision", {}).get("decision") == "resume_pending"
            and resume_report["summary"]["reused_entries"] >= partial_count
        ),
    }
    (report_dir / "tile_cursor_recovery_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 实际验证记录",
        "",
        f"总体结果：{'通过' if report['ok'] else '未通过'}",
        "",
        "| 验收项 | 结果 |",
        "|---|---|",
    ]
    for name, ok in report["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} |")
    recovery = summary["recovery"]
    tile_recovery = summary.get("tile_recovery", {})
    lines.extend([
        "",
        "## SIGKILL 恢复",
        "",
        f"- 被终止进程返回码：`{recovery['sigkill_returncode']}`（`-9` 表示 SIGKILL）",
        f"- 终止前已提交：帧 `{recovery['committed_before_kill']['completed_frames']}`，"
        f"全局版本 `{recovery['committed_before_kill']['global_version']}`",
        f"- 恢复决策：`{recovery['restore_decision']['decision']}`",
        f"- 恢复渲染 HIT/RECOMPUTE：`{recovery['resume_summary']['reused_entries']}` / "
        f"`{recovery['resume_summary']['recomputed_entries']}`",
        f"- 恢复后 PNG 与冷全量逐字节一致：`{recovery['png_bytes_equal_cold_full']}`",
        f"- 半提交 tile 游标 SIGKILL：`{tile_recovery.get('sigkill_returncode')}`，"
        f"终止前 `{tile_recovery.get('partial_cursor_before_kill', {}).get('cursor')}`，"
        f"恢复 HIT/RECOMPUTE=`{tile_recovery.get('resume_summary', {}).get('reused_entries')}`/"
        f"`{tile_recovery.get('resume_summary', {}).get('recomputed_entries')}`，"
        f"PNG 一致=`{tile_recovery.get('png_bytes_equal_cold_tile8')}`",
        "",
        "## 命令流",
        "",
    ])
    for item in summary["command_summary"]["per_command"]:
        lines.append(
            f"- `{item['command_id']}` {item['op']} [{item['affected_frame_range'][0]},"
            f"{item['affected_frame_range'][1]}] HIT={item['reused']} "
            f"RECOMPUTE={item['recomputed']} status={item['status']} {item['code']}"
        )
    lines.append("")
    return "\n".join(lines)
