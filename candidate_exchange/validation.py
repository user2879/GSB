from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .edits import apply_operation, copy_dataset
from .engine import PipelineOptions, run_pipeline
from .generator import generate_dataset
from .injection import apply_injections
from .snapshot import SnapshotStore
from .util import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    remove_tree,
)


def _state(report: dict[str, Any]) -> dict[str, Any]:
    conversion = copy.deepcopy(report.get("conversion"))
    if isinstance(conversion, dict):
        conversion.pop("outputs", None)
    tracks = []
    for track in report.get("tracks", []):
        normalized = dict(track)
        normalized.pop("reused", None)
        tracks.append(normalized)
    return {
        "active_version": report.get("active_version"),
        "frames": report.get("frames"),
        "tracks": tracks,
        "failures": report.get("failures"),
        "conversion": conversion,
        "injection_hits": report.get("injections", {}).get("hits", []),
    }


def _ledger_projection(report: dict[str, Any]) -> dict[str, Any]:
    invalidation = report.get("invalidation", {})
    frames = sorted(
        {
            (
                item["series_name"],
                -1 if item.get("frame_index") is None else item["frame_index"],
                item.get("cache_key"),
            )
            for item in invalidation.get("frame_ledger", [])
        }
    )
    tracks = sorted(
        {
            (item["track_id"], item.get("cache_key"))
            for item in invalidation.get("track_ledger", [])
        }
    )
    return {"frame_cache_projection": frames, "track_cache_projection": tracks}


def _check(
    checks: dict[str, Any], name: str, passed: bool, details: Any | None = None
) -> None:
    checks[name] = {"passed": bool(passed), "details": details}


def _diff_state(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    differing = [key for key in expected if expected[key] != actual.get(key)]
    return {"equal": not differing, "differing_sections": differing}


def _run(
    data_dir: Path,
    output_dir: Path,
    snapshot_root: Path,
    *,
    workers: int,
    interval: int,
    active_version: str | None = None,
    force_full: bool = False,
    injections_path: Path | None = None,
) -> dict[str, Any]:
    return run_pipeline(
        PipelineOptions(
            data_dir=data_dir,
            output_dir=output_dir,
            snapshot_root=snapshot_root,
            active_version=active_version,
            workers=workers,
            snapshot_commit_interval=interval,
            force_full=force_full,
            injections_path=injections_path,
        )
    )


def _summary_markdown(report: dict[str, Any]) -> str:
    checks = report["checks"]
    failed = [name for name, value in checks.items() if not value["passed"]]
    lines = [
        "# 候选框交换与跨帧匹配验证报告",
        "",
        f"- 数据集种子：`{report['dataset']['seed']}`",
        f"- 序列数：`{report['dataset']['series_count']}`",
        f"- 帧数：`{report['dataset']['frame_count']}`",
        f"- 候选框数：`{report['dataset']['candidate_count']}`",
        f"- 检查总数：`{report['summary']['total_checks']}`",
        f"- 通过：`{report['summary']['passed_checks']}`",
        f"- 失败：`{report['summary']['failed_checks']}`",
        f"- 总状态：`{'PASS' if report['summary']['all_checks_passed'] else 'FAIL'}`",
        "",
        "## 关键结果",
        "",
        f"- 单线程 / 多线程状态一致：`{checks['concurrency_state_equivalence']['passed']}`",
        f"- 增量 / 空白全量最终状态一致：`{checks['incremental_state_equivalence']['passed']}`",
        f"- 增量缓存键账本投影一致：`{checks['incremental_ledger_projection']['passed']}`",
        f"- v2→v1 坐标偏差直方图桶数：`{report['results']['conversion']['v2_to_v1_histogram_buckets']}`",
        f"- v2→v1 最大坐标偏差：`{report['results']['conversion']['v2_to_v1_max_coordinate_deviation']}`",
        f"- v2→v1 最大得分偏差：`{report['results']['conversion']['v2_to_v1_max_score_deviation']}`",
        f"- 类别映射完整还原：`{report['results']['conversion']['class_mapping_complete']}`",
        f"- 注入项命中：`{report['results']['injection']['hit_count']}/{report['results']['injection']['configured_count']}`",
        f"- 强制终止后恢复点：`{report['results']['snapshot_recovery']['recovery_point_version']}`",
        f"- 恢复前存在最终退出报告：`{report['results']['snapshot_recovery']['report_existed_after_termination']}`",
        "",
        "## 编辑序列",
        "",
        "| 操作 | 帧复用 | 帧重算 | 轨迹复用 | 轨迹重算 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for step in report["results"]["incremental"]["steps"]:
        lines.append(
            f"| {step['id']} {step['type']} | {step['summary']['reused_frame_count']} | "
            f"{step['summary']['recomputed_frame_count']} | {step['summary']['reused_track_count']} | "
            f"{step['summary']['recomputed_track_count']} |"
        )
    lines.extend(["", "## 检查明细", ""])
    for name, value in checks.items():
        lines.append(f"- `{'PASS' if value['passed'] else 'FAIL'}` `{name}`")
    if failed:
        lines.extend(["", "失败项：`" + "`, `".join(failed) + "`"])
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 进程退出码只表达运行状态：`0` 成功、`2` 配置/IO 错误、`3` 未预期异常、`75` 模拟强制终止。",
            "- `E_xxx` 业务码只写入 JSON 报告或 stderr，不参与进程退出码。",
            "- 全量重建与增量运行比较的是最终帧、轨迹、偏差和失败清单；`reused` 操作标记按定义不同，因此以最终 cache-key 账本投影比较可审计依赖关系。",
            "- 强制终止测试通过独立 Python 进程执行，终止前不会生成最终验证报告。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_validation(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 20261122,
    keep_work: bool = False,
) -> dict[str, Any]:
    data = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (data / "candidates_v1.txt").exists():
        generate_dataset(data, seed=seed, force=True)
    work = output / "validation_work"
    remove_tree(work)
    work.mkdir(parents=True, exist_ok=True)
    checks: dict[str, Any] = {}
    project_root = Path(__file__).resolve().parents[1]
    dataset_summary = read_json(data / "ground_truth.json")

    baseline_single = _run(
        data,
        work / "baseline_single",
        work / "snapshots_single",
        workers=1,
        interval=1,
    )
    baseline_multi = _run(
        data,
        work / "baseline_multi",
        work / "snapshots_multi",
        workers=4,
        interval=2,
    )
    concurrency_state = _diff_state(_state(baseline_single), _state(baseline_multi))
    _check(
        checks,
        "concurrency_state_equivalence",
        concurrency_state["equal"],
        concurrency_state,
    )
    _check(
        checks,
        "failure_order_stable",
        baseline_multi["failures"]
        == sorted(
            baseline_multi["failures"],
            key=lambda item: (
                item["series_name"],
                -1 if item["frame_index"] is None else item["frame_index"],
                item["stage"],
                item["error_code"],
                item["injection_id"] or "",
                item["message"],
            ),
        ),
    )
    _check(
        checks,
        "pool_reentry_observed",
        any(track["reentered_pool"] for track in baseline_multi["tracks"]),
        {"reentered_tracks": sum(1 for track in baseline_multi["tracks"] if track["reentered_pool"])},
    )

    repeat_cached = _run(
        data,
        work / "baseline_repeat",
        work / "snapshots_multi",
        workers=4,
        interval=2,
    )
    cached_state = _diff_state(_state(baseline_multi), _state(repeat_cached))
    _check(
        checks,
        "unchanged_run_uses_cache",
        repeat_cached["summary"]["reused_frame_count"] == baseline_multi["summary"]["frame_count"]
        and repeat_cached["summary"]["reused_track_count"] == baseline_multi["summary"]["track_count"]
        and cached_state["equal"],
        {
            "state": cached_state,
            "reused_frames": repeat_cached["summary"]["reused_frame_count"],
            "reused_tracks": repeat_cached["summary"]["reused_track_count"],
        },
    )

    edit_data = work / "editable_dataset"
    copy_dataset(data, edit_data, force=True)
    edit_snapshot = work / "snapshots_edit"
    _run(edit_data, work / "edit_baseline", edit_snapshot, workers=4, interval=2)
    operations_payload = read_json(edit_data / "edit_sequence.json")
    step_reports: list[dict[str, Any]] = []
    for index, operation in enumerate(operations_payload["operations"], start=1):
        apply_operation(edit_data, operation)
        report = _run(
            edit_data,
            work / f"edit_step_{index:02d}",
            edit_snapshot,
            workers=4,
            interval=2,
        )
        step_reports.append(
            {
                "id": operation["id"],
                "type": operation["type"],
                "summary": report["summary"],
                "state": _state(report),
                "ledger": _ledger_projection(report),
            }
        )
    final_incremental = step_reports[-1]
    final_full = _run(
        edit_data,
        work / "edit_final_full",
        work / "snapshots_edit_full",
        workers=1,
        interval=1,
        force_full=True,
    )
    incremental_state = _diff_state(_state(final_full), final_incremental["state"])
    _check(
        checks,
        "incremental_state_equivalence",
        incremental_state["equal"],
        incremental_state,
    )
    ledger_equal = final_incremental["ledger"] == _ledger_projection(final_full)
    _check(
        checks,
        "incremental_ledger_projection",
        ledger_equal,
        {
            "final_incremental_reused_frames": final_incremental["summary"]["reused_frame_count"],
            "final_incremental_reused_tracks": final_incremental["summary"]["reused_track_count"],
            "projection_equal": ledger_equal,
            "note": "Operational reused/recomputed labels intentionally differ; final cache-key dependency projections must match.",
        },
    )
    edit_checks = {
        "version_switch_invalidates_all_frames": (
            step_reports[0]["summary"]["recomputed_frame_count"]
            == step_reports[0]["summary"]["frame_count"]
        ),
        "candidate_replace_is_frame_local": (
            step_reports[1]["summary"]["recomputed_frame_count"] == 1
            and step_reports[1]["summary"]["reused_frame_count"] > 0
        ),
        "image_replace_is_frame_local": (
            step_reports[2]["summary"]["recomputed_frame_count"] == 1
            and step_reports[2]["summary"]["reused_frame_count"] > 0
        ),
        "iou_change_reuses_frames_but_recomputes_tracks": (
            step_reports[3]["summary"]["reused_frame_count"]
            == step_reports[3]["summary"]["frame_count"]
            and step_reports[3]["summary"]["reused_track_count"] == 0
        ),
        "capacity_change_invalidates_detection_and_tracks": (
            step_reports[4]["summary"]["recomputed_frame_count"]
            == step_reports[4]["summary"]["frame_count"]
        ),
        "final_local_edit_reuses_unaffected_tracks": (
            step_reports[5]["summary"]["reused_track_count"] > 0
            and step_reports[5]["summary"]["recomputed_track_count"] > 0
        ),
    }
    _check(
        checks,
        "edit_sequence_invalidation_semantics",
        all(edit_checks.values()),
        edit_checks,
    )
    _check(
        checks,
        "incremental_does_not_reuse_affected_tracks",
        final_incremental["summary"]["reused_track_count"] > 0
        and final_incremental["summary"]["recomputed_track_count"] > 0,
        {
            "reused_tracks": final_incremental["summary"]["reused_track_count"],
            "recomputed_tracks": final_incremental["summary"]["recomputed_track_count"],
        },
    )

    conversion = baseline_multi["conversion"]
    v1_to_v2 = conversion["checks"]["v1_to_v2"]
    v2_to_v1 = conversion["checks"]["v2_to_v1"]
    sidecar_path = Path(conversion["outputs"]["sidecar"])
    _check(
        checks,
        "conversion_v1_to_v2_exact",
        v1_to_v2["serialized_then_parsed"]
        and v1_to_v2["max_coordinate_deviation"] == 0.0
        and v1_to_v2["max_score_deviation"] == 0.0,
        {
            "max_coordinate_deviation": v1_to_v2["max_coordinate_deviation"],
            "max_score_deviation": v1_to_v2["max_score_deviation"],
        },
    )
    _check(
        checks,
        "conversion_v2_to_v1_residual_captured",
        v2_to_v1["serialized_then_parsed"]
        and v2_to_v1["max_coordinate_deviation"] > 0
        and len(v2_to_v1["coordinate_abs_error_histogram"]) > 1
        and sidecar_path.exists()
        and conversion["sidecar_entry_count"] > 0,
        {
            "max_coordinate_deviation": v2_to_v1["max_coordinate_deviation"],
            "max_score_deviation": v2_to_v1["max_score_deviation"],
            "coordinate_histogram": v2_to_v1["coordinate_abs_error_histogram"],
            "score_histogram": v2_to_v1["score_abs_error_histogram"],
            "sidecar_exists": sidecar_path.exists(),
            "sidecar_entries": conversion["sidecar_entry_count"],
        },
    )
    _check(
        checks,
        "conversion_class_mapping_restored",
        v1_to_v2["class_mapping"]["complete"] and v2_to_v1["class_mapping"]["complete"],
        {
            "v1_to_v2": v1_to_v2["class_mapping"],
            "v2_to_v1": v2_to_v1["class_mapping"],
        },
    )
    _check(
        checks,
        "score_residual_histogram_nonzero",
        any(float(key) > 0 for key in v2_to_v1["score_abs_error_histogram"]),
        v2_to_v1["score_abs_error_histogram"],
    )

    mismatch_data = work / "version_mismatch_dataset"
    copy_dataset(data, mismatch_data, force=True)
    v2_path = mismatch_data / "candidates_v2.json"
    document = read_json(v2_path)
    changed = False
    for frame in document["frames"]:
        if frame["series_name"] == "alpha_seq" and frame["frame_index"] == 2:
            frame["candidates"][0]["class_name"] = "wrong_class"
            changed = True
            break
    if not changed:
        raise RuntimeError("version mismatch fixture target not found")
    atomic_write_json(v2_path, document)
    mismatch_report = _run(
        mismatch_data,
        work / "version_mismatch_out",
        work / "snapshots_mismatch",
        workers=2,
        interval=2,
    )
    mismatch_failures = [
        failure
        for failure in mismatch_report["failures"]
        if failure["error_code"] == "E_VERSION_MISMATCH"
        and failure["series_name"] == "alpha_seq"
        and failure["frame_index"] == 2
    ]
    failed_keys = {
        (failure["series_name"], failure["frame_index"])
        for failure in mismatch_report["failures"]
        if failure["frame_index"] is not None
    }
    mismatch_track_leak = any(
        (track["series_name"], frame["frame_index"]) in failed_keys
        for track in mismatch_report["tracks"]
        for frame in track["frames"]
    )
    _check(
        checks,
        "version_mismatch_isolation",
        bool(mismatch_failures) and not mismatch_track_leak,
        {"failures": mismatch_failures, "failed_frame_in_track": mismatch_track_leak},
    )

    injected_data = work / "injected_dataset"
    injection_result = apply_injections(
        data,
        injected_data,
        data / "injections.json",
        force=True,
    )
    injection_report = _run(
        injected_data,
        work / "injected_out",
        work / "snapshots_injected",
        workers=4,
        interval=2,
    )
    injection_hits = injection_report["injections"]["hits"]
    expected_by_id = {
        item["id"]: item["expected_code"]
        for item in injection_report["injections"]["configured"]
    }
    all_expected_observed = all(
        item["hit"] and item["observed_error_code"] == expected_by_id.get(item["id"])
        for item in injection_hits
    )
    injected_failed_keys = {
        (failure["series_name"], failure["frame_index"])
        for failure in injection_report["failures"]
        if failure["frame_index"] is not None
    }
    injected_track_leak = any(
        (track["series_name"], frame["frame_index"]) in injected_failed_keys
        for track in injection_report["tracks"]
        for frame in track["frames"]
    )
    _check(
        checks,
        "injection_all_items_hit_with_business_codes",
        len(injection_hits) == len(expected_by_id) and all_expected_observed,
        {
            "configured": len(expected_by_id),
            "hits": injection_hits,
            "failures": injection_report["failures"],
        },
    )
    _check(
        checks,
        "injection_failed_frames_excluded_from_tracks",
        not injected_track_leak,
        {"failed_frames": sorted(injected_failed_keys), "leak": injected_track_leak},
    )
    _check(
        checks,
        "injection_continues_other_frames",
        injection_report["summary"]["successful_frame_count"] > 0,
        injection_report["summary"],
    )

    recovery_root = work / "recovery"
    crash_out = recovery_root / "crash_out"
    recovery_snapshots = recovery_root / "snapshots"
    crash_command = [
        sys.executable,
        "-m",
        "candidate_exchange",
        "run",
        "--data",
        str(data),
        "--out",
        str(crash_out),
        "--snapshots",
        str(recovery_snapshots),
        "--workers",
        "2",
        "--snapshot-interval",
        "2",
        "--crash-after-stage",
        "matching",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root) + os.pathsep + environment.get("PYTHONPATH", "")
    crash_process = subprocess.run(
        crash_command,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    report_after_crash = crash_out / "validation_report.json"
    report_existed_after_termination = report_after_crash.exists()
    temporary_marker = recovery_snapshots / "versions" / ".tmp-ignored-partial"
    temporary_marker.mkdir(parents=True, exist_ok=True)
    (temporary_marker / "partial.txt").write_text("incomplete snapshot", encoding="utf-8")
    restart_command = [
        sys.executable,
        "-m",
        "candidate_exchange",
        "run",
        "--data",
        str(data),
        "--out",
        str(crash_out),
        "--snapshots",
        str(recovery_snapshots),
        "--workers",
        "2",
        "--snapshot-interval",
        "2",
    ]
    restart_process = subprocess.run(
        restart_command,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    recovered_report = (
        read_json(report_after_crash) if report_after_crash.exists() else None
    )
    recovery_ok = (
        crash_process.returncode == 75
        and not report_existed_after_termination
        and restart_process.returncode == 0
        and recovered_report is not None
        and recovered_report["snapshot_recovery_at_start"]["recovered"]
        and recovered_report["snapshot"]["recovery_point_version"] is not None
        and "detection" in recovered_report["snapshot_recovery_at_start"]["available_stages"]
    )
    recovery_state_match = (
        recovered_report is not None
        and _diff_state(_state(baseline_multi), _state(recovered_report))["equal"]
    )
    _check(
        checks,
        "forced_termination_and_snapshot_recovery",
        recovery_ok,
        {
            "crash_exit_code": crash_process.returncode,
            "crash_stderr": crash_process.stderr.strip(),
            "report_existed_after_termination": report_existed_after_termination,
            "restart_exit_code": restart_process.returncode,
            "recovery_at_start": recovered_report["snapshot_recovery_at_start"]
            if recovered_report
            else None,
            "final_snapshot": recovered_report["snapshot"] if recovered_report else None,
            "ignored_temporary_directory": ".tmp-ignored-partial",
        },
    )
    _check(
        checks,
        "recovered_state_matches_full_run",
        recovery_state_match,
        {"state": _diff_state(_state(baseline_multi), _state(recovered_report))}
        if recovered_report
        else {"error": "no recovered report"},
    )

    total = len(checks)
    passed = sum(1 for value in checks.values() if value["passed"])
    report = {
        "schema_version": "candidate-exchange-validation-v1",
        "generated_unix_ms": int(time.time() * 1000),
        "dataset": {
            "path": str(data),
            "seed": dataset_summary["seed"],
            "series_count": len(dataset_summary["series"]),
            "frame_count": sum(len(frames) for frames in dataset_summary["series"].values()),
            "candidate_count": baseline_multi["summary"]["candidate_count"],
        },
        "checks": checks,
        "summary": {
            "all_checks_passed": passed == total,
            "total_checks": total,
            "passed_checks": passed,
            "failed_checks": total - passed,
        },
        "results": {
            "baseline": {
                "single_thread": baseline_single["summary"],
                "multi_thread": baseline_multi["summary"],
            },
            "incremental": {
                "steps": [
                    {
                        "id": step["id"],
                        "type": step["type"],
                        "summary": step["summary"],
                    }
                    for step in step_reports
                ],
                "final_full": final_full["summary"],
                "final_incremental": final_incremental["summary"],
            },
            "conversion": {
                "v1_to_v2_max_coordinate_deviation": v1_to_v2["max_coordinate_deviation"],
                "v1_to_v2_max_score_deviation": v1_to_v2["max_score_deviation"],
                "v2_to_v1_max_coordinate_deviation": v2_to_v1["max_coordinate_deviation"],
                "v2_to_v1_max_score_deviation": v2_to_v1["max_score_deviation"],
                "v2_to_v1_histogram_buckets": len(
                    v2_to_v1["coordinate_abs_error_histogram"]
                ),
                "v2_to_v1_coordinate_histogram": v2_to_v1[
                    "coordinate_abs_error_histogram"
                ],
                "v2_to_v1_score_histogram": v2_to_v1["score_abs_error_histogram"],
                "class_mapping_complete": (
                    v1_to_v2["class_mapping"]["complete"]
                    and v2_to_v1["class_mapping"]["complete"]
                ),
                "sidecar_path": str(sidecar_path),
                "sidecar_metadata_hash": conversion["sidecar_metadata_hash"],
            },
            "injection": {
                "configured_count": len(expected_by_id),
                "hit_count": sum(1 for item in injection_hits if item["hit"]),
                "hits": injection_hits,
                "failure_count": len(injection_report["failures"]),
                "failed_frames": sorted(injected_failed_keys),
                "application_log": injection_result["applications"],
            },
            "snapshot_recovery": {
                "crash_exit_code": crash_process.returncode,
                "restart_exit_code": restart_process.returncode,
                "report_existed_after_termination": report_existed_after_termination,
                "recovery_point_version": recovered_report["snapshot"][
                    "recovery_point_version"
                ]
                if recovered_report
                else None,
                "recovered_from_version": recovered_report["snapshot"][
                    "recovered_from_version"
                ]
                if recovered_report
                else None,
                "available_stages_at_start": recovered_report[
                    "snapshot_recovery_at_start"
                ]["available_stages"]
                if recovered_report
                else [],
                "ignored_temporary_directories": recovered_report[
                    "snapshot_recovery_at_start"
                ].get("ignored_temporary_directories", [])
                if recovered_report
                else [],
            },
        },
    }
    report_path = output / "validation_report.json"
    summary_path = output / "validation_summary.md"
    report["artifacts"] = {
        "validation_report": str(report_path),
        "validation_summary": str(summary_path),
        "work_dir": str(work),
        "baseline_report": baseline_multi["report_path"],
        "injected_report": injection_report["report_path"],
        "recovery_report": str(report_after_crash),
    }
    atomic_write_json(report_path, report)
    atomic_write_text(summary_path, _summary_markdown(report))
    if not keep_work:
        # Keep reports and conversion artifacts; drop bulky editable copies and
        # immutable snapshot trees. The recovery evidence is embedded in the report.
        for path in (
            work / "editable_dataset",
            work / "injected_dataset",
            work / "version_mismatch_dataset",
        ):
            remove_tree(path)
        for path in work.glob("snapshots_*"):
            remove_tree(path)
    return report
