"""可重复演示入口：一条命令完成生成→切分→并行测量→降级→异常注入→哈希。

固定序列、固定参数、无时间戳，两次相同输入与预算的运行逐字节一致。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from .errors import (
    E_AUDIO_FORMAT,
    E_BLOCK_BOUNDS,
    E_MERGE_ORDER,
    E_PEAK_OVER,
    E_THREAD_RANGE,
    E_WINDOW_INVALID,
)
from .errors import AudioFormatError, BlockBoundsError, ThreadRangeError, WindowInvalidError, MergeOrderError, PeakOverError
from .meter import measure_whole_program
from .report import canonical_json, sha256_bytes, sha256_file, write_report, build_run_report
from .scheduler import SchedulerConfig, run_scheduler
from .synth import write_program
from .tasks import serialize_manifest, split_tasks, validate_window_params, WARMUP
from .wavio import read_wav24

WINDOW_SECONDS = 3.0
OVERLAP_SECONDS = 0.0


def _run_scenarios(workspace: Path) -> Dict[str, Any]:
    wav = workspace / "sample_5ch1_48k24.wav"
    samples, rate, channels = read_wav24(str(wav))
    tasks = split_tasks(samples.shape[1], channels, WINDOW_SECONDS, OVERLAP_SECONDS)
    manifest_text = serialize_manifest(
        tasks,
        {"window_seconds": WINDOW_SECONDS, "overlap_seconds": OVERLAP_SECONDS, "sample_rate": rate},
    )
    (workspace / "task_manifest.json").write_text(manifest_text, encoding="utf-8")

    reference = measure_whole_program(samples)
    (workspace / "reference_whole.json").write_text(canonical_json(reference), encoding="utf-8")

    # 内存预算三档：宽松（全驻留）、紧张（触发 LRU 淘汰与 RECOMPUTE）、极紧（触发 E_BUDGET_DENIED）
    from .bufferpool import RAW_GRID
    chunk_bytes = RAW_GRID * channels * 8  # 一个 10ms 网格原始块（float64, 6 声道）
    scenarios = [
        ("S1_unrestricted_4w", SchedulerConfig(max_workers=4, memory_budget_bytes=None)),
        ("S2_serial_1w", SchedulerConfig(max_workers=1, memory_budget_bytes=None)),
        ("S3_tight_4w", SchedulerConfig(max_workers=4, memory_budget_bytes=chunk_bytes * 2)),
        ("S4_minimal_2w", SchedulerConfig(max_workers=2, memory_budget_bytes=chunk_bytes - 1)),
        ("S5_timebudget_serial", SchedulerConfig(max_workers=8, task_time_budget_seconds=1e-9)),
    ]

    run_hashes: Dict[str, str] = {}
    run_index: List[Dict[str, Any]] = []
    for run_id, cfg in scenarios:
        run = run_scheduler(samples, tasks, cfg)
        report = build_run_report(
            run_id,
            {
                "max_workers": cfg.max_workers,
                "memory_budget_bytes": cfg.memory_budget_bytes,
                "task_time_budget_seconds": cfg.task_time_budget_seconds,
                "window_seconds": WINDOW_SECONDS,
            },
            json.loads(manifest_text),
            run,
            reference,
        )
        info = write_report(str(workspace / "runs"), f"{run_id}.json", report)
        run_hashes[run_id] = info["sha256"]
        run_index.append(
            {
                "run_id": run_id,
                "report_sha256": info["sha256"],
                "effective_workers": run.effective_workers,
                "cache_stats": report["cache_stats"],
                "degradation": report["degradation"],
                "comparison_pass": report["merged"]["reference_comparison"]["all_pass"],
            }
        )

    faults = _inject_faults(samples, tasks)
    return {
        "schema": "loudbench.demo-index/v1",
        "window_seconds": WINDOW_SECONDS,
        "warmup_samples": WARMUP,
        "task_count": len(tasks),
        "reference": reference,
        "runs": run_index,
        "run_report_hashes": run_hashes,
        "fault_injection": faults,
    }


def _inject_faults(samples, tasks) -> List[Dict[str, Any]]:
    faults: List[Dict[str, Any]] = []

    def record(name: str, exc: Exception) -> None:
        faults.append({"case": name, "error": exc.to_dict() if hasattr(exc, "to_dict") else {"message": str(exc)}})

    # E_AUDIO_FORMAT：非法声道数
    try:
        bad = samples[:3, :]
        from .wavio import write_wav24
        import tempfile, os

        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        write_wav24(tmp, bad)
        read_wav24(tmp)
    except AudioFormatError as exc:
        record("illegal_channel_count", exc)
    finally:  # pragma: no cover - 清理
        try:
            os.remove(tmp)
        except OSError:
            pass

    # E_WINDOW_INVALID / E_BLOCK_BOUNDS
    try:
        validate_window_params(0.2, 0.0)
    except WindowInvalidError as exc:
        record("window_too_short", exc)
    try:
        validate_window_params(2.0, 2.0)
    except WindowInvalidError as exc:
        record("overlap_covers_window", exc)

    # E_BLOCK_BOUNDS：窗口重叠超出可用 warm-up 区间
    try:
        validate_window_params(3.0, 1.0)
    except BlockBoundsError as exc:
        record("overlap_exceeds_warmup", exc)

    # E_THREAD_RANGE
    try:
        run_scheduler(samples, tasks, SchedulerConfig(max_workers=0))
    except ThreadRangeError as exc:
        record("thread_out_of_range", exc)
    try:
        run_scheduler(samples, tasks, SchedulerConfig(max_workers=10_000))
    except ThreadRangeError as exc:
        record("thread_over_hard_limit", exc)

    # E_MERGE_ORDER（注入归并故障）
    try:
        run_scheduler(samples, tasks, SchedulerConfig(max_workers=2, inject_merge_fault=True))
    except MergeOrderError as exc:
        record("merge_order_fault", exc)

    # E_PEAK_OVER（构造超过 0 dBTP 的素材并开启量程检查）
    try:
        hot = samples.copy()
        hot[0, :] = 1.0
        run_scheduler(hot, tasks, SchedulerConfig(max_workers=1, raise_on_peak_over=True))
    except PeakOverError as exc:
        record("peak_over_range", exc)

    # E_BUDGET_DENIED 已在 S4 场景触发，这里再给一个确定性单缓冲拒绝样例
    try:
        run_scheduler(samples, tasks, SchedulerConfig(max_workers=1, memory_budget_bytes=1))
    except Exception as exc:  # 预算拒绝以报告错误形式出现，不抛出；保留兜底
        record("budget_denied_edge", exc)

    expected = {
        E_AUDIO_FORMAT,
        E_WINDOW_INVALID,
        E_BLOCK_BOUNDS,
        E_THREAD_RANGE,
        E_MERGE_ORDER,
        E_PEAK_OVER,
    }
    seen = {f["error"]["code"] for f in faults if "code" in f["error"]}
    missing = sorted(expected - seen)
    return {
        "faults": faults,
        "covered_codes": sorted(seen),
        "expected_codes": sorted(expected),
        "missing_codes": missing,
        "budget_denied_demonstrated_in_run": "S4_minimal_2w",
    }


def run_demo(workspace_dir: str, clean: bool = True) -> Dict[str, str]:
    workspace = Path(workspace_dir)
    if clean and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    paths = write_program(str(workspace))
    index = _run_scenarios(workspace)

    index_text = canonical_json(index)
    (workspace / "demo_index.json").write_text(index_text, encoding="utf-8")

    hash_manifest = {
        "schema": "loudbench.hash-manifest/v1",
        "files": {
            "sample_5ch1_48k24.wav": sha256_file(paths["wav"]),
            "ground_truth_labels.json": sha256_file(paths["labels"]),
            "task_manifest.json": sha256_file(str(workspace / "task_manifest.json")),
            "reference_whole.json": sha256_file(str(workspace / "reference_whole.json")),
            "demo_index.json": sha256_bytes(index_text.encode("utf-8")),
        },
        "run_reports": index["run_report_hashes"],
    }
    for p in sorted((workspace / "runs").glob("*.json")):
        hash_manifest["files"][f"runs/{p.name}"] = sha256_file(str(p))
    text = canonical_json(hash_manifest)
    (workspace / "hash_manifest.json").write_text(text, encoding="utf-8")
    return {
        "workspace": str(workspace),
        "hash_manifest": str(workspace / "hash_manifest.json"),
        "demo_index": str(workspace / "demo_index.json"),
    }


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="响度与动态范围并行测量工作台演示")
    parser.add_argument("--workspace", default="workspace/demo")
    parser.add_argument("--verify-twice", action="store_true", help="连续运行两次并比对逐字节一致")
    args = parser.parse_args(argv)

    first = run_demo(args.workspace)
    print(json.dumps(first, indent=2, ensure_ascii=False))
    if args.verify_twice:
        first_manifest = Path(first["hash_manifest"]).read_bytes()
        run_demo(args.workspace)
        second_manifest = Path(first["hash_manifest"]).read_bytes()
        if first_manifest != second_manifest:
            raise SystemExit("两次运行哈希清单不一致，确定性验证失败")
        print("两次运行逐字节一致：hash_manifest.json SHA256 =", sha256_bytes(second_manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
