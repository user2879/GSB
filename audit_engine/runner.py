"""并发审计调度、检查点恢复与原子报告。"""

from __future__ import annotations

import concurrent.futures as futures
import json
import os
from dataclasses import asdict
from typing import Any, Callable, Optional

from .budget import BudgetConfig
from .checkpoint import build_checkpoint, load_checkpoint, save_checkpoint, should_rebuild
from .engine import audit_file
from .errors import E_BUDGET, E_LOCK, E_TIMEOUT
from .locking import atomic_write
from .models import FileResult, NormalizationPolicy
from .parser import load_manifest_file, RecordParseError

# 进程级退出码
EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_BUDGET_OR_TIMEOUT = 2
EXIT_LOCK = 3
EXIT_FATAL = 4


def _serialize_file_result(result: FileResult) -> dict[str, Any]:
    payload = asdict(result)
    return payload


def _process_one(
    manifest_path: str,
    policies: list[NormalizationPolicy],
    budget: BudgetConfig,
    work_probe: Optional[Callable[[int], None]],
) -> FileResult:
    records, rejections, metadata = load_manifest_file(manifest_path)
    return audit_file(
        file_id=metadata["file_id"],
        path=manifest_path,
        root=metadata["root"],
        records=records,
        rejections=rejections,
        policies=policies,
        budget=budget,
        work_probe=work_probe,
    )


def _manifest_file_id(manifest_path: str) -> str:
    """读取清单内部 file_id；损坏文件返回 basename 以保证调度不中断。"""
    try:
        _, _, metadata = load_manifest_file(manifest_path)
        return metadata["file_id"]
    except RecordParseError:
        return _file_id_of(manifest_path)


def _run_serial(
    paths: list[str],
    policies: list[NormalizationPolicy],
    budget: BudgetConfig,
    work_probe: Optional[Callable[[int], None]],
) -> list[FileResult]:
    return [_process_one(path, policies, budget, work_probe) for path in paths]


def _run_threaded(
    paths: list[str],
    policies: list[NormalizationPolicy],
    budget: BudgetConfig,
    work_probe: Optional[Callable[[int], None]],
) -> list[FileResult]:
    with futures.ThreadPoolExecutor(max_workers=max(1, budget.concurrency)) as pool:
        submitted = [
            pool.submit(_process_one, path, policies, budget, work_probe) for path in paths
        ]
        return [future.result() for future in submitted]


def _run_processes(
    paths: list[str],
    policies: list[NormalizationPolicy],
    budget: BudgetConfig,
    work_probe: Optional[Callable[[int], None]] = None,
) -> list[FileResult]:
    """独立子进程后端：每个文件一个 Process，超时真实 terminate，
    被终止文件不会中断同批其余文件。
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn") if os.name == "nt" else mp.get_context("fork")
    results: list[Optional[FileResult]] = [None] * len(paths)
    semaphore = ctx.Semaphore(max(1, budget.concurrency))

    def _worker(path, policies_arg, budget_arg, probe_arg, queue_arg):
        with semaphore:
            try:
                queue_arg.put(("ok", _process_one(path, policies_arg, budget_arg, probe_arg)))
            except Exception as exc:  # noqa: BLE001
                queue_arg.put(("error", f"{type(exc).__name__}: {exc}"))

    for idx, path in enumerate(paths):
        queue: "mp.Queue" = ctx.Queue()
        process = ctx.Process(
            target=_worker,
            args=(path, policies, budget, work_probe, queue),
        )
        process.start()
        process.join(timeout=budget.per_file_timeout_seconds)
        if process.is_alive():
            process.terminate()  # 真实机制终止该文件的后续处理
            process.join(timeout=5)
            placeholder = _budget_placeholder(path, E_TIMEOUT)
            object.__setattr__(placeholder, "file_id", _manifest_file_id(path))
            results[idx] = placeholder
            continue
        if not queue.empty():
            kind, payload = queue.get()
            if kind == "ok":
                results[idx] = payload
            else:
                placeholder = _budget_placeholder(path, E_TIMEOUT)
                object.__setattr__(placeholder, "file_id", _manifest_file_id(path))
                placeholder.status = "error"
                placeholder.codes = [f"WORKER_ERROR:{payload.split(':', 1)[0]}"]
                results[idx] = placeholder
        else:
            placeholder = _budget_placeholder(path, E_TIMEOUT)
            object.__setattr__(placeholder, "file_id", _manifest_file_id(path))
            results[idx] = placeholder
    return [result for result in results if result is not None]


def _budget_placeholder(path: str, code: str) -> FileResult:
    return FileResult(
        file_id=os.path.basename(path),
        path=path,
        status="timeout" if code == E_TIMEOUT else "budget",
        codes=[code],
        verdicts=[],
        rejections=[],
        elapsed_seconds=0.0,
        processed_records=0,
        dropped_records=-1,
        truth={"consistent_count": 0, "inconsistent_count": 0, "no_truth_count": 0,
               "inconsistent_details": []},
    )


def _aggregate(results: list[FileResult]) -> dict[str, Any]:
    aggregate_codes: dict[str, int] = {}
    for result in results:
        # 记录级业务码按每条策略裁决计数，解析拒绝按记录计数
        for verdict in result.verdicts:
            for code in verdict.codes:
                aggregate_codes[code] = aggregate_codes.get(code, 0) + 1
        for rejection in result.rejections:
            aggregate_codes[rejection.code] = aggregate_codes.get(rejection.code, 0) + 1
        for code in result.codes:
            if code in (E_TIMEOUT, E_BUDGET):
                aggregate_codes[code] = aggregate_codes.get(code, 0) + 1
    return {
        "files": len(results),
        "processed_records": sum(r.processed_records for r in results),
        "dropped_records": sum(r.dropped_records for r in results),
        "rejected_records": sum(len(r.rejections) for r in results),
        "code_counts": dict(sorted(aggregate_codes.items())),
        "truth_consistent": sum(r.truth.get("consistent_count", 0) for r in results),
        "truth_inconsistent": sum(r.truth.get("inconsistent_count", 0) for r in results),
    }


def run_batch(
    manifest_paths: list[str],
    policies: list[NormalizationPolicy],
    budget: BudgetConfig,
    checkpoint_dir: str,
    report_path: str,
    backend: str = "thread",
    work_probe: Optional[Callable[[int], None]] = None,
) -> tuple[dict[str, Any], int]:
    """运行整批审计并返回 (报告对象, 退出码)。"""
    ordered_paths = sorted(os.path.abspath(p) for p in manifest_paths)
    file_ids = [_manifest_file_id(path) for path in ordered_paths]
    if len(set(file_ids)) != len(file_ids):
        duplicates = sorted({fid for fid in file_ids if file_ids.count(fid) > 1})
        raise ValueError(f"清单 file_id 在批次内不唯一: {duplicates}")
    checkpoint = load_checkpoint(checkpoint_dir, policies)
    completed: set[str] = set()
    prior_results: list[FileResult] = []
    if checkpoint and not should_rebuild(checkpoint):
        completed = {item["file_id"] for item in checkpoint.get("completed_files", [])}
        prior_results = _restore_results(checkpoint)
    pending = [p for p in ordered_paths if _manifest_file_id(p) not in completed]

    if backend == "serial":
        new_results = _run_serial(pending, policies, budget, work_probe)
    elif backend == "process":
        new_results = _run_processes(pending, policies, budget, work_probe)
    else:
        new_results = _run_threaded(pending, policies, budget, work_probe)

    all_results = prior_results + new_results
    all_results.sort(key=lambda r: r.file_id)

    completed_payload = [
        {"file_id": result.file_id, "status": result.status, "codes": result.codes}
        for result in all_results
    ]
    checkpoint_data = build_checkpoint(completed_payload, _aggregate(all_results), policies)
    try:
        save_checkpoint(checkpoint_dir, checkpoint_data)
    except Exception as exc:  # E_LOCK 透传
        if getattr(exc, "code", None) == E_LOCK:
            raise
        raise

    report = {
        "policies": [p.as_dict() for p in policies],
        "budget": {
            "concurrency": budget.concurrency,
            "per_file_timeout_seconds": budget.per_file_timeout_seconds,
            "max_records_per_file": budget.max_records_per_file,
            "backend": backend,
        },
        "aggregate": _aggregate(all_results),
        "files": [_serialize_file_result(result) for result in all_results],
    }
    atomic_write(report_path, json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return report, _exit_code(all_results)


def _file_id_of(path: str) -> str:
    return os.path.basename(path)


def _restore_results(checkpoint: dict[str, Any]) -> list[FileResult]:
    """检查点只保存文件级状态；恢复后待续接文件重新审计，已完成文件以占位状态保留。"""
    restored: list[FileResult] = []
    for item in checkpoint.get("completed_files", []):
        restored.append(
            FileResult(
                file_id=item["file_id"],
                path="",
                status=item["status"],
                codes=item["codes"],
                verdicts=[],
                rejections=[],
                elapsed_seconds=0.0,
                processed_records=0,
                dropped_records=0,
                truth={"consistent_count": 0, "inconsistent_count": 0, "no_truth_count": 0,
                       "inconsistent_details": []},
            )
        )
    return restored


def _exit_code(results: list[FileResult]) -> int:
    if any(r.status in ("timeout", "budget") or E_TIMEOUT in r.codes or E_BUDGET in r.codes
           for r in results):
        return EXIT_BUDGET_OR_TIMEOUT
    if any(r.codes for r in results):
        return EXIT_VIOLATIONS
    return EXIT_OK
