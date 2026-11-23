"""端到端验收：生成确定性样本 → 全量/增量等价 → 异常/并发/预算报告汇总。

用法：
    python -m retbench.acceptance <workspace_dir> [--samples N]
输出 <workspace_dir>/reports/acceptance_summary.json 并在 stdout 打印摘要。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from typing import Any

from .concurrency import BoundedPool
from .config import Command, load_commands, load_config
from .engine import Engine
from .errors import E_INTERACT_BUDGET_EXCEEDED, E_SNAPSHOT_STALE
from .fingerprint import digest_obj
from .generator import generate_workspace
from .runner import _normalize, _write_json, equivalence_check


def _report_fingerprint(report: dict[str, Any]) -> str:
    comparable = copy.deepcopy(report)
    comparable.pop("concurrency", None)
    return digest_obj(_normalize(comparable))


def run_acceptance(workspace: str, sample_count: int = 12) -> dict[str, Any]:
    if os.path.exists(workspace):
        shutil.rmtree(workspace, ignore_errors=True)
    paths = generate_workspace(workspace, sample_count=sample_count)
    config_path = paths["config"]
    sample_root = paths["samples"]
    command_path = paths["commands"]
    report_dir = os.path.join(workspace, "reports")
    os.makedirs(report_dir, exist_ok=True)

    config = load_config(config_path)
    commands = load_commands(command_path, config)

    # ---- 增量路径（含基线）----
    inc_engine = Engine(config, sample_root,
                        os.path.join(workspace, "snapshots_inc"))
    inc_engine.load_samples()
    pool = BoundedPool(config.thread_pool_size, config.work_ms_per_task)
    baseline = inc_engine.baseline_report(pool)
    timeline: list[dict[str, Any]] = []
    budget_events: list[dict[str, Any]] = []
    decision_trace: list[dict[str, Any]] = [baseline["decision"]]
    for index, command in enumerate(commands):
        outcome = inc_engine.handle_command(command, index + 1, pool)
        decision_trace.append(outcome.report["decision"])
        timeline.append({
            "seq": outcome.seq,
            "op": outcome.op,
            "elapsed_ms": round(outcome.elapsed_ms, 3),
            "reused": outcome.reused,
            "recomputed": outcome.recomputed,
            "affected_samples": outcome.affected_samples,
            "stale": outcome.stale,
        })
        if outcome.budget_exceeded:
            budget_events.append({
                "seq": outcome.seq,
                "code": E_INTERACT_BUDGET_EXCEEDED,
                "elapsed_ms": round(outcome.elapsed_ms, 3),
            })
    incremental_report = inc_engine.last_report

    # ---- 全量路径（隔离副本，从空白重建）----
    full_root = os.path.join(workspace, "_full_samples")
    shutil.copytree(sample_root, full_root)
    full_config = load_config(config_path)
    full_engine = Engine(full_config, full_root,
                         os.path.join(workspace, "snapshots_full"))
    full_engine.load_samples()
    for command in commands:
        full_engine.apply_command(command)
    full_pool = BoundedPool(full_config.thread_pool_size,
                            full_config.work_ms_per_task)
    full_pipeline = full_engine.run_pipeline(full_pool)
    full_report = full_engine.build_report(full_pipeline, None, False, 0.0, [], 0)

    equivalence = equivalence_check(full_report, incremental_report)

    # ---- 并发一致性：不同并发度结果指纹相同 ----
    concurrency: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for workers in (1, 2, 4):
        worker_config = load_config(config_path)
        worker_config.thread_pool_size = workers
        engine = Engine(worker_config, sample_root,
                        os.path.join(workspace, f"snap_pool_{workers}"))
        engine.load_samples()
        worker_pool = BoundedPool(workers)
        engine.baseline_report(worker_pool)
        fp = _report_fingerprint(engine.last_report)
        fingerprints.add(fp)
        concurrency.append({
            "configured_workers": workers,
            "observed_peak": worker_pool.peak_concurrency,
            "report_fingerprint": fp,
        })

    # ---- 预算超限演示（独立引擎，小预算+模拟耗时）----
    budget_config = load_config(config_path)
    budget_config.latency_budget_ms = 5
    budget_config.work_ms_per_task = 20
    budget_engine = Engine(budget_config, sample_root,
                           os.path.join(workspace, "snap_budget"))
    budget_engine.load_samples()
    budget_pool = BoundedPool(budget_config.thread_pool_size, 20)
    budget_engine.baseline_report(budget_pool)
    budget_outcome = budget_engine.handle_command(
        Command(seq=0, op="set_weight",
                payload={"channel": "color", "value": 0.9}),
        1, budget_pool,
    )

    # ---- 快照陈旧演示 ----
    drift_config = load_config(config_path)
    drift_config.weights = dict(config.weights)
    drift_config.weights["edge"] = 0.77
    drift_engine = Engine(drift_config, sample_root,
                          os.path.join(workspace, "snapshots_inc"))
    stale_code, _ = drift_engine.resume_or_rebuild()

    summary = {
        "workspace": workspace,
        "sample_count": sample_count,
        "failures": baseline["failures"],
        "decision_trace": decision_trace,
        "equivalence": equivalence,
        "concurrency": {
            "runs": concurrency,
            "deterministic_across_workers": len(fingerprints) == 1,
        },
        "budget_demo": {
            "limit_ms": budget_config.latency_budget_ms,
            "exceeded": budget_outcome.budget_exceeded,
            "stale_summary_returned": budget_outcome.stale,
            "code": E_INTERACT_BUDGET_EXCEEDED
            if budget_outcome.budget_exceeded else None,
        },
        "snapshot_stale_demo": {
            "expected_code": E_SNAPSHOT_STALE,
            "observed_code": stale_code,
            "rebuild_triggered": stale_code == E_SNAPSHOT_STALE,
        },
        "command_timeline": timeline,
        "budget_events": budget_events,
    }
    _write_json(os.path.join(report_dir, "acceptance_summary.json"), summary)
    return summary


def _load_history(report_dir: str) -> list[dict[str, Any]]:
    path = os.path.join(report_dir, "command_history.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="工作台端到端验收")
    parser.add_argument("workspace")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    summary = run_acceptance(args.workspace, args.samples)
    summary["wall_seconds"] = round(time.perf_counter() - start, 3)
    _write_json(os.path.join(args.workspace, "reports",
                             "acceptance_summary.json"), summary)
    print(json.dumps({
        "equivalent": summary["equivalence"]["equivalent"],
        "deterministic_across_workers":
            summary["concurrency"]["deterministic_across_workers"],
        "budget_exceeded_demo": summary["budget_demo"]["exceeded"],
        "snapshot_stale_demo": summary["snapshot_stale_demo"]["rebuild_triggered"],
        "failure_count": len(summary["failures"]),
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2))
    return 0 if summary["equivalence"]["equivalent"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
