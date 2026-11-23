"""工作台运行入口：全量/增量执行、等价性校验与 JSON 报告。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any

from .concurrency import BoundedPool
from .config import load_commands, load_config
from .engine import Engine
from .fingerprint import digest_obj
from .samples import load_injection_manifest


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def run_full(config_path: str, command_path: str | None = None,
             sample_root: str = "", report_path: str = "",
             snapshot_root: str | None = None) -> dict[str, Any]:
    """从空白状态按"最终样本+最终参数"一次性全量重建。

    若提供命令流，先按顺序重放其中的参数/替换类编辑（不经过缓存账本），
    使全量运行与增量运行收敛到同一最终状态。
    """

    config = load_config(config_path)
    engine = Engine(config, sample_root, snapshot_root)
    engine.load_samples()
    if command_path:
        commands = load_commands(command_path, config)
        for command in commands:
            engine.apply_command(command)
    pool = BoundedPool(config.thread_pool_size, config.work_ms_per_task)
    pipeline = engine.run_pipeline(pool)
    report = engine.build_report(pipeline, None, False, 0.0, [], 0)
    report["run_mode"] = "full"
    _write_json(report_path, report)
    return report


def run_incremental(config_path: str, command_path: str, sample_root: str,
                    out_dir: str, snapshot_root: str | None = None) -> dict[str, Any]:
    """按命令流逐条增量执行，输出每条命令后的稳定报告与最终汇总。"""

    config = load_config(config_path)
    commands = load_commands(command_path, config)
    engine = Engine(config, sample_root, snapshot_root)
    engine.load_samples()
    resume_code, cursor = engine.resume_or_rebuild()
    pool = BoundedPool(config.thread_pool_size, config.work_ms_per_task)

    # 基线：初始参数下的完整结果，同时填满账本，后续命令才能精确复用
    baseline = engine.baseline_report(pool)
    history: list[dict[str, Any]] = [{
        "seq": -1,
        "op": "baseline",
        "stale": False,
        "budget_exceeded": False,
        "elapsed_ms": 0.0,
        "affected_samples": [],
        "reused": [],
        "recomputed": baseline["ledger"]["recomputed"],
        "report": baseline,
    }]
    snapshots: list[str] = [engine.manifest.to_dict()]
    for index, command in enumerate(commands):
        outcome = engine.handle_command(command, index + 1, pool)
        history.append({
            "seq": outcome.seq,
            "op": outcome.op,
            "stale": outcome.stale,
            "budget_exceeded": outcome.budget_exceeded,
            "elapsed_ms": round(outcome.elapsed_ms, 3),
            "affected_samples": outcome.affected_samples,
            "reused": outcome.reused,
            "recomputed": outcome.recomputed,
            "report": outcome.report,
        })
        snapshots.append(engine.manifest.to_dict())

    final_report = copy.deepcopy(engine.last_report)
    final_report["run_mode"] = "incremental"
    final_report["snapshot_resume"] = {
        "code": resume_code,
        "cursor": cursor,
    }
    final_report["command_history_len"] = len(history)
    _write_json(os.path.join(out_dir, "incremental_report.json"), final_report)
    _write_json(os.path.join(out_dir, "command_history.json"), history)
    _write_json(os.path.join(out_dir, "snapshot_timeline.json"), snapshots)
    return final_report


# 增量 vs 全量逐字段对比时忽略的运行期元数据字段
_IGNORED_PATHS = {
    "run_mode", "command_cursor", "command", "budget", "snapshot_resume",
    "command_history_len",
}


def _normalize(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _normalize(value) for key, value in sorted(node.items())
                if key not in _IGNORED_PATHS}
    if isinstance(node, list):
        return [_normalize(item) for item in node]
    return node


def equivalence_check(full_report: dict[str, Any],
                      incremental_report: dict[str, Any]) -> dict[str, Any]:
    left = _normalize(full_report)
    right = _normalize(incremental_report)
    equal = digest_obj(left) == digest_obj(right)
    mismatches: list[str] = []
    if not equal:
        _collect_mismatches(left, right, "$", mismatches)
    return {
        "equivalent": equal,
        "full_fingerprint": digest_obj(left),
        "incremental_fingerprint": digest_obj(right),
        "mismatches": mismatches[:50],
    }


def _collect_mismatches(left: Any, right: Any, path: str,
                        out: list[str]) -> None:
    if type(left) is not type(right):
        out.append(f"{path}: type {type(left).__name__} != {type(right).__name__}")
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                out.append(f"{path}.{key}: missing side")
            else:
                _collect_mismatches(left[key], right[key], f"{path}.{key}", out)
    elif isinstance(left, list):
        if len(left) != len(right):
            out.append(f"{path}: length {len(left)} != {len(right)}")
        for index, (a, b) in enumerate(zip(left, right)):
            _collect_mismatches(a, b, f"{path}[{index}]", out)
    elif left != right:
        out.append(f"{path}: {left!r} != {right!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨模态图文检索诊断工作台")
    parser.add_argument("workspace", help="包含 config.yaml/samples/commands.json 的目录")
    parser.add_argument("--out", default=None, help="报告输出目录（默认 <workspace>/reports）")
    args = parser.parse_args(argv)

    workspace = args.workspace
    out_dir = args.out or os.path.join(workspace, "reports")
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(workspace, "config.yaml")
    sample_root = os.path.join(workspace, "samples")
    command_path = os.path.join(workspace, "commands.json")
    injection_path = os.path.join(workspace, "injections.json")

    injections = load_injection_manifest(injection_path)

    # 增量路径先跑：保留原始样本上的真实替换轨迹与精确失效账本
    incremental_report = run_incremental(
        config_path, command_path, sample_root, out_dir,
        os.path.join(workspace, "snapshots_inc"),
    )

    # 全量路径在隔离副本上从空白重建，避免与增量路径的文件物化互相污染
    import shutil

    full_sample_root = os.path.join(workspace, "_full_samples")
    shutil.rmtree(full_sample_root, ignore_errors=True)
    shutil.copytree(sample_root, full_sample_root)
    full_report = run_full(
        config_path, command_path, full_sample_root,
        os.path.join(out_dir, "full_report.json"),
        os.path.join(workspace, "snapshots_full"),
    )
    check = equivalence_check(full_report, incremental_report)
    payload = {
        "injections": injections,
        "equivalence": check,
    }
    _write_json(os.path.join(out_dir, "equivalence.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if check["equivalent"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
