"""工作台命令行入口。"""
from __future__ import annotations

import argparse
import copy
import os

from . import injection as inj
from .commands import load_commands
from .config import load_config
from .errors import FailureLog
from .pipeline import Pipeline
from .pairs import load_pairs
from .report import write_html, write_json
from .samplegen import write_workspace
from .snapshot import SnapshotStore


def _load_all(pairs_path: str, config_path: str, failures: FailureLog):
    cfg = load_config(config_path, failures)
    if cfg is None:
        return None, {}
    pairs = load_pairs(pairs_path, cfg.normalize, failures)
    return cfg, pairs


def cmd_init(args) -> int:
    paths = write_workspace(args.root)
    print("已生成样本工作区：")
    for name, path in paths.items():
        print(f"  - {name}: {os.path.relpath(path, args.root)}")
    return 0


def _snapshot_paths(root: str):
    return (
        os.path.join(root, "runtime", "snapshots"),
        os.path.join(root, "runtime", "tmp"),
    )


def cmd_run(args) -> int:
    failures = FailureLog()
    cfg, pairs = _load_all(args.pairs, args.config, failures)
    if cfg is None:
        print("配置非法，已记录 E_CONFIG_INVALID，终止全量运行。")
        _dump_failures(failures, args.report)
        return 2

    snap_root, tmp_root = _snapshot_paths(args.root)
    store = SnapshotStore(snap_root, tmp_root) if args.snapshot else None
    pipe = Pipeline(cfg, pairs, snapshot_store=store, parallel=not args.serial)
    pipe.run_all_batches()
    # 合并加载阶段失败
    for entry in failures.entries():
        pipe.report.failures.add(entry)

    payload = _build_payload(pipe)
    _emit_reports(args.report, payload)
    _print_summary(pipe)
    return 0


def cmd_replay(args) -> int:
    failures = FailureLog()
    cfg, pairs = _load_all(args.pairs, args.config, failures)
    if cfg is None:
        print("配置非法，终止命令流回放。")
        return 2
    commands = load_commands(args.commands, failures)

    snap_root, tmp_root = _snapshot_paths(args.root)
    store = SnapshotStore(snap_root, tmp_root) if args.snapshot else None
    pipe = Pipeline(cfg, pairs, snapshot_store=store, parallel=not args.serial)
    pipe.run_all_batches()  # 初始基线
    for entry in failures.entries():
        pipe.report.failures.add(entry)

    step_payloads = []
    for command in commands:
        pipe.apply_and_run(command)
        step_payloads.append({
            "seq": command.seq, "op": command.op,
            "canonical_results": pipe.report.canonical_results(pipe.cache),
        })

    # ---- 全量重建：从空白缓存对“最终状态”一次性计算 ----
    rebuild = Pipeline(pipe.cfg, pipe.pairs, parallel=not args.serial, full_compute=True)
    rebuild.run_all_batches()
    final_incremental = pipe.report.data_artifacts(pipe.cache)
    final_rebuild = rebuild.report.data_artifacts(rebuild.cache)
    artifacts_equal = final_incremental == final_rebuild
    # 输入类失败（E_PAIR_INVALID/E_CONFIG_INVALID/E_PAIR_NOT_FOUND/E_COMMAND_INVALID）
    # 必须一致；预算耗尽 E_TRAIN_BUDGET_EXHAUSTED 是增量过程中的瞬时运维事件，
    # 一次性全量重建不会经历，故作为运维历史单独呈现而不纳入数据等价判定。
    operational = {"E_TRAIN_BUDGET_EXHAUSTED", "E_SNAPSHOT_STALE"}
    def input_failures(rep):
        return [f for f in rep["failures"] if f["code"] not in operational]
    final_failures_equal = (
        input_failures(pipe.report.canonical_results(pipe.cache))
        == input_failures(rebuild.report.canonical_results(rebuild.cache))
    )
    equivalent = artifacts_equal and final_failures_equal

    payload = _build_payload(pipe)
    payload["equivalence"] = {
        "incremental_equals_full_rebuild": equivalent,
        "data_artifacts_equal": artifacts_equal,
        "final_failures_equal": final_failures_equal,
        "steps": len(commands),
        "note": "数据产物逐字段一致；预算账本/失效事件为有状态运维历史，单独列示",
    }
    payload["replay_steps"] = step_payloads
    payload["full_rebuild_results"] = rebuild.report.canonical_results(rebuild.cache)
    _emit_reports(args.report, payload)
    print(f"增量 vs 全量重建逐字段一致：{equivalent}")
    _print_summary(pipe)
    return 0 if equivalent else 3


def cmd_inject(args) -> int:
    failures = FailureLog()
    cfg, pairs = _load_all(args.pairs, args.config, failures)
    config_results = inj.run_config_injections(args.config)
    command_result = inj.run_command_injection(pairs, cfg) if cfg else None
    all_results = config_results + ([command_result] if command_result else [])
    all_hit = all(r["hit"] for r in all_results)
    payload = {"injection_report": all_results, "all_hit": all_hit}
    write_json(os.path.join(args.report, "injection_report.json"), payload)
    for r in all_results:
        print(f"{r['injection_id']} {r['kind']}: hit={r['hit']} "
              f"expected={r['expected_code']} observed={r['observed_codes']}")
    return 0 if all_hit else 4


def cmd_verify(args) -> int:
    """串行 vs 并发逐字段一致性核验。"""
    failures = FailureLog()
    cfg, pairs = _load_all(args.pairs, args.config, failures)
    if cfg is None:
        return 2
    p_parallel = Pipeline(cfg, copy.deepcopy(pairs), parallel=True)
    p_parallel.run_all_batches()
    p_serial = Pipeline(cfg, copy.deepcopy(pairs), parallel=False)
    p_serial.run_all_batches()
    same = (p_parallel.report.canonical_results(p_parallel.cache)
            == p_serial.report.canonical_results(p_serial.cache))
    print(f"并发 vs 串行逐字段一致：{same}（max_workers={cfg.max_workers}）")
    return 0 if same else 5


def _build_payload(pipe: Pipeline) -> dict:
    return {
        "canonical_results": pipe.report.canonical_results(pipe.cache),
        "failures": pipe.report.failures.to_list(),
        "invalidation_events": pipe.report.invalidation_events,
        "reuse_events": pipe.report.reuse_events,
        "batch_plans": pipe.report.batch_plans,
        "boundary_list": pipe.report.boundary_list,
        "score_breakdown": pipe.report.score_breakdown,
        "summaries": pipe.report.summaries,
        "budget_ledger": pipe.report.budget_ledger,
        "config": pipe.cfg.to_dict(),
    }


def _emit_reports(report_dir: str, payload: dict) -> None:
    os.makedirs(report_dir, exist_ok=True)
    write_json(os.path.join(report_dir, "diagnostics.json"), payload)
    write_html(os.path.join(report_dir, "diagnostics.html"), payload)


def _dump_failures(failures: FailureLog, report_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)
    write_json(os.path.join(report_dir, "diagnostics.json"),
               {"failures": failures.to_list()})


def _print_summary(pipe: Pipeline) -> None:
    ledger = pipe.report.budget_ledger
    print(f"处理对={len(pipe.cache.score.pair_index)} "
          f"预算消耗={ledger['consumed']}/{ledger['global_budget']} "
          f"剩余={ledger['remaining']} 待办={ledger['todo']}")
    for plan in pipe.report.batch_plans:
        print(f"  批次 {plan['batch_id']}: 处理={len(plan['processed'])} "
              f"新计费={plan['charged_count']} 待办={plan['deferred']} "
              f"阻断={plan['budget_blocked']}")
    if pipe.report.failures.entries():
        print("失败清单：")
        for e in pipe.report.failures.entries():
            print(f"  [{e.code}] {e.sort_key} {e.stage}: {e.message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reward-workbench",
        description="本地确定性 LLM 偏好对齐与奖励建模诊断工作台",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="生成固定种子样本工作区")
    p_init.add_argument("--root", default=".")
    p_init.set_defaults(func=cmd_init)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".")
    common.add_argument("--pairs", default="samples/preference_pairs.jsonl")
    common.add_argument("--config", default="config/reward_config.yaml")
    common.add_argument("--report", default="reports")
    common.add_argument("--serial", action="store_true", help="强制串行执行")
    common.add_argument("--snapshot", action="store_true", help="启用阶段化快照")

    p_run = sub.add_parser("run", parents=[common], help="一次性全量运行")
    p_run.set_defaults(func=cmd_run)

    p_replay = sub.add_parser("replay", parents=[common], help="回放命令流并校验一致性")
    p_replay.add_argument("--commands", default="commands/command_stream.json")
    p_replay.set_defaults(func=cmd_replay)

    p_inject = sub.add_parser("inject", parents=[common], help="异常注入演练")
    p_inject.set_defaults(func=cmd_inject)

    p_verify = sub.add_parser("verify", parents=[common], help="并发/串行一致性核验")
    p_verify.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
