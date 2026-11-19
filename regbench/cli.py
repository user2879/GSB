"""命令行入口：run / check-config / replay / consistency-check / interactive。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from typing import Any

from .config import load_config
from .errors import (
    StageWriteError,
    E_CONFIG_MISMATCH,
    E_TASK_FAILED,
    ConfigMismatchError,
    FailureLog,
)
from .report import build_report, compare_reports
from .runner import StopController, run_training
from .session import InteractiveSession
from .snapshot import SnapshotStore


def _load_or_exit(config_path: str):
    config, failures = load_config(config_path)
    if config is None:
        print(json.dumps({"failures": [f.to_dict() for f in failures]}, ensure_ascii=False, indent=2))
        return None, failures, 2
    return config, failures, 0


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
    else:
        print(text)


def _build_failpoint(spec: str | None):
    if not spec:
        return None
    stage, epoch_token, task_marker = spec.rsplit(":", 2)
    epoch = int(epoch_token)

    def failpoint(stage_name: str, epoch_no: int, task_key: str) -> None:
        if stage_name == stage and epoch_no == epoch and task_marker in task_key:
            if stage_name == "snapshot_write":
                raise StageWriteError(
                    f"failpoint 触发快照写入失败: epoch={epoch_no} task={task_key}"
                )
            raise RuntimeError(
                f"failpoint 触发: stage={stage_name} epoch={epoch_no} task={task_key}"
            )

    return failpoint


def cmd_run(args: argparse.Namespace) -> int:
    config, config_failures, code = _load_or_exit(args.config)
    if config is None:
        return code
    if args.workers:
        from .config import ExperimentConfig
        from .session import _with_workers

        config = _with_workers(config, args.workers)
    if args.reset:
        SnapshotStore(args.snapshot_dir).reset()

    store = SnapshotStore(args.snapshot_dir)
    stop = StopController()

    def handler(signum, frame):  # noqa: ANN001
        stop.request_stop()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    failure_log = FailureLog()
    failure_log.extend(config_failures)
    try:
        outcome = run_training(
            config,
            store,
            failure_log=failure_log,
            stop_controller=stop,
            failpoint=_build_failpoint(args.failpoint),
            force_full=args.force_full,
        )
    except ConfigMismatchError as exc:
        payload = {
            "error": E_CONFIG_MISMATCH,
            "message": str(exc),
            "snapshot_dir": args.snapshot_dir,
        }
        _write_json(args.report, payload)
        return 3

    report = build_report(
        config,
        outcome,
        concurrency_label=f"workers={config.scheduler.max_workers}",
    )
    report["config_validation_failures"] = [f.to_dict() for f in config_failures]
    report["resume_info"] = (
        {
            "version": outcome.resume_info.version,
            "recovery_point": outcome.resume_info.recovery_point,
            "completed_tasks": outcome.resume_info.completed_tasks,
            "consumed_epochs": outcome.resume_info.consumed_epochs,
        }
        if outcome.resume_info
        else None
    )
    _write_json(args.report, report)
    has_hard_failure = any(
        record.code == E_TASK_FAILED for record in failure_log.records
    )
    return 1 if has_hard_failure else 0


def cmd_check_config(args: argparse.Namespace) -> int:
    _, failures, _ = _load_or_exit(args.config)
    payload = {
        "config": args.config,
        "failure_count": len(failures),
        "failures": [item.to_dict() for item in failures],
    }
    _write_json(args.report, payload)
    return 0 if not failures else 4


def cmd_replay(args: argparse.Namespace) -> int:
    config, config_failures, code = _load_or_exit(args.config)
    if config is None:
        return code
    if args.reset:
        SnapshotStore(args.snapshot_dir).reset()
    session = InteractiveSession(config, args.snapshot_dir)
    traces = session.replay_file(args.commands)
    run_info = session.run()
    outcome = run_info["outcome"]
    report = build_report(
        session.config,
        outcome,
        concurrency_label=f"workers={session.config.scheduler.max_workers}",
        command_trace=[trace.to_dict() for trace in traces],
        cache_stats=session.cache_stats(),
    )
    report["config_validation_failures"] = [f.to_dict() for f in config_failures]
    _write_json(args.report, report)
    return 0


def cmd_consistency(args: argparse.Namespace) -> int:
    from .config import ExperimentConfig
    from .session import _with_workers

    base_config, config_failures, code = _load_or_exit(args.config)
    if base_config is None:
        return code
    reports: dict[str, dict[str, Any]] = {}

    # 1) 串行一次性全量。
    serial_store = SnapshotStore(os.path.join(args.snapshot_dir, "serial"))
    serial_store.reset()
    serial_config = _with_workers(base_config, 1)
    serial_outcome = run_training(
        serial_config, serial_store, failure_log=FailureLog()
    )
    reports["serial"] = build_report(
        serial_config, serial_outcome, concurrency_label="workers=1"
    )

    # 2) 高并发一次性全量。
    parallel_workers = max(2, base_config.scheduler.max_workers)
    parallel_store = SnapshotStore(os.path.join(args.snapshot_dir, "parallel"))
    parallel_store.reset()
    parallel_config = _with_workers(base_config, parallel_workers)
    parallel_outcome = run_training(
        parallel_config, parallel_store, failure_log=FailureLog()
    )
    reports["parallel"] = build_report(
        parallel_config, parallel_outcome, concurrency_label=f"workers={parallel_workers}"
    )

    # 3) 串行运行到指定轮次中断，再恢复续跑。
    resumed_store = SnapshotStore(os.path.join(args.snapshot_dir, "resumed"))
    resumed_store.reset()

    first_task_key = reports["serial"]["tasks"][0]["task_key"]

    def failpoint(stage: str, epoch: int, task_key: str) -> None:
        if stage == "train" and epoch == args.interrupt_epoch and task_key == first_task_key:
            raise KeyboardInterrupt("simulated interrupt")

    stop = StopController()
    try:
        run_training(
            serial_config,
            resumed_store,
            failure_log=FailureLog(),
            stop_controller=stop,
            failpoint=failpoint,
        )
    except KeyboardInterrupt:
        pass
    resume_info = resumed_store.resume_or_raise(serial_config.config_hash)
    resumed_outcome = run_training(
        serial_config, resumed_store, failure_log=FailureLog()
    )
    reports["resumed"] = build_report(
        serial_config,
        resumed_outcome,
        concurrency_label="workers=1+resume",
    )
    reports["resumed"]["resume_point"] = (
        {
            "version": resume_info.version,
            "recovery_point": resume_info.recovery_point,
            "completed_tasks": resume_info.completed_tasks,
            "consumed_epochs": resume_info.consumed_epochs,
        }
        if resume_info
        else None
    )

    consistency = compare_reports(reports)
    for report in reports.values():
        report["concurrency_consistency"] = consistency
    payload = {
        "consistency": consistency,
        "reports": reports,
        "config_validation_failures": [f.to_dict() for f in config_failures],
    }
    _write_json(args.report, payload)
    return 0 if consistency["status"] == "identical" else 5


def cmd_interactive(args: argparse.Namespace) -> int:
    config, _, code = _load_or_exit(args.config)
    if config is None:
        return code
    session = InteractiveSession(config, args.snapshot_dir)
    print("regbench 交互会话，输入 help 查看命令，exit 退出。")
    for line in sys.stdin:
        line = line.strip()
        if line in ("exit", "quit"):
            break
        if line == "help":
            print(
                "switch <scheme>; set-penalty <scheme> <v>; set-ratio <scheme> <v>; "
                "set-lr <v>; set-workers <n>; status; show; run"
            )
            continue
        if line == "run":
            info = session.run()
            print(json.dumps({k: v for k, v in info.items() if k != "outcome"}, ensure_ascii=False))
            continue
        trace = session.execute_line(line)
        print(json.dumps(trace.to_dict(), ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regbench")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="执行训练并输出 JSON 报告")
    run.add_argument("--config", required=True)
    run.add_argument("--snapshot-dir", required=True)
    run.add_argument("--report")
    run.add_argument("--workers", type=int)
    run.add_argument("--reset", action="store_true")
    run.add_argument("--force-full", action="store_true")
    run.add_argument("--failpoint")
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check-config", help="仅校验配置并输出失败清单")
    check.add_argument("--config", required=True)
    check.add_argument("--report")
    check.set_defaults(func=cmd_check_config)

    replay = sub.add_parser("replay", help="回放交互命令流后执行训练")
    replay.add_argument("--config", required=True)
    replay.add_argument("--commands", required=True)
    replay.add_argument("--snapshot-dir", required=True)
    replay.add_argument("--report")
    replay.add_argument("--reset", action="store_true")
    replay.set_defaults(func=cmd_replay)

    consistency = sub.add_parser(
        "consistency-check", help="串行/并发/中断恢复三路结果逐字段比对"
    )
    consistency.add_argument("--config", required=True)
    consistency.add_argument("--snapshot-dir", required=True)
    consistency.add_argument("--report")
    consistency.add_argument("--interrupt-epoch", type=int, default=3)
    consistency.set_defaults(func=cmd_consistency)

    interactive = sub.add_parser("interactive", help="进入交互式会话")
    interactive.add_argument("--config", required=True)
    interactive.add_argument("--snapshot-dir", required=True)
    interactive.set_defaults(func=cmd_interactive)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
