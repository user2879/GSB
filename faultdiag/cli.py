"""命令行入口: run / replay / ui, 以及三类异常注入。"""

from __future__ import annotations

import argparse
import json
import sys

from .errors import DiagError
from .replay import replay_verify
from .runner import RunConfig, run
from .spec import load_domains, load_events, load_instances
from .ledger import Ledger
from .structured_log import StructuredLog


def _add_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--instances", required=True)
    parser.add_argument("--domains", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--workdir", required=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="faultdiag",
                                     description="故障隔离与恢复诊断组件")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="执行故障事件流并下发隔离/恢复指令")
    _add_spec_args(p_run)
    p_run.add_argument("--batch-size", type=int, default=3)
    p_run.add_argument("--global-limit", type=int, default=4)
    p_run.add_argument("--domain-quota", type=int, default=2)
    p_run.add_argument("--base-port", type=int, default=24100)
    p_run.add_argument("--inject", choices=["event", "level", "snapshot"],
                       default=None, help="注入异常: 非法事件类型/等级越界/快照摘要不符")

    p_replay = sub.add_parser("replay", help="恢复后一致性重放校验")
    _add_spec_args(p_replay)

    p_ui = sub.add_parser("ui", help="Tkinter 诊断界面")
    p_ui.add_argument("--workdir", required=True)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "run":
            report = run(RunConfig(
                instances_path=args.instances, domains_path=args.domains,
                events_path=args.events, workdir=args.workdir,
                batch_size=args.batch_size, global_limit=args.global_limit,
                domain_quota=args.domain_quota, base_port=args.base_port,
                inject=args.inject))
        elif args.cmd == "replay":
            domains = load_domains(args.domains)
            instances = load_instances(args.instances, domains)
            events = load_events(args.events, instances)
            ledger = Ledger(f"{args.workdir}/ledger.db", {})
            try:
                report = replay_verify(ledger, instances, domains, events)
            finally:
                ledger.close()
            report = {"status": "ok", **report}
        else:
            from .ui import launch
            launch(args.workdir)
            return 0
    except DiagError as exc:
        try:
            StructuredLog(args.workdir).error("command failed", code=exc.code,
                                              details=exc.details)
        except OSError:
            pass
        print(json.dumps({"status": "error", **exc.to_dict()},
                         ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
