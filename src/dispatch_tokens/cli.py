"""命令行入口：validate / run / verify / replay。"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .engine import SchedulerEngine
from .errors import DispatchError
from .ledger import Ledger
from .runner import (Runner, rebuild_from_scratch,
                     verify_incremental_vs_rebuild)
from .spec import load_spec


def _print(payload: dict) -> None:
    json.dump(payload, sys.stdout, sort_keys=True, ensure_ascii=False,
              indent=2)
    sys.stdout.write("\n")


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    _print({"ok": True, "nodes": len(spec.nodes),
            "shards": len(spec.shards),
            "revisions": len(spec.revisions),
            "events": len(spec.events)})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    runner = Runner(spec, args.ledger, with_workers=not args.offline)
    report = runner.run()
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2)
            + "\n", encoding="utf-8")
    verify_incremental_vs_rebuild(spec, report)
    _print({"ok": True, "report": args.report or "<stdout-only>",
            "shards": len(report["outcomes"]),
            "refill_records": len(report["refill_audit"]),
            "decisions": len(report["decisions"]),
            "rebuild_verify": "E_DIFF-free"})
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    verify_incremental_vs_rebuild(spec, report)
    _print({"ok": True, "rebuild_verify": "E_DIFF-free"})
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    ledger = Ledger(args.ledger, spec)
    replay_outcomes = rebuild_from_scratch(spec)
    ledger.verify_replay(replay_outcomes)
    _print({"ok": True, "replay": "E_REPLAY-free",
            "shards": len(replay_outcomes)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dispatch-tokens")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="强校验本地 JSON 规格")
    p_validate.add_argument("spec")
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="运行确定性调度与真实回环下发")
    p_run.add_argument("spec")
    p_run.add_argument("--ledger", required=True)
    p_run.add_argument("--report")
    p_run.add_argument("--offline", action="store_true",
                       help="不起节点子进程，仅跑确定性引擎")
    p_run.set_defaults(func=cmd_run)

    p_verify = sub.add_parser("verify", help="增量结果 vs 空白全量重建")
    p_verify.add_argument("spec")
    p_verify.add_argument("--report", required=True)
    p_verify.set_defaults(func=cmd_verify)

    p_replay = sub.add_parser("replay", help="账本恢复重放一致性校验")
    p_replay.add_argument("spec")
    p_replay.add_argument("--ledger", required=True)
    p_replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DispatchError as exc:
        json.dump(exc.to_dict(), sys.stdout, sort_keys=True,
                  ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
