"""命令行入口：generate / diagnose / replay / snapshot / verify-ledger。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import samples
from .errors import DiagError
from .logging_utils import StructuredLogger
from .pipeline import (persist_session, restore_session, run_diagnosis,
                       verify_ledger)
from .replay import replay_and_verify


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True))


def _cmd_generate(args: argparse.Namespace) -> int:
    result = samples.generate_workspace(args.outdir)
    _print(result)
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    logger = StructuredLogger(path=args.log) if args.log else None
    session = run_diagnosis(
        wire_paths=args.wire, event_paths=args.events,
        window_ms=args.window_ms, min_samples=args.min_samples,
        arbitration_rule=args.arbitration, logger=logger)
    payload: Dict[str, Any] = {
        "model": session.model.to_dict(),
        "topology": session.topology.to_dict(),
        "gap_conclusions": {f: c.to_dict()
                            for f, c in session.conclusions.items()},
        "rate_statistics": session.statistics,
    }
    if args.snapshot or args.ledger:
        persisted = persist_session(
            session, args.snapshot or Path(args.outdir) / "snapshot.json",
            args.ledger or Path(args.outdir) / "ledger.jsonl")
        payload["persisted"] = persisted
    _print(payload)
    if logger:
        logger.close()
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    logger = StructuredLogger(path=args.log) if args.log else None
    session = run_diagnosis(wire_paths=args.wire, event_paths=args.events,
                            window_ms=args.window_ms,
                            min_samples=args.min_samples,
                            arbitration_rule=args.arbitration,
                            logger=logger)
    frames_by_flow: Dict[str, List[Any]] = {}
    for flow, obs in session.model.flows.items():
        frames_by_flow[flow] = list(obs.frames)
    report = replay_and_verify(frames_by_flow, session.conclusions,
                               port=args.port, tamper=args.tamper)
    _print(report.to_dict())
    if logger:
        logger.close()
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    logger = StructuredLogger(path=args.log) if args.log else None
    session, report = restore_session(args.snapshot, logger=logger)
    payload = {"restore": report.to_dict(),
               "topology": session.topology.to_dict(),
               "gap_conclusions": {f: c.to_dict()
                                   for f, c in session.conclusions.items()}}
    if args.append_capture:
        session.append_capture(args.append_capture)
        payload["after_append"] = {
            "flows": len(session.model.flows),
            "gap_conclusions": {f: c.to_dict()
                                for f, c in session.conclusions.items()},
        }
    _print(payload)
    if logger:
        logger.close()
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    logger = StructuredLogger(path=args.log) if args.log else None
    results = verify_ledger(args.ledger, workspace=args.workspace,
                            logger=logger)
    _print({"verified_entries": results})
    if logger:
        logger.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="netdiag", description="本地丢包与网络故障诊断工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性样本")
    gen.add_argument("--outdir", default="output/sample")
    gen.set_defaults(func=_cmd_generate)

    diag = sub.add_parser("diagnose", help="解码/合并/归因/统计")
    diag.add_argument("--wire", nargs="*", default=[])
    diag.add_argument("--events", nargs="*", default=[])
    diag.add_argument("--window-ms", type=int, default=100)
    diag.add_argument("--min-samples", type=int, default=4)
    diag.add_argument("--arbitration", default="wire_preferred",
                      choices=["wire_preferred", "event_preferred"])
    diag.add_argument("--snapshot")
    diag.add_argument("--ledger")
    diag.add_argument("--log")
    diag.add_argument("--outdir", default="output")
    diag.set_defaults(func=_cmd_diagnose)

    rep = sub.add_parser("replay", help="回环 TCP 重放并校验结论")
    rep.add_argument("--wire", nargs="*", default=[])
    rep.add_argument("--events", nargs="*", default=[])
    rep.add_argument("--window-ms", type=int, default=100)
    rep.add_argument("--min-samples", type=int, default=4)
    rep.add_argument("--arbitration", default="wire_preferred")
    rep.add_argument("--port", type=int, default=41793)
    rep.add_argument("--tamper", action="store_true",
                     help="注入额外序号以演示 E_REPLAY")
    rep.add_argument("--log")
    rep.set_defaults(func=_cmd_replay)

    snap = sub.add_parser("snapshot", help="打开快照并整体替换恢复")
    snap.add_argument("snapshot")
    snap.add_argument("--append-capture")
    snap.add_argument("--log")
    snap.set_defaults(func=_cmd_snapshot)

    ver = sub.add_parser("verify-ledger", help="按账本重放历史结论")
    ver.add_argument("ledger")
    ver.add_argument("--workspace", default=".")
    ver.add_argument("--log")
    ver.set_defaults(func=_cmd_verify)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DiagError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
