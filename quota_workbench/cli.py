"""Command line interface.

Subcommands:
  run              validate, orchestrate instances, probe, arbitrate, persist
  recover          post-kill recovery over the committed ledger
  replay           re-execute probing + arbitration and verify the ledger
  snapshot-verify  compare a snapshot against the current declaration
  gui              Tkinter diagnostics interface

Fault injection (run only, input/parameter-only):
  --inject port-occupied     hold one declared port so the child bind fails -> E_PORT
  --inject quota-overflow    bump one tenant share so the sum exceeds the cap -> E_QUOTA
  --inject snapshot-corrupt  flip bytes in the latest snapshot after writing -> E_SNAPSHOT
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

from .errors import WorkbenchError
from .slog import StructuredLogger
from . import ledger as ledger_mod
from . import runner as runner_mod
from . import snapshot as snapshot_mod
from .spec import load_spec, load_requests


def _emit_error(err: WorkbenchError, logger: StructuredLogger) -> int:
    payload = err.to_dict()
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
    logger.log("error", code=err.code, message=err.message,
               field=err.field, line=err.line)
    return 2


def _open_logger(args) -> StructuredLogger:
    return StructuredLogger(getattr(args, "log", None))


def cmd_run(args) -> int:
    logger = _open_logger(args)
    try:
        raw_override = None
        if args.inject == "quota-overflow":
            with open(args.spec, "r", encoding="utf-8") as fh:
                raw_override = json.load(fh)
            raw_override["tenants"][0]["share"] = (
                raw_override["total_share_cap"]
                - sum(t["share"] for t in raw_override["tenants"][1:])
                + 1
            )
            logger.log("inject", kind="quota-overflow",
                       tenant=raw_override["tenants"][0]["tenant_id"])
        spec = load_spec(args.spec, _raw_override=raw_override)
        requests = load_requests(args.requests, spec)
        logger.log("spec_loaded", tenants=len(spec.tenants),
                   instances=len(spec.instances), requests=len(requests))

        holder = None
        if args.inject == "port-occupied":
            victim = spec.instances[sorted(spec.instances)[0]]
            holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            holder.bind((victim.host, victim.port))
            holder.listen(1)
            logger.log("inject", kind="port-occupied",
                       instance=victim.instance_id, port=victim.port)
        try:
            outcome = runner_mod.execute_run(
                spec, requests, args.db, args.snapshot_dir,
                probe_rounds=args.rounds, batch_size=args.batch_size,
                crash_after_batch=args.crash_after_batch, logger=logger)
        finally:
            if holder is not None:
                holder.close()

        if args.inject == "snapshot-corrupt":
            path = os.path.join(args.snapshot_dir, "latest.json")
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            first_tenant = sorted(data["quota_final"])[0]
            data["quota_final"][first_tenant]["consumed"] += 1
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            logger.log("inject", kind="snapshot-corrupt", snapshot=path)
            try:
                snapshot_mod.verify_snapshot(path, spec, outcome["content_digest"], logger)
            except WorkbenchError as err:
                return _emit_error(err, logger)

        print(json.dumps({
            "status": "ok",
            "requests": len(outcome["result"].finalize_order),
            "content_digest": outcome["content_digest"],
            "snapshot": outcome["snapshot_path"],
        }, sort_keys=True))
        return 0
    except WorkbenchError as err:
        return _emit_error(err, logger)
    finally:
        logger.close()


def cmd_recover(args) -> int:
    logger = _open_logger(args)
    try:
        runner_mod.cleanup_orphans(args.db, logger)
        report = ledger_mod.recover(args.db, logger)
        report["note"] = (
            "in-flight uncommitted requests marked unknown and must be re-run; "
            "no exit report is produced for the killed run"
        )
        print(json.dumps(report, sort_keys=True))
        return 0
    except WorkbenchError as err:
        return _emit_error(err, logger)
    finally:
        logger.close()


def cmd_replay(args) -> int:
    logger = _open_logger(args)
    try:
        spec = load_spec(args.spec)
        requests = load_requests(args.requests, spec)
        report = runner_mod.replay_verify(spec, requests, args.db, args.rounds, logger)
        print(json.dumps({"status": "replay_ok", **report}, sort_keys=True))
        return 0
    except WorkbenchError as err:
        return _emit_error(err, logger)
    finally:
        logger.close()


def cmd_snapshot_verify(args) -> int:
    logger = _open_logger(args)
    try:
        spec = load_spec(args.spec)
        content_digest = None
        if os.path.exists(args.db):
            content_digest = ledger_mod.load_committed(args.db)["meta"].get("content_digest")
        try:
            report = snapshot_mod.verify_snapshot(args.snapshot, spec, content_digest, logger)
        except WorkbenchError as err:
            if args.replace:
                replaced = snapshot_mod.replace_all(args.db, args.snapshot_dir, logger)
                print(json.dumps({"status": "replaced",
                                  "diffs": err.details["diffs"],
                                  "removed": replaced["removed"]}, sort_keys=True))
            return _emit_error(err, logger)
        print(json.dumps({"status": "snapshot_ok", **report}, sort_keys=True))
        return 0
    except WorkbenchError as err:
        return _emit_error(err, logger)
    finally:
        logger.close()


def cmd_gui(args) -> int:
    from .gui import run_gui
    run_gui(args.db, args.snapshot_dir)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quota_workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--log", default=os.path.join("work", "run.log"))

    p = sub.add_parser("run")
    p.add_argument("--spec", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--db", default=os.path.join("work", "ledger.db"))
    p.add_argument("--snapshot-dir", default=os.path.join("work", "snapshots"))
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--crash-after-batch", type=int, default=None)
    p.add_argument("--inject", choices=["port-occupied", "quota-overflow", "snapshot-corrupt"])
    common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("recover")
    p.add_argument("--db", default=os.path.join("work", "ledger.db"))
    common(p)
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("replay")
    p.add_argument("--spec", required=True)
    p.add_argument("--requests", required=True)
    p.add_argument("--db", default=os.path.join("work", "ledger.db"))
    p.add_argument("--rounds", type=int, default=6)
    common(p)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("snapshot-verify")
    p.add_argument("--spec", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--db", default=os.path.join("work", "ledger.db"))
    p.add_argument("--snapshot-dir", default=os.path.join("work", "snapshots"))
    p.add_argument("--replace", action="store_true",
                   help="perform whole-replacement recovery on mismatch")
    common(p)
    p.set_defaults(func=cmd_snapshot_verify)

    p = sub.add_parser("gui")
    p.add_argument("--db", default=os.path.join("work", "ledger.db"))
    p.add_argument("--snapshot-dir", default=os.path.join("work", "snapshots"))
    common(p)
    p.set_defaults(func=cmd_gui)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
