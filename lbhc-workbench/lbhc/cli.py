"""Command line interface.

  python -m lbhc.cli run     --spec S --ledger L [--rounds N] [--path P]
  python -m lbhc.cli verify  --spec S --ledger L
  python -m lbhc.cli ui      --ledger L
  python -m lbhc.cli gen-spec --out S
  python -m lbhc.cli inject  {spec-corrupt,port-occupied,ledger-digest} ...

Exit codes: 0 ok, 2 E_SPEC, 3 E_PORT, 4 E_LEDGER, 5 E_REPLAY.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import sys
import tempfile

from . import __version__
from .errors import (BenchError, EXIT_OK, EXIT_LEDGER, EXIT_PORT,
                     EXIT_REPLAY, EXIT_SPEC)
from .logutil import StructuredLogger


def _print_error(err):
    print(json.dumps(err.to_dict(), ensure_ascii=False), file=sys.stderr)


def _cmd_run(args):
    from .engine import run_engine
    logger = StructuredLogger(name="run")
    try:
        report = run_engine(args.spec, args.ledger, rounds=args.rounds,
                            path=args.path, report_path=args.report,
                            logger=logger)
    except BenchError as err:
        logger.emit("error", **err.to_dict())
        _print_error(err)
        return err.EXIT
    print(json.dumps({
        "status": "ok",
        "spec_version": report["spec_version"],
        "path": report["path"],
        "path_reason": report["path_params"]["reason"],
        "probes": report["replay"]["probes_replayed"],
        "transitions": report["replay"]["transitions_verified"],
        "initiated_sequence": report["initiated_sequence"],
        "peak_global_inflight": report["budget"]["peak_global_inflight"],
        "peak_pool_inflight": report["budget"]["peak_pool_inflight"],
        "stale_diff": report["stale_diff"],
        "report": args.report,
    }, ensure_ascii=False, indent=2))
    return EXIT_OK


def _cmd_verify(args):
    from .ledger import Ledger
    from .replay import replay_verify
    from .spec import load_spec
    logger = StructuredLogger(name="verify")
    try:
        spec = load_spec(args.spec)
        ledger = Ledger(args.ledger)
        try:
            result = replay_verify(ledger, spec)
        finally:
            ledger.close()
    except BenchError as err:
        logger.emit("error", **err.to_dict())
        _print_error(err)
        return err.EXIT
    print(json.dumps({"status": "replay_ok", **result}, indent=2))
    return EXIT_OK


def _cmd_ui(args):
    try:
        from .ui import run_ui
    except ImportError as exc:
        print(f"tkinter is unavailable on this interpreter: {exc}",
              file=sys.stderr)
        return 1
    return run_ui(args.ledger, spec_path=args.spec)


def _cmd_gen_spec(args):
    from .sample import sample_spec
    from .spec import write_spec
    write_spec(sample_spec(base_port=args.base_port), args.out)
    print(f"sample spec written to {args.out}")
    return EXIT_OK


def _cmd_inject(args):
    logger = StructuredLogger(name="inject")
    if args.kind == "spec-corrupt":
        broken = tempfile.mktemp(suffix=".json", prefix="lbhc-corrupt-")
        shutil.copyfile(args.spec, broken)
        with open(broken, "r+", encoding="utf-8") as fh:
            doc = json.load(fh)
            doc["pools"][0]["failure_threshold"] = 0  # invalid
            doc["pools"][0]["backends"][0]["host"] = "10.0.0.9"  # invalid
            fh.seek(0)
            json.dump(doc, fh, indent=2)
            fh.truncate()
        from .engine import run_engine
        try:
            run_engine(broken, args.ledger, rounds=1, logger=logger)
        except BenchError as err:
            logger.emit("anomaly", injection="spec-corrupt", **err.to_dict())
            _print_error(err)
            return err.EXIT
        return EXIT_OK

    if args.kind == "port-occupied":
        from .spec import load_spec
        spec = load_spec(args.spec)
        victim = None
        for pool in spec.pools:
            for b in pool.backends:
                if b.behavior != "refused":
                    victim = b
                    break
            if victim:
                break
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        blocker.bind((victim.host, victim.port))
        blocker.listen(1)
        from .engine import run_engine
        try:
            run_engine(args.spec, args.ledger, rounds=1, logger=logger)
        except BenchError as err:
            logger.emit("anomaly", injection="port-occupied",
                        **err.to_dict())
            _print_error(err)
            return err.EXIT
        finally:
            blocker.close()
        return EXIT_OK

    if args.kind == "ledger-digest":
        # tamper with a committed ledger event out-of-band, then verify:
        # the sha256 hash chain no longer matches the stored content
        db = sqlite3.connect(args.ledger)
        row = db.execute("SELECT MIN(id) FROM events").fetchone()
        if not row or row[0] is None:
            db.close()
            print("ledger has no events to tamper; run first",
                  file=sys.stderr)
            return 1
        db.execute("UPDATE events SET payload=payload||' ' WHERE id=?",
                   (row[0],))
        db.commit()
        db.close()
        from .ledger import Ledger
        from .replay import replay_verify
        from .spec import load_spec
        try:
            spec = load_spec(args.spec)
            ledger = Ledger(args.ledger)
            try:
                replay_verify(ledger, spec)
            finally:
                ledger.close()
        except BenchError as err:
            logger.emit("anomaly", injection="ledger-digest",
                        **err.to_dict())
            _print_error(err)
            return err.EXIT
        return EXIT_OK

    print(f"unknown injection kind {args.kind}", file=sys.stderr)
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lbhc",
                                 description="LB health-check workbench")
    ap.add_argument("--version", action="version",
                    version=f"lbhc {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run health-check rounds")
    p.add_argument("--spec", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--path", choices=("auto", "full", "stratified"),
                   default="auto")
    p.add_argument("--report", default=None)
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("verify", help="replay-verify a ledger")
    p.add_argument("--spec", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(fn=_cmd_verify)

    p = sub.add_parser("ui", help="open the Tkinter diagnostics UI")
    p.add_argument("--ledger", required=True)
    p.add_argument("--spec", default=None)
    p.set_defaults(fn=_cmd_ui)

    p = sub.add_parser("gen-spec", help="write the deterministic sample spec")
    p.add_argument("--out", required=True)
    p.add_argument("--base-port", type=int, default=18601)
    p.set_defaults(fn=_cmd_gen_spec)

    p = sub.add_parser("inject", help="inject a structured anomaly")
    p.add_argument("kind", choices=("spec-corrupt", "port-occupied",
                                    "ledger-digest"))
    p.add_argument("--spec", required=True)
    p.add_argument("--ledger", required=True)
    p.set_defaults(fn=_cmd_inject)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
