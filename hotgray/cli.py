"""Command line entry points for demos, recompute, replay and fault injection."""

import argparse
import json
import sys

from .errors import HotGrayError
from .gateway import Gateway
from .snapshot import authoritative_recompute, compare_snapshots, snapshot_digest
from .spec import load_declaration


def _print(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _gateway(args):
    return Gateway(args.declaration, args.db)


def cmd_validate(args):
    declaration = load_declaration(args.declaration)
    _print({"ok": True,
            "backends": len(declaration["backends"]),
            "rules": len(declaration["rules"]),
            "requests": len(declaration["request_stream"])})
    return 0


def cmd_run(args):
    gateway = _gateway(args)
    try:
        gateway.start()
        if args.hotswap_after:
            stream = gateway.declaration["request_stream"]
            for request in stream[:args.hotswap_after]:
                gateway.admit_request(request)
            gateway.flush_all()
            swaps = gateway.hotswap_gray_backends()
            for request in stream[args.hotswap_after:]:
                gateway.admit_request(request)
        else:
            for request in gateway.declaration["request_stream"]:
                gateway.admit_request(request)
        gateway.flush_all()
        event = gateway.rebuild_snapshot()
        _print({"ok": True, "hits": gateway.store.count_hits(),
                "hotswaps": swaps if args.hotswap_after else [],
                "snapshot": event["snapshot"]})
    finally:
        gateway.stop()
    return 0


def cmd_recompute(args):
    gateway = _gateway(args)
    try:
        declaration = load_declaration(args.declaration)
        recomputed = authoritative_recompute(declaration, gateway.store)
        snapshot = gateway.store.load_latest_snapshot()
        if snapshot is None:
            raise HotGrayError("no snapshot available")
        compare_snapshots(snapshot["body"], recomputed["groups"])
        digest = snapshot_digest(
            recomputed["groups"], snapshot["param_versions"],
            snapshot["ledger_seq_from"], snapshot["ledger_seq_to"],
        )
        if digest != snapshot["digest"]:
            raise HotGrayError("snapshot envelope digest mismatch")
        _print({"ok": True, "verdict": "byte_identical",
                "ledger_range": [snapshot["ledger_seq_from"],
                                 snapshot["ledger_seq_to"]]})
    finally:
        gateway.store.close()
    return 0


def cmd_replay(args):
    gateway = _gateway(args)
    try:
        recovery = gateway.recover()
        snapshot = gateway.verify_replay()
        _print({"ok": True, "recovery": recovery,
                "replay_verdict": "reproducible",
                "snapshot_id": snapshot["snapshot_id"],
                "digest": snapshot["digest"]})
    finally:
        gateway.stop()
    return 0


def cmd_inject(args):
    import sqlite3

    gateway = _gateway(args)
    try:
        gateway.start()
        for request in gateway.declaration["request_stream"]:
            gateway.admit_request(request)
        gateway.flush_all()
        gateway.rebuild_snapshot()
        conn = sqlite3.connect(args.db)
        if args.fault == "corrupt_reply":
            row = conn.execute(
                "SELECT ledger_seq, output_digest FROM hits ORDER BY ledger_seq "
                "LIMIT 1"
            ).fetchone()
            conn.execute(
                "UPDATE hits SET output_digest=? WHERE ledger_seq=?",
                ("0" * 64, row[0]),
            )
            conn.commit()
            target = SnapshotError
        elif args.fault == "missing_param_version":
            row = conn.execute(
                "SELECT backend_id, backend_seq FROM param_versions "
                "ORDER BY backend_seq DESC LIMIT 1"
            ).fetchone()
            conn.execute(
                "DELETE FROM param_versions WHERE backend_id=? AND backend_seq=?",
                row,
            )
            conn.commit()
            target = ReplayError
        elif args.fault == "ledger_digest":
            row = conn.execute(
                "SELECT ledger_seq FROM hits ORDER BY ledger_seq LIMIT 1"
            ).fetchone()
            conn.execute(
                "UPDATE hits SET content_digest=? WHERE ledger_seq=?",
                ("f" * 64, row[0]),
            )
            conn.commit()
            target = ReplayError
        conn.close()
        declaration = load_declaration(args.declaration)
        try:
            authoritative_recompute(declaration, gateway.store)
        except target as exc:
            _print({"injected": args.fault, "code": exc.code,
                    "field": exc.field, "message": str(exc)})
            return 0
        raise AssertionError(f"fault {args.fault} was not detected")
    finally:
        gateway.stop()
    return 0


def cmd_gui(args):
    try:
        import tkinter  # noqa: F401
        from .gui import launch_gui
    except ImportError as exc:
        print(f"Tkinter unavailable in this environment: {exc}", file=sys.stderr)
        print("On Windows use the official python.org installer (Tkinter bundled); "
              "on Debian/WSL install python3-tk.", file=sys.stderr)
        return 3
    launch_gui(args.declaration, args.db)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="hotgray")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="strictly validate a declaration file")
    p.add_argument("declaration")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run routing, hot swap and snapshot rebuild")
    p.add_argument("declaration")
    p.add_argument("--db", required=True)
    p.add_argument("--hotswap-after", type=int, default=0,
                   help="hot-swap gray backends after N admitted requests")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("recompute", help="authoritative replay and comparison")
    p.add_argument("declaration")
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_recompute)

    p = sub.add_parser("replay", help="crash recovery and replay verification")
    p.add_argument("declaration")
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("inject", help="inject one of the three fault classes")
    p.add_argument("declaration")
    p.add_argument("--db", required=True)
    p.add_argument("--fault", required=True,
                   choices=["corrupt_reply", "missing_param_version",
                            "ledger_digest"])
    p.set_defaults(func=cmd_inject)

    p = sub.add_parser("gui", help="open the Tkinter diagnostic view")
    p.add_argument("declaration")
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_gui)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except HotGrayError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
