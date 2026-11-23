from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .demo import DATA, default_spec, print_report, run_demo
from .diagnostics import environment_report
from .engine import GatewayEngine, ModelBank
from .errors import LabError
from .gateway import GatewayTCPServer, TcpBackendTransport, WorkerManager
from .ledger import Ledger
from .replay import authoritative_recompute, replay_remaining
from .snapshot import SnapshotStore, verify_snapshot_document
from .specs import load_spec, spec_to_json


def _add_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backends", default=str(DATA / "backends.json"))
    parser.add_argument("--rules", default=str(DATA / "rules.json"))
    parser.add_argument("--samples", default=str(DATA / "samples.ndjson"))


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_doctor(args: argparse.Namespace) -> int:
    spec = None
    if args.spec:
        spec = default_spec()
    report = environment_report(spec)
    _print_json(report)
    if not report["tkinter"].get("available"):
        print("WARN  tkinter import failed; core/CLI/demo remain usable, GUI requires OS Tk runtime.", file=sys.stderr)
    if not report["loopback_bind"].get("ok"):
        return 1
    if any(not item.get("bindable") for item in report["declared_backend_ports"]):
        return 1
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.backends, args.rules, args.samples)
    payload = spec_to_json(spec)
    payload["ok"] = True
    _print_json(payload)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    report = run_demo(args.out, args.scale_exp)
    print_report(report, args.out)
    return 0 if all(report["acceptance"].values()) else 1


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch_gui
    except Exception as exc:
        print(f"E_RUNTIME: cannot import GUI: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return launch_gui(args.workdir)


def cmd_replay(args: argparse.Namespace) -> int:
    ledger = Ledger(args.db)
    try:
        ledger_info = ledger.verify_integrity()
        store = SnapshotStore(args.snapshot)
        snapshot_payload = store.read_valid()["payload"]
        authoritative = authoritative_recompute(ledger, snapshot_payload)
        if args.checkpoint:
            with Path(args.checkpoint).open("r", encoding="utf-8") as handle:
                checkpoint_document = json.load(handle)
            verify_snapshot_document(checkpoint_document)
            recovery_payload = checkpoint_document["payload"]
            recovery_source = "checkpoint"
        else:
            recovery_payload = snapshot_payload
            recovery_source = "snapshot"
        recovery = replay_remaining(ledger, recovery_payload)
        _print_json(
            {
                "ok": True,
                "ledger": ledger_info,
                "authoritative": {
                    "completed_batches": authoritative["completed_batches"],
                    "ledger_content_digest": authoritative["ledger_content_digest"],
                },
                "recovery": {
                    "source": recovery_source,
                    "resumed_after_batch_index": recovery["resumed_after_batch_index"],
                    "replayed_batches": recovery["replayed_batches"],
                    "batch_ids": recovery["batch_ids"],
                },
            }
        )
        return 0
    finally:
        ledger.close()


def cmd_serve(args: argparse.Namespace) -> int:
    spec = load_spec(args.backends, args.rules, args.samples)
    if any(item.port == args.gateway_port for item in spec.backends):
        raise LabError("E_SPEC", "gateway port collides with a backend port", field="gateway_port")
    db = Path(args.db)
    snapshot = Path(args.snapshot)
    manager = WorkerManager(spec.backends)
    transport: TcpBackendTransport | None = None
    ledger: Ledger | None = None
    server: GatewayTCPServer | None = None
    try:
        manager.start()
        bank = ModelBank(spec)
        transport = TcpBackendTransport(bank)
        ledger = Ledger(db)
        engine = GatewayEngine(
            spec,
            ledger,
            scale_exp=args.scale_exp,
            snapshot_path=snapshot,
            transport=transport,
        )
        try:
            SnapshotStore(snapshot).load_or_replace(engine.manifest(), engine.payload())
        except LabError as exc:
            if exc.code != "E_SNAPSHOT":
                raise
            print(
                canonical_json(
                    {
                        "warning": "E_SNAPSHOT",
                        "message": exc.message,
                        "details": exc.details,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        server = GatewayTCPServer(engine, host="127.0.0.1", port=args.gateway_port)
        print(
            canonical_json(
                {
                    "ready": True,
                    "python": sys.version.replace("\n", " "),
                    "sqlite3": __import__("sqlite3").sqlite_version,
                    "numpy": __import__("numpy").__version__,
                    "gateway": f"127.0.0.1:{server.port}",
                    "ledger": str(db),
                    "snapshot": str(snapshot),
                    "backends": [
                        {"backend_id": item.backend_id, "port": item.port} for item in spec.backends
                    ],
                    "protocol": "JSONL; send samples, then {\"op\":\"flush\"}",
                }
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.2)
        return 0
    except KeyboardInterrupt:
        print("stopping", file=sys.stderr)
        return 0
    finally:
        if server is not None:
            server.server_close()
        if transport is not None:
            transport.close()
        if ledger is not None:
            ledger.close()
        manager.stop()


def cmd_snapshot(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.snapshot)
    document = store.read_valid()
    _print_json(
        {
            "ok": True,
            "manifest": document["manifest"],
            "content_digest": document["content_digest"],
            "progress": document["payload"].get("progress", {}),
            "rule_stats": document["payload"].get("rule_stats", {}),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gray-merge-lab",
        description="Local Windows deterministic gray-release receipt merge diagnostics lab",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="print versions and loopback binding checks")
    doctor.add_argument("--spec", action="store_true", help="also probe declared backend ports")
    doctor.set_defaults(func=cmd_doctor)

    validate = sub.add_parser("validate", help="strongly validate declarations and samples")
    _add_spec_args(validate)
    validate.set_defaults(func=cmd_validate)

    demo = sub.add_parser("demo", help="run the complete loopback demonstration")
    demo.add_argument("--out", default="artifacts/demo")
    demo.add_argument("--scale-exp", type=int, default=3)
    demo.set_defaults(func=cmd_demo)

    gui = sub.add_parser("gui", help="launch the Tkinter diagnostics console")
    gui.add_argument("--workdir", default="artifacts/app")
    gui.set_defaults(func=cmd_gui)

    replay = sub.add_parser("replay", help="verify ledger integrity then recompute/replay")
    replay.add_argument("--db", default="artifacts/demo/ledger.sqlite3")
    replay.add_argument("--snapshot", default="artifacts/demo/snapshot.json")
    replay.add_argument("--checkpoint", help="optional checkpoint document for remaining-batch recovery")
    replay.set_defaults(func=cmd_replay)

    serve = sub.add_parser("serve", help="start local backend workers and the JSONL gateway")
    _add_spec_args(serve)
    serve.add_argument("--gateway-port", type=int, default=28190)
    serve.add_argument("--scale-exp", type=int, default=3)
    serve.add_argument("--db", default="artifacts/serve/ledger.sqlite3")
    serve.add_argument("--snapshot", default="artifacts/serve/snapshot.json")
    serve.set_defaults(func=cmd_serve)

    snapshot = sub.add_parser("snapshot", help="show and verify a snapshot")
    snapshot.add_argument("--snapshot", default="artifacts/demo/snapshot.json")
    snapshot.set_defaults(func=cmd_snapshot)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LabError as exc:
        _print_json(exc.to_dict())
        return 2
    except (OSError, ValueError) as exc:
        _print_json({"error": "E_IO", "message": f"{type(exc).__name__}: {exc}"})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
