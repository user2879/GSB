"""Command line entry point for the repeatable local demo."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from .chaos import run_kill_recovery
from .engine import Engine
from .errors import ShardMeshError
from .generator import demo_spec_dict, generate_samples, write_demo_files
from .ledger import Ledger
from .replay import replay_ledger
from .snapshot import load_or_rebuild_snapshot, write_snapshot
from .spec import load_spec, parse_spec
from .verification import verify_spec
from .view import build_engine_view, build_ledger_view


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reset_sqlite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


def cmd_init(args: argparse.Namespace) -> int:
    paths = write_demo_files(args.dir)
    print(json.dumps({"ok": True, "generated": {key: str(value) for key, value in paths.items()}}, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    print(json.dumps({"ok": True, "shards": len(spec.shards), "nodes": len(spec.nodes), "events": len(spec.events), "topology": spec.topology}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    raw_spec = _read_json(Path(args.spec))
    if getattr(args, "dynamic_ports", False):
        from .chaos import free_port

        for node, port in zip(raw_spec["nodes"], [free_port() for _ in raw_spec["nodes"]]):
            node["port"] = port
    spec = parse_spec(raw_spec, args.spec)
    samples_path = Path(args.samples)
    if samples_path.exists():
        samples = _read_json(samples_path)
    else:
        samples = generate_samples(raw_spec)
    ledger_path = Path(args.dir) / "ledger.sqlite3"
    if args.reset:
        _reset_sqlite(ledger_path)
    ledger = Ledger(ledger_path)
    ledger.initialize_spec(spec.to_dict())
    engine = Engine(spec, samples, ledger, work_dir=Path(args.dir) / "runtime")
    try:
        with engine:
            records = engine.run_events(spec.events)
            view = build_engine_view(engine)
        view["snapshot_version"] = ledger.increment_snapshot_version()
        snapshot_path = Path(args.dir) / "current-view.json"
        snapshot = write_snapshot(snapshot_path, view)
        loaded, backup = load_or_rebuild_snapshot(snapshot_path, ledger, lambda: build_ledger_view(ledger))
        replay_report = replay_ledger(ledger, samples)
        report = {
            "ok": True,
            "committed": len(records),
            "topology": spec.topology,
            "invalidation_plans": engine.invalidation_plans,
            "capacity_revocations": engine.capacity_revocations,
            "replay": replay_report,
            "snapshot": str(snapshot_path),
            "snapshot_digest": snapshot["content_digest"],
            "snapshot_rebuilt_from_backup": str(backup) if backup else None,
            "loaded_snapshot_digest_matches": loaded["content_digest"] == snapshot["content_digest"],
        }
        (Path(args.dir) / "run-report.json").write_text(json.dumps(report, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(report, sort_keys=True, indent=2, default=str))
    finally:
        ledger.close()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    if args.spec:
        spec = load_spec(args.spec)
        raw = _read_json(Path(args.spec))
        samples_path = Path(args.samples) if args.samples else Path(args.spec).with_name("samples.json")
        samples = _read_json(samples_path) if samples_path.exists() else generate_samples(raw)
    else:
        raw = demo_spec_dict()
        if args.dynamic_ports:
            from .chaos import free_port

            raw = demo_spec_dict(ports=[free_port(), free_port(), free_port()])
        spec = parse_spec(raw)
        samples = generate_samples(raw)
    report = verify_spec(spec, samples, args.dir)
    compact = {
        "consistent": report["consistent"],
        "code": report["code"],
        "byte_check": report["byte_check"],
        "invalidation_count": len(report["invalidation_plans"]),
        "report_path": str(Path(args.dir) / "verification-report.json"),
    }
    print(json.dumps(compact, sort_keys=True, indent=2))
    return 0


def cmd_chaos(args: argparse.Namespace) -> int:
    report = run_kill_recovery(args.dir)
    print(json.dumps({key: value for key, value in report.items() if key != "post_rerun_replay"}, sort_keys=True, indent=2, default=str))
    return 0 if report["consistent"] else 3


def cmd_replay(args: argparse.Namespace) -> int:
    samples = _read_json(Path(args.samples))
    ledger = Ledger(args.ledger)
    try:
        report = replay_ledger(ledger, samples)
        print(json.dumps(report, sort_keys=True, indent=2))
    finally:
        ledger.close()
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import run_tk_app

    return run_tk_app(args.spec, args.ledger)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shardmesh", description="Deterministic shard mesh demo")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="generate spec.json and samples.json")
    init.add_argument("--dir", type=Path, default=Path("artifacts/demo"))
    init.set_defaults(func=cmd_init)

    validate = sub.add_parser("validate", help="strongly validate a JSON spec")
    validate.add_argument("--spec", type=Path, default=Path("artifacts/demo/spec.json"))
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run the real subprocess/scheduler demo")
    run.add_argument("--spec", type=Path, default=Path("artifacts/demo/spec.json"))
    run.add_argument("--samples", type=Path, default=Path("artifacts/demo/samples.json"))
    run.add_argument("--dir", type=Path, default=Path("artifacts/demo"))
    run.add_argument("--reset", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--dynamic-ports", action="store_true", help="replace ports in memory with free loopback ports")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="incremental vs blank full rebuild")
    verify.add_argument("--dir", type=Path, default=Path("artifacts/verification"))
    verify.add_argument("--spec", type=Path)
    verify.add_argument("--samples", type=Path)
    verify.add_argument("--dynamic-ports", action="store_true")
    verify.set_defaults(func=cmd_verify)

    chaos = sub.add_parser("chaos", help="SIGKILL recovery and replay")
    chaos.add_argument("--dir", type=Path, default=Path("artifacts/chaos"))
    chaos.set_defaults(func=cmd_chaos)

    replay = sub.add_parser("replay", help="replay and verify a ledger")
    replay.add_argument("--ledger", type=Path, required=True)
    replay.add_argument("--samples", type=Path, required=True)
    replay.set_defaults(func=cmd_replay)

    gui = sub.add_parser("gui", help="open the optional Tkinter current view")
    gui.add_argument("--spec", type=Path, default=Path("artifacts/demo/spec.json"))
    gui.add_argument("--ledger", type=Path, default=Path("artifacts/demo/ledger.sqlite3"))
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ShardMeshError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), flush=True)
        return 2
    except Exception as exc:
        print(json.dumps({"ok": False, "code": "E_INTERNAL", "message": str(exc), "traceback": traceback.format_exc()}, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
