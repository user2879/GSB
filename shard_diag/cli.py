from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .errors import DiagnosticError
from .impact import event_edit_impact
from .ledger import Ledger
from .spec import load_events, load_spec


def load_session_spec(args: argparse.Namespace):
    spec = load_spec(args.spec)
    scale = getattr(args, "scale", None)
    if scale:
        from dataclasses import replace

        runtime = replace(spec.runtime, fixed_point_scale=scale)
        spec = replace(spec, runtime=runtime)
    return spec


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, default=str))


def cmd_doctor(_args: argparse.Namespace) -> int:
    modules = ["socket", "threading", "sqlite3", "numpy"]
    report: dict[str, Any] = {"ok": True, "modules": {}}
    for name in modules:
        try:
            module = importlib.import_module(name)
            report["modules"][name] = {"ok": True, "version": getattr(module, "__version__", "stdlib")}
        except Exception as exc:
            report["ok"] = False
            report["modules"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import ctypes
        import os
        import sys

        root = Path(__file__).resolve().parent.parent
        local_lib = root / ".sysdeps" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
        if local_lib.exists():
            os.environ["LD_LIBRARY_PATH"] = str(local_lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
            for pattern in ("libXss.so*", "libXft.so*", "libtcl8.6.so*", "libtk8.6.so*"):
                for library in sorted(local_lib.glob(pattern)):
                    try:
                        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
        for extra in (
            "/usr/lib/python3/dist-packages",
            "/home/zsl/.feedback/python-base/lib/python3.11/site-packages",
        ):
            if Path(extra).exists() and extra not in sys.path:
                sys.path.append(extra)
        import tkinter

        report["modules"]["tkinter"] = {"ok": True, "version": getattr(tkinter, "TkVersion", "unknown")}
    except Exception as exc:
        report["ok"] = False
        report["modules"]["tkinter"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "install_hint": "sudo apt-get update && sudo apt-get install -y python3-tk tk8.6 libtk8.6",
        }
    import socket

    report["loopback"] = "127.0.0.1"
    report["sqlite"] = __import__("sqlite3").sqlite_version
    print_json(report)
    return 0 if report["ok"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    result: dict[str, Any] = {
        "ok": True,
        "spec_digest": spec.digest(),
        "nodes": len(spec.nodes),
        "shards": len(spec.shards),
        "global_budget": spec.global_budget,
    }
    if args.events:
        events = load_events(args.events, spec)
        result["events"] = len(events.events)
        result["events_digest"] = events.digest()
    print_json(result)
    return 0


def cmd_ports(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    from .protocol import ensure_port_available

    statuses = []
    for node in spec.nodes:
        ensure_port_available(node.port, node.node_id)
        statuses.append({"node_id": node.node_id, "host": node.host, "port": node.port, "available": True})
    print_json({"ok": True, "ports": statuses})
    return 0


def cmd_samples(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    out = Path(args.out)
    written = []
    for node in spec.nodes:
        for shard in spec.shards:
            path = out / f"{node.node_id}__{shard.shard_id}.json"
            from .determinism import write_sample_artifact

            write_sample_artifact(path, node, shard)
            written.append(str(path))
    print_json({"ok": True, "written": len(written), "dir": str(out)})
    return 0


def cmd_water(args: argparse.Namespace) -> int:
    spec = load_session_spec(args)
    from .watermark import global_budget_attribution, node_water_report, compare_fixed_water

    allocations = [(shard.shard_id, shard.resource_units) for shard in spec.shards]
    budget = global_budget_attribution(allocations, spec.global_budget, spec.runtime.fixed_point_scale)
    nodes = [
        node_water_report(
            node_id=node.node_id,
            active_shards=0,
            active_leases=0,
            concurrency_quota=node.concurrency_quota,
            max_leases=node.max_leases_per_node,
            scale=spec.runtime.fixed_point_scale,
        )
        for node in spec.nodes
    ]
    comparisons = []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            comparisons.append({
                "a": left["node_id"],
                "b": right["node_id"],
                "slot": compare_fixed_water(left["slot_water_q"], right["slot_water_q"]),
                "lease": compare_fixed_water(left["lease_water_q"], right["lease_water_q"]),
            })
    print_json({"ok": True, "global": budget.to_dict(), "nodes": nodes, "node_comparisons": comparisons})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_session_spec(args)
    events = load_events(args.events, spec) if args.events else None
    ledger = Ledger.open_session(args.ledger, spec.digest())
    report: dict[str, Any]
    try:
        from .engine import ExecutionEngine
        from .node_manager import NodeManager

        manager = NodeManager(spec, runtime_dir=args.runtime_dir)
        try:
            manager.start_all()
            engine = ExecutionEngine(spec, ledger, nodes=manager, samples_dir=args.samples_dir)
            report = engine.run_events(events or [])
        finally:
            manager.stop_all(force=True)
        report["ledger"] = str(args.ledger)
        report["ledger_replaced"] = ledger.replaced
        report["ledger_previous"] = ledger.previous
        report["content_digest"] = ledger.update_root_digest_locked() if False else ledger.meta().get("content_digest")
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print_json({"ok": True, "summary": report["summary"], "ledger": args.ledger, "output": args.output})
        return 0
    finally:
        ledger.close()


def cmd_recover(args: argparse.Namespace) -> int:
    spec = load_session_spec(args)
    ledger = Ledger.open_session(args.ledger, spec.digest())
    try:
        from .engine import recover_incomplete_shards

        report = recover_incomplete_shards(ledger)
        print_json({"ok": True, **report})
        return 0
    finally:
        ledger.close()


def cmd_verify(args: argparse.Namespace) -> int:
    from .verifier import verify_files

    spec = load_session_spec(args)
    events = load_events(args.events, spec)
    from .verifier import verify_ledger

    report = verify_ledger(spec, events, args.ledger)
    print_json(report)
    return 0


def cmd_impact(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    old = load_events(args.old_events, spec) if args.old_events else None
    new = load_events(args.new_events, spec)
    print_json({"ok": True, "impact": event_edit_impact(spec, old, new)})
    return 0


def cmd_recompute(args: argparse.Namespace) -> int:
    spec = load_session_spec(args)
    old = load_events(args.old_events, spec) if args.old_events else None
    new = load_events(args.new_events, spec)
    from .incremental import recompute_edited_event_stream

    report = recompute_edited_event_stream(spec, new, args.ledger, old_events=old)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print_json({
        "ok": True,
        "first_changed_seq": report["impact"]["first_changed_seq"],
        "aggregate_recompute_shards": report["impact"]["aggregate_recompute_shards"],
        "cache_hits": len(report["aggregate_cache_hits"]),
        "outcome_rows_unchanged": report["outcome_rows_unchanged_by_incremental_pass"],
        "output": args.output,
    })
    return 0


def cmd_snapshot(args: argparse.Namespace) -> int:
    spec = load_session_spec(args)
    ledger = Ledger.open_session(args.ledger, spec.digest())
    try:
        destination = ledger.backup_to(args.out)
        print_json({"ok": True, "snapshot": str(destination)})
        return 0
    finally:
        ledger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shard-diag", description="Local shard rollback diagnostics and replay ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--spec", required=True)
    validate.add_argument("--events")
    validate.set_defaults(func=cmd_validate)

    ports = sub.add_parser("ports")
    ports.add_argument("--spec", required=True)
    ports.set_defaults(func=cmd_ports)

    samples = sub.add_parser("samples")
    samples.add_argument("--spec", required=True)
    samples.add_argument("--out", default="samples")
    samples.set_defaults(func=cmd_samples)

    water = sub.add_parser("water", help="print fixed-point error attribution report")
    water.add_argument("--spec", required=True)
    water.add_argument("--scale", type=int)
    water.set_defaults(func=cmd_water)

    run = sub.add_parser("run")
    run.add_argument("--spec", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--ledger", default=".runtime/ledger.sqlite")
    run.add_argument("--runtime-dir", default=".runtime")
    run.add_argument("--samples-dir", default="samples")
    run.add_argument("--output", default=".runtime/last-run.json")
    run.add_argument("--scale", type=int, help="session fixed-point scale; overrides runtime.fixed_point_scale")
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover")
    recover.add_argument("--spec", required=True)
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--scale", type=int)
    recover.set_defaults(func=cmd_recover)

    verify = sub.add_parser("verify")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--events", required=True)
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--scale", type=int)
    verify.set_defaults(func=cmd_verify)

    impact = sub.add_parser("impact")
    impact.add_argument("--spec", required=True)
    impact.add_argument("--old-events")
    impact.add_argument("--new-events", required=True)
    impact.set_defaults(func=cmd_impact)

    recompute = sub.add_parser("recompute", help="recompute only aggregates/water scopes affected by an event edit")
    recompute.add_argument("--spec", required=True)
    recompute.add_argument("--old-events")
    recompute.add_argument("--new-events", required=True)
    recompute.add_argument("--ledger", required=True)
    recompute.add_argument("--output", default=".runtime/incremental.json")
    recompute.add_argument("--scale", type=int)
    recompute.set_defaults(func=cmd_recompute)

    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--spec", required=True)
    snapshot.add_argument("--ledger", required=True)
    snapshot.add_argument("--out", required=True)
    snapshot.add_argument("--scale", type=int)
    snapshot.set_defaults(func=cmd_snapshot)

    doctor = sub.add_parser("doctor")
    doctor.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DiagnosticError as exc:
        print_json(exc.to_dict())
        return 3 if exc.code == "E_VERIFY" else 2
    except KeyboardInterrupt:
        print_json({"ok": False, "error": {"code": "E_INTERRUPTED", "message": "interrupted"}})
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
