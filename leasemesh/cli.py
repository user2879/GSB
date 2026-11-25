from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from .cluster import NodeCluster, allocate_ports, list_residual_python_nodes
from .engine import Engine
from .errors import MeshError
from .evolution import propagate_invalidation
from .ownership import adjudicate_events
from .snapshot import reconcile_snapshot
from .spec import load_declaration, load_events, load_requests
from .store import Ledger
from .verify import verify_ledger


EXIT_CODES = {
    "E_SPEC": 2,
    "E_PORT": 10,
    "E_TIMEOUT": 11,
    "E_FIELD": 12,
    "E_LEASE": 13,
    "E_STATE": 14,
    "E_LIMIT": 15,
    "E_VERIFY": 16,
    "E_IO": 1,
}


def print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False))


def _write_output(path: Optional[str], payload: Any) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _reset_ledger_files(ledger_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(ledger_path) + suffix)
        if path.exists():
            path.unlink()


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_declaration(args.declarations)
    result: dict[str, Any] = {
        "ok": True,
        "code": "OK",
        "declaration": args.declarations,
        "declaration_digest": spec["declaration_digest"],
        "nodes": len(spec["nodes"]),
        "slots": len(spec["slots"]),
    }
    if args.requests:
        requests = load_requests(args.requests, spec)
        result["requests"] = len(requests)
    if args.events:
        events = load_events(args.events)
        result["events"] = len(events)
    return result


def cmd_allocate_ports(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_declaration(args.declarations)
    runtime = allocate_ports(spec, args.out)
    return {"ok": True, "code": "OK", "out": args.out, "ports": [node["port"] for node in runtime["nodes"]]}


def cmd_check_ports(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_declaration(args.declarations)
    residual = list_residual_python_nodes(spec)
    if residual:
        raise MeshError(
            "one or more declared loopback ports are occupied by a residual process",
            code="E_PORT",
            details={"residual": residual},
        )
    return {
        "ok": True,
        "code": "OK",
        "residual": [],
        "ports": [{"node_id": node["node_id"], "port": node["port"], "state": "free"} for node in spec["nodes"]],
    }


def cmd_adjudicate(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_declaration(args.declarations) if args.declarations else {}
    events = load_events(args.events)
    result = adjudicate_events(events, spec)
    _write_output(args.out, result)
    return {
        "ok": True,
        "code": "OK",
        "out": args.out,
        "owners": len(result["owners"]),
        "decisions": len(result["decisions"]),
        "reclamation_totals": result["reclamation_totals"],
        "capacity": result["capacity"],
    }


def cmd_execute(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_declaration(args.declarations)
    if args.allocate_ports:
        runtime_path = Path(args.workdir) / "allocated-declarations.json"
        spec = allocate_ports(spec, runtime_path)
        spec["_path"] = str(runtime_path)
    requests = load_requests(args.requests, spec)
    ledger_path = Path(args.ledger)
    if args.reset:
        _reset_ledger_files(ledger_path)
    ledger = Ledger(ledger_path, spec)
    cluster = NodeCluster(spec, args.workdir)
    try:
        cluster.start()
        engine = Engine(
            spec,
            ledger,
            cluster,
            crash_before_commit_request=args.crash_before_commit,
        )
        results = engine.execute(requests)
        snapshot_result = reconcile_snapshot(args.snapshot, spec, ledger) if args.snapshot else None
        return {
            "ok": all(item.get("ok") for item in results),
            "code": "OK" if all(item.get("ok") for item in results) else "PARTIAL_FAILURE",
            "ledger": str(ledger_path),
            "results": results,
            "snapshot": snapshot_result,
            "scheduler_peak": results[0].get("scheduler_peak") if results else None,
        }
    finally:
        cluster.stop()
        ledger.close()


def cmd_recover(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_declaration(args.declarations)
    ledger = Ledger(args.ledger, spec)
    try:
        recovery = ledger.recover_pending()
        if args.rerun_manifest:
            _write_output(args.rerun_manifest, {"requests": recovery["rerun_requests"]})
        snapshot = reconcile_snapshot(args.snapshot, spec, ledger) if args.snapshot else None
        return {
            "ok": True,
            "code": "OK",
            "ledger": args.ledger,
            "recovery": recovery,
            "rerun_manifest": args.rerun_manifest,
            "snapshot": snapshot,
        }
    finally:
        ledger.close()


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    return verify_ledger(args.ledger, args.declarations)


def _events_from_ledger(ledger_path: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    ledger = Ledger(ledger_path, spec, readonly=True)
    try:
        rows = ledger.conn.execute(
            "SELECT payload_json FROM commits WHERE kind IN ('lease_result','domain_event') ORDER BY seq"
        ).fetchall()
        events = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            if payload.get("event"):
                events.append(payload["event"])
        return events
    finally:
        ledger.close()


def cmd_evolve(args: argparse.Namespace) -> dict[str, Any]:
    old_spec = load_declaration(args.old)
    new_spec = load_declaration(args.new)
    if args.events:
        events = load_events(args.events)
    elif args.ledger:
        events = _events_from_ledger(args.ledger, old_spec)
    else:
        raise MeshError("evolve requires --events or --ledger", code="E_SPEC", field="events")
    report = propagate_invalidation(old_spec, new_spec, events)
    _write_output(args.out, report)
    return {
        "ok": True,
        "code": "OK",
        "out": args.out,
        "invalid_set": report["invalid_set"],
        "trace_steps": len(report["trace"]),
        "all_unaffected_bytes_identical": report["all_unaffected_bytes_identical"],
    }


def cmd_inject_spec(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source)
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(source.read_text(encoding="utf-8"))
    if args.kind == "bad_interval":
        data["renew_interval_ms"] = int(data["lease_duration_ms"]) + 5
    elif args.kind == "negative_concurrency":
        data["global_concurrency"] = 0
    elif args.kind == "duplicate_port":
        data["nodes"][1]["port"] = data["nodes"][0]["port"]
    elif args.kind == "bad_event_seq":
        events_path = Path(args.events_source)
        events = json.loads(events_path.read_text(encoding="utf-8"))
        events["events"][-1]["seq"] = events["events"][0]["seq"]
        target.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise MeshError(f"unknown injection kind {args.kind}", code="E_SPEC", field="kind")
    if args.kind != "bad_event_seq":
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        if args.kind == "bad_event_seq":
            from .jsonio import load_json_document
            from .spec import validate_events

            validate_events(load_json_document(target))
        else:
            load_declaration(target)
    except MeshError as exc:
        result = {"injected": args.kind, "target": str(target), **exc.to_dict()}
        _write_output(args.out, result)
        return result
    raise MeshError("injected corruption was unexpectedly accepted", code="E_VERIFY")


def cmd_inject_stale(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_declaration(args.declarations)
    if args.allocate_ports:
        allocated = Path(args.workdir) / "stale-declarations.json"
        spec = allocate_ports(spec, allocated)
        spec["_path"] = str(allocated)
    cluster = NodeCluster(spec, args.workdir)
    try:
        cluster.start()
        lease_id = "lease-stale-injection"
        grant = cluster.request(
            spec["nodes"][0]["node_id"],
            {"op": "grant", "request_id": "stale-grant", "lease_id": lease_id, "slot_id": spec["slots"][0]},
        )
        renewed = cluster.request(
            spec["nodes"][0]["node_id"],
            {
                "op": "renew",
                "request_id": "stale-renew-current",
                "lease_id": lease_id,
                "generation": 1,
                "label": f"{lease_id}@1",
            },
        )
        try:
            cluster.request(
                spec["nodes"][0]["node_id"],
                {
                    "op": "renew",
                    "request_id": "stale-renew-old",
                    "lease_id": lease_id,
                    "generation": 1,
                    "label": f"{lease_id}@1",
                },
            )
        except MeshError as exc:
            return {
                "injected": "stale_generation_renewal",
                "ok": False,
                **exc.to_dict(),
                "grant_generation": grant.get("generation"),
                "current_generation": renewed.get("generation"),
                "node_did_not_advance": exc.details.get("generation") == renewed.get("generation"),
            }
        raise MeshError("stale renewal was unexpectedly accepted", code="E_VERIFY")
    finally:
        cluster.stop()


def cmd_tk(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from .tkui import run_tk

        return run_tk(args)
    except Exception as exc:
        if isinstance(exc, MeshError):
            raise
        raise MeshError(
            "Tkinter is unavailable; install system package with: sudo apt-get install -y python3-tk",
            code="E_IO",
            details={"import_error": str(exc)},
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leasemesh", description="Shared lease/replica-segment mesh")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--declarations", required=True)
    validate.add_argument("--requests")
    validate.add_argument("--events")
    validate.set_defaults(func=cmd_validate)

    ports = sub.add_parser("allocate-ports")
    ports.add_argument("--declarations", required=True)
    ports.add_argument("--out", required=True)
    ports.set_defaults(func=cmd_allocate_ports)

    check_ports = sub.add_parser("check-ports")
    check_ports.add_argument("--declarations", required=True)
    check_ports.set_defaults(func=cmd_check_ports)

    adjudicate = sub.add_parser("adjudicate")
    adjudicate.add_argument("--events", required=True)
    adjudicate.add_argument("--declarations")
    adjudicate.add_argument("--out")
    adjudicate.set_defaults(func=cmd_adjudicate)

    execute = sub.add_parser("execute")
    execute.add_argument("--declarations", required=True)
    execute.add_argument("--requests", required=True)
    execute.add_argument("--workdir", default="work")
    execute.add_argument("--ledger", default="work/ledger.sqlite3")
    execute.add_argument("--snapshot", default="work/snapshot.json")
    execute.add_argument("--reset", action="store_true")
    execute.add_argument("--allocate-ports", action="store_true")
    execute.add_argument("--crash-before-commit")
    execute.set_defaults(func=cmd_execute)

    recover = sub.add_parser("recover")
    recover.add_argument("--declarations", required=True)
    recover.add_argument("--ledger", default="work/ledger.sqlite3")
    recover.add_argument("--snapshot")
    recover.add_argument("--rerun-manifest")
    recover.set_defaults(func=cmd_recover)

    verify = sub.add_parser("verify")
    verify.add_argument("--declarations", required=True)
    verify.add_argument("--ledger", required=True)
    verify.set_defaults(func=cmd_verify)

    evolve = sub.add_parser("evolve")
    evolve.add_argument("--old", required=True)
    evolve.add_argument("--new", required=True)
    evolve.add_argument("--events")
    evolve.add_argument("--ledger")
    evolve.add_argument("--out", required=True)
    evolve.set_defaults(func=cmd_evolve)

    inject = sub.add_parser("inject-spec")
    inject.add_argument("--source", default="samples/declarations.json")
    inject.add_argument("--events-source", default="samples/events.json")
    inject.add_argument("--target", default="work/injected/broken.json")
    inject.add_argument("--kind", choices=["bad_interval", "negative_concurrency", "duplicate_port", "bad_event_seq"], default="bad_interval")
    inject.add_argument("--out")
    inject.set_defaults(func=cmd_inject_spec)

    stale = sub.add_parser("inject-stale")
    stale.add_argument("--declarations", default="samples/declarations.json")
    stale.add_argument("--workdir", default="work/inject-stale")
    stale.add_argument("--allocate-ports", action="store_true")
    stale.set_defaults(func=cmd_inject_stale)

    tk = sub.add_parser("tk")
    tk.set_defaults(func=cmd_tk)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        print_json(result)
        if isinstance(result, dict) and result.get("ok") is False and result.get("code") in EXIT_CODES:
            return EXIT_CODES[result["code"]]
        if isinstance(result, dict) and result.get("ok") is False:
            return 1
        return 0
    except MeshError as exc:
        print_json(exc.to_dict())
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        print_json({"ok": False, "code": "E_IO", "error": "interrupted"})
        return 130
    except Exception as exc:
        print_json(
            {
                "ok": False,
                "code": "E_IO",
                "error": str(exc),
                "type": type(exc).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
