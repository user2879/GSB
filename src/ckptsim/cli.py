from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import CkptError
from .impact import expected_batch_records, selective_recompute
from .ledger import Ledger, SnapshotIdentity
from .plan import build_event_plan
from .process import check_ports_free
from .runner import Runner, RunnerOptions, clean_runtime_artifacts
from .spec import load_bundle, load_layers
from .verify import verify_ledger


def _add_spec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shards", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--events", required=True)


def _load(args: argparse.Namespace):
    return load_bundle(args.shards, args.layers, args.checkpoints, args.events)


def _emit(payload: Any) -> None:
    print(canonical_json(payload))


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load(args)
    plan = build_event_plan(bundle)
    return {
        "ok": True,
        "content_digest": bundle.content_digest,
        "shard_set_version": bundle.shard_set_version,
        "parameter_dialect": bundle.parameter_dialect,
        "shards": [shard.shard_id for shard in bundle.shards.shards],
        "parameter_versions": [
            version.parameter_version for version in bundle.layers.versions
        ],
        "checkpoints": [checkpoint.checkpoint_id for checkpoint in bundle.checkpoints.checkpoints],
        "event_count": len(bundle.events.events),
        "plan": plan.to_dict(),
    }


def command_ports(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load(args)
    check_ports_free(bundle.shards.shards)
    return {
        "ok": True,
        "ports": {
            shard.shard_id: {"host": shard.host, "port": shard.port}
            for shard in bundle.shards.shards
        },
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load(args)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    removed = clean_runtime_artifacts(workdir) if args.fresh else []
    ledger_path = Path(args.ledger) if args.ledger else workdir / "ledger.sqlite"
    runner = Runner(
        bundle,
        RunnerOptions(
            ledger_path=ledger_path,
            lease_ttl_ms=args.lease_ttl_ms,
            client_timeout=args.client_timeout,
            startup_timeout=args.startup_timeout,
            auto_recover=not args.no_auto_recover,
            check_ports=not args.skip_port_check,
        ),
    )
    result = runner.run()
    result["removed_before_run"] = removed
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def command_impact(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load(args)
    edited = load_layers(args.edited_layers)
    plan = build_event_plan(bundle)
    baseline = expected_batch_records(bundle, plan)
    result = selective_recompute(bundle, edited, plan, baseline)
    if args.record_ledger:
        ledger = Ledger.open(
            args.record_ledger,
            identity=SnapshotIdentity.from_bundle(bundle),
            initial_versions=plan.initial_versions,
            initial_generation=0,
        )
        try:
            digest = ledger.record_impact(result["impact"])
            result["recorded_ledger"] = str(args.record_ledger)
            result["recorded_digest"] = digest
            result["snapshot"] = (
                ledger.snapshot_report.to_dict() if ledger.snapshot_report else None
            )
        finally:
            ledger.close()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return {"ok": True, **result}


def command_verify(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load(args)
    result = verify_ledger(
        bundle,
        args.ledger,
        edited_layers_path=args.edited_layers,
        raise_on_error=not args.no_raise,
    )
    return {"ok": result.verified, **result.to_dict()}


def command_recover(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load(args)
    plan = build_event_plan(bundle)
    ledger = Ledger.open(
        args.ledger,
        identity=SnapshotIdentity.from_bundle(bundle),
        initial_versions=plan.initial_versions,
        initial_generation=0,
    )
    try:
        recovered = ledger.recover_inflight(args.detail)
        return {
            "ok": True,
            "recovered": recovered,
            "snapshot": ledger.snapshot_report.to_dict() if ledger.snapshot_report else None,
            "batches": ledger.list_batches_by_seq(),
            "attempts": ledger.list_attempts(),
            "note": "Uncommitted work is marked unknown and must be replayed; no exit report is invented.",
        }
    finally:
        ledger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ckptsim",
        description="Deterministic local checkpoint, recovery and replay simulator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate all JSON declarations")
    _add_spec_args(validate)
    validate.set_defaults(handler=command_validate)

    ports = subparsers.add_parser("ports", help="probe configured loopback ports")
    _add_spec_args(ports)
    ports.set_defaults(handler=command_ports)

    run = subparsers.add_parser("run", help="run the TCP training simulation")
    _add_spec_args(run)
    run.add_argument("--workdir", default="artifacts/run")
    run.add_argument("--ledger")
    run.add_argument("--output")
    run.add_argument("--fresh", action="store_true")
    run.add_argument("--lease-ttl-ms", type=int, default=30_000)
    run.add_argument("--client-timeout", type=float, default=10.0)
    run.add_argument("--startup-timeout", type=float, default=8.0)
    run.add_argument("--no-auto-recover", action="store_true")
    run.add_argument("--skip-port-check", action="store_true")
    run.set_defaults(handler=command_run)

    impact = subparsers.add_parser("impact", help="compute dependency invalidation")
    _add_spec_args(impact)
    impact.add_argument("--edited-layers", required=True)
    impact.add_argument("--record-ledger")
    impact.add_argument("--output")
    impact.set_defaults(handler=command_impact)

    verify = subparsers.add_parser("verify", help="independently replay and verify a ledger")
    _add_spec_args(verify)
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--edited-layers")
    verify.add_argument("--no-raise", action="store_true")
    verify.set_defaults(handler=command_verify)

    recover = subparsers.add_parser("recover", help="recover in-flight ledger records")
    _add_spec_args(recover)
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--detail", default="operator recovery")
    recover.set_defaults(handler=command_recover)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.handler(args)
        _emit(payload)
        return 0 if payload.get("ok", True) else 1
    except CkptError as exc:
        _emit({"ok": False, "error": exc.to_dict()})
        return 2
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "error": {"code": "E_INTERNAL", "detail": f"{type(exc).__name__}: {exc}"},
            }
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
