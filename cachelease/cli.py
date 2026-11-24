from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .engine import Engine
from .errors import CacheLeaseError, HandshakeError, SpecError
from .ledger import Ledger
from .samplegen import generate_sample
from .spec import group_batches, load_config, load_events
from .verify import verify_incremental


EXIT_CODES = {
    "E_SPEC": 2,
    "E_HANDSHAKE": 3,
    "E_REPLY": 4,
    "E_DIVERGE": 5,
    "E_REPLAY": 6,
}


def _write_report(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(payload: dict[str, Any]) -> None:
    print(canonical_json(payload))


def _load(args: argparse.Namespace):
    config = load_config(args.config)
    events = load_events(args.events, config)
    return config, events


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    config, events = _load(args)
    batches = group_batches(events)
    node_summary = []
    for node in config.nodes:
        node_summary.append(
            {
                "node_id": node.node_id,
                "endpoint": f"{node.host}:{node.port}",
                "capacity_slots": node.capacity_slots,
                "lease_limit": node.lease_limit,
                "seed": node.seed,
                "quota_version": node.quota_version,
                "request_concurrency": node.request_concurrency,
            }
        )
    return {
        "ok": True,
        "config_digest": config.config_digest,
        "nodes": node_summary,
        "stages": [stage.__dict__ for stage in config.stages],
        "artifacts": [
            {
                "key": artifact.key,
                "stage": artifact.stage,
                "refs": list(artifact.refs),
                "shards": artifact.shards,
                "shard_bytes": artifact.shard_bytes,
            }
            for artifact in config.artifacts
        ],
        "events": len(events),
        "batches": len(batches),
        "first_seq": events[0].seq,
        "last_seq": events[-1].seq,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config, events = _load(args)
    ledger = Ledger(args.ledger, config)
    try:
        engine = Engine(config, events, ledger, args.work_dir)
        report = engine.run(
            until_seq=args.until_seq,
            resume=args.resume,
            crash_at_batch=args.crash_at_batch,
        )
        return report
    finally:
        ledger.close()


def _recover(args: argparse.Namespace) -> dict[str, Any]:
    config, events = _load(args)
    ledger = Ledger(args.ledger, config)
    try:
        engine = Engine(config, events, ledger, args.work_dir)
        return engine.recover()
    finally:
        ledger.close()


def _verify(args: argparse.Namespace) -> dict[str, Any]:
    config, events = _load(args)
    return verify_incremental(
        config,
        events,
        checkpoint_seq=args.checkpoint_seq,
        work_dir=args.work_dir,
    )


def _port_check(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    results = []
    for node in config.nodes:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        occupied = False
        try:
            probe.settimeout(0.1)
            occupied = probe.connect_ex((node.host, node.port)) == 0
        finally:
            probe.close()
        results.append(
            {
                "node_id": node.node_id,
                "endpoint": f"{node.host}:{node.port}",
                "available": not occupied,
            }
        )
    if not all(item["available"] for item in results):
        raise HandshakeError("one or more declared loopback ports are occupied", "port-check")
    return {"ok": True, "ports": results}


def _generate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = generate_sample(args.out, args.seed)
    return {"ok": True, "output": str(Path(args.out)), "manifest": manifest}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cachelease", description="Multi-node cache lease and capacity-watermark diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic sample files")
    generate.add_argument("--out", default="samples/sample")
    generate.add_argument("--seed", type=int, default=243)
    generate.set_defaults(func=_generate)

    validate = subparsers.add_parser("validate", help="validate declarations and NDJSON")
    validate.add_argument("--config", required=True)
    validate.add_argument("--events", required=True)
    validate.add_argument("--report")
    validate.set_defaults(func=_validate)

    run = subparsers.add_parser("run", help="run a fresh or resumed pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--events", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--work-dir", default=".cachelease-work")
    run.add_argument("--report")
    run.add_argument("--until-seq", type=int)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--crash-at-batch", type=int, help=argparse.SUPPRESS)
    run.set_defaults(func=_run)

    recover = subparsers.add_parser("recover", help="recover committed state and replay-validate a ledger")
    recover.add_argument("--config", required=True)
    recover.add_argument("--events", required=True)
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--work-dir", default=".cachelease-work-recover")
    recover.add_argument("--report")
    recover.set_defaults(func=_recover)

    verify = subparsers.add_parser("verify", help="compare incremental suffix advancement with full rebuild")
    verify.add_argument("--config", required=True)
    verify.add_argument("--events", required=True)
    verify.add_argument("--checkpoint-seq", required=True, type=int)
    verify.add_argument("--work-dir", default=".cachelease-verify")
    verify.add_argument("--report")
    verify.set_defaults(func=_verify)

    port_check = subparsers.add_parser("port-check", help="check declared loopback ports")
    port_check.add_argument("--config", required=True)
    port_check.add_argument("--report")
    port_check.set_defaults(func=_port_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
        _write_report(getattr(args, "report", None), result)
        _print_summary(result)
        return 0
    except CacheLeaseError as exc:
        payload = exc.to_dict()
        _write_report(getattr(args, "report", None), payload)
        print(canonical_json(payload))
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        print(canonical_json({"error": {"code": "E_INTERRUPTED", "message": "interrupted"}}))
        return 130
    except Exception as exc:
        payload = {"error": {"code": "E_INTERNAL", "message": str(exc), "type": type(exc).__name__}}
        _write_report(getattr(args, "report", None), payload)
        print(canonical_json(payload))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

