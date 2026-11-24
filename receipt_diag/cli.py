"""Command line interface for the inference receipt diagnostics package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .engine import INJECTIONS, DiagnosticsEngine
from .errors import DiagnosticError
from .invalidation import compute_invalidation
from .ledger import Ledger
from .planning import plan_batches
from .samplegen import write_sample
from .specs import load_specs


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _add_sample(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-dir", required=True)


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    _add_sample(parser)
    parser.add_argument("--db", required=True)
    parser.add_argument("--artifacts", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="receipt-diag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("samples", help="generate fixed-seed samples")
    sample_parser.add_argument("--output", required=True)
    sample_parser.add_argument("--preset", choices=("basic", "changed"), default="basic")
    sample_parser.add_argument("--seed", type=int, default=20261125)
    sample_parser.add_argument("--port-base", type=int, default=43171)

    validate_parser = subparsers.add_parser("validate", help="validate all three JSON files")
    _add_sample(validate_parser)

    plan_parser = subparsers.add_parser("plan", help="print or write the deterministic plan")
    _add_sample(plan_parser)
    plan_parser.add_argument("--out")

    run_parser = subparsers.add_parser("run", help="run a fresh inference session")
    _add_runtime(run_parser)
    run_parser.add_argument("--inject", choices=INJECTIONS, default="none")
    run_parser.add_argument("--crash-after-commits", type=int)
    run_parser.add_argument("--receipt-format", choices=("r1", "r2", "r3"))

    status_parser = subparsers.add_parser("status", help="inspect ledger/recovery state")
    _add_sample(status_parser)
    status_parser.add_argument("--db", required=True)

    recover_parser = subparsers.add_parser("recover", help="recover a hard-terminated run")
    _add_runtime(recover_parser)
    recover_parser.add_argument("--receipt-format", choices=("r1", "r2", "r3"))

    replay_parser = subparsers.add_parser("replay", help="recompute and verify every batch")
    _add_runtime(replay_parser)
    replay_parser.add_argument("--receipt-format", choices=("r1", "r2", "r3"))

    invalidate_parser = subparsers.add_parser(
        "invalidate", help="recompute only batches invalidated by a changed sample"
    )
    invalidate_parser.add_argument("--old-sample-dir", required=True)
    invalidate_parser.add_argument("--new-sample-dir", required=True)
    invalidate_parser.add_argument("--db", required=True)
    invalidate_parser.add_argument("--artifacts", required=True)
    invalidate_parser.add_argument("--inject", choices=INJECTIONS, default="none")
    invalidate_parser.add_argument("--report")

    snapshot_export = subparsers.add_parser("snapshot-export", help="export a session snapshot")
    snapshot_export.add_argument("--db", required=True)
    snapshot_export.add_argument("--out", required=True)

    snapshot_restore = subparsers.add_parser(
        "snapshot-restore", help="version/digest checked whole-snapshot replacement"
    )
    snapshot_restore.add_argument("--db", required=True)
    snapshot_restore.add_argument("--snapshot", required=True)
    return parser


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "samples":
        return write_sample(
            args.output,
            preset=args.preset,
            seed=args.seed,
            port_base=args.port_base,
        )
    if args.command == "validate":
        spec = load_specs(args.sample_dir)
        plan = plan_batches(spec)
        return {
            "ok": True,
            "sample_dir": str(Path(args.sample_dir).resolve()),
            "request_count": len(spec.requests),
            "worker_count": len(spec.workers),
            "batch_count": len(plan.batches),
            "spec_fingerprint": spec.fingerprint(),
            "plan_fingerprint": plan.fingerprint,
        }
    if args.command == "plan":
        spec = load_specs(args.sample_dir)
        plan = plan_batches(spec)
        if args.out:
            atomic_write_json(args.out, plan.to_dict())
        return {
            "ok": True,
            "plan": plan.to_dict(),
            "written_to": str(Path(args.out).resolve()) if args.out else None,
        }
    if args.command == "run":
        spec = load_specs(args.sample_dir)
        with Ledger(args.db) as ledger:
            engine = DiagnosticsEngine(
                spec, args.artifacts, receipt_format=args.receipt_format
            )
            return engine.run(
                ledger,
                injection=args.inject,
                crash_after_commits=args.crash_after_commits,
            )
    if args.command == "status":
        spec = load_specs(args.sample_dir)
        with Ledger(args.db) as ledger:
            engine = DiagnosticsEngine(spec, Path(args.db).parent / "artifacts-status")
            return {
                "ok": True,
                "plan_fingerprint": engine.plan.fingerprint,
                "status": engine.recovery_status(ledger),
            }
    if args.command == "recover":
        spec = load_specs(args.sample_dir)
        with Ledger(args.db) as ledger:
            engine = DiagnosticsEngine(
                spec, args.artifacts, receipt_format=args.receipt_format
            )
            return engine.recover(ledger)
    if args.command == "replay":
        spec = load_specs(args.sample_dir)
        with Ledger(args.db) as ledger:
            engine = DiagnosticsEngine(
                spec, args.artifacts, receipt_format=args.receipt_format
            )
            return engine.replay(ledger)
    if args.command == "invalidate":
        old_spec = load_specs(args.old_sample_dir)
        new_spec = load_specs(args.new_sample_dir)
        old_plan = plan_batches(old_spec)
        new_plan = plan_batches(new_spec)
        invalidation = compute_invalidation(old_spec, new_spec, old_plan, new_plan)
        if args.report:
            atomic_write_json(args.report, invalidation.to_dict())
        with Ledger(args.db) as ledger:
            if ledger.plan_fingerprint and ledger.plan_fingerprint != old_plan.fingerprint:
                raise DiagnosticError(
                    "E_LEDGER",
                    "ledger.plan_fingerprint",
                    f"ledger has {ledger.plan_fingerprint!r}, expected {old_plan.fingerprint!r}",
                )
            engine = DiagnosticsEngine(new_spec, args.artifacts)
            return engine.apply_invalidation(
                ledger,
                old_spec,
                old_plan,
                invalidation,
                injection=args.inject,
            )
    if args.command == "snapshot-export":
        with Ledger(args.db) as ledger:
            snapshot = ledger.export_snapshot(args.out)
        return {
            "ok": True,
            "snapshot_path": str(Path(args.out).resolve()),
            "snapshot_version": snapshot["snapshot_version"],
            "content_digest": snapshot["content_digest"],
            "batch_ids": [
                batch["batch_id"] for batch in snapshot["batches"]
            ],
        }
    if args.command == "snapshot-restore":
        snapshot = Ledger.load_snapshot(args.snapshot)
        with Ledger(args.db) as ledger:
            replaced = ledger.restore_snapshot(snapshot)
            digest = ledger.content_digest
            version = ledger.snapshot_version
        return {
            "ok": True,
            "snapshot_path": str(Path(args.snapshot).resolve()),
            "replaced_batch_ids": list(replaced),
            "ledger_content_digest": digest,
            "ledger_snapshot_version": version,
        }
    raise DiagnosticError("E_CONFIG", "cli.command", f"unknown command {args.command!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
        _print(result)
        return 0
    except DiagnosticError as exc:
        _print(exc.as_dict())
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - preserves a stable CLI envelope
        _print(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "path": "cli",
                "detail": repr(exc),
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
