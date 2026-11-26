from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .acceptance import run_acceptance
from .errors import GSBError, SchemaViolation, SnapshotUnavailable
from .edits import validate_command_stream
from .incremental import IncrementalEngine
from .pipeline import resume_from_snapshot_state, run_corpus
from .samplegen import SEED, generate_sample_dir
from .schema import load_jsonl, load_sample
from .snapshot import SnapshotStore


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsb", description="Deterministic negation-scope ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-sample", help="generate deterministic local samples")
    generate.add_argument("--dir", default="samples")
    generate.add_argument("--seed", type=int, default=SEED)

    validate = sub.add_parser("validate", help="load and validate a sample directory")
    validate.add_argument("--sample-dir", default="samples")

    run = sub.add_parser("run", help="run one corpus batch")
    run.add_argument("--sample-dir", default="samples")
    run.add_argument("--out", default="artifacts/run-report.json")
    run.add_argument("--budget", type=int, default=None)
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--no-scheduler", action="store_true")

    verify = sub.add_parser("verify", help="run the local acceptance checks")
    verify.add_argument("--sample-dir", default="samples")
    verify.add_argument("--artifact-dir", default="artifacts")

    edits = sub.add_parser("apply-edits", help="apply the deterministic edit stream with reuse ledger")
    edits.add_argument("--sample-dir", default="samples")
    edits.add_argument("--edits", default=None)
    edits.add_argument("--out", default="artifacts/edit-report.json")

    resume = sub.add_parser("resume-snapshot", help="recover only from a committed snapshot")
    resume.add_argument("--snapshot-dir", default="artifacts/snapshot-demo")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate-sample":
            path = generate_sample_dir(args.dir, args.seed)
            _print_json({"ok": True, "sample_dir": str(path), "seed": args.seed})
            return 0

        if args.command == "validate":
            bundle = load_sample(args.sample_dir)
            _print_json({
                "ok": True,
                "ticket_count": len(bundle["tickets"]),
                "vocab_count": len(bundle["vocab"]),
                "trigger_count": len(bundle["triggers"]),
                "conflict_count": len(bundle["conflicts"]),
                "channel_count": len(bundle["channels"]),
            })
            return 0

        if args.command == "run":
            bundle = load_sample(args.sample_dir)
            result = run_corpus(
                bundle,
                use_scheduler=not args.no_scheduler,
                budget_points=args.budget,
                max_workers=args.workers,
            )
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _print_json({
                "ok": True,
                "complete": result["complete"],
                "conclusion_count": len(result["conclusions"]),
                "pending_ticket_count": len(result["pending_tickets"]),
                "error_summary": result["error_summary"],
                "report": str(out),
            })
            return 0 if result["complete"] else 3

        if args.command == "verify":
            report = run_acceptance(args.sample_dir, args.artifact_dir)
            summary = {
                "ok": report["ok"],
                "checks": sorted(report["checks"]),
                "incremental_steps": len(report["checks"]["incremental_reuse"]["edit_stream_steps"]),
                "budget_matrix_rows": len(report["checks"]["budget_concurrency"]["matrix"]),
                "gear_source_counts": report["checks"]["error_propagation"]["gear_report"]["source_counts"],
                "artifact": str(Path(args.artifact_dir) / "acceptance-report.json"),
            }
            _print_json(summary)
            return 0

        if args.command == "apply-edits":
            bundle = load_sample(args.sample_dir)
            edit_path = Path(args.edits) if args.edits else Path(args.sample_dir) / "edits.jsonl"
            commands = load_jsonl(edit_path)
            validate_command_stream(commands)
            engine = IncrementalEngine(bundle)
            reports = [engine.apply(command) for command in commands]
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {"ok": True, "step_count": len(reports), "steps": reports}
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _print_json({
                "ok": True,
                "step_count": len(reports),
                "last_global_version": engine.bundle["config"]["global_version"],
                "report": str(out),
            })
            return 0

        if args.command == "resume-snapshot":
            store = SnapshotStore(args.snapshot_dir)
            state = store.load_current()
            result = resume_from_snapshot_state(state)
            _print_json({
                "ok": True,
                "recovered": True,
                "manifest_id": state["manifest"]["snapshot_id"],
                "stage": state["manifest"]["stage"],
                "conclusion_count": len(result["conclusions"]),
            })
            return 0

        return 2
    except SchemaViolation as exc:
        _print_json(exc.to_dict())
        return 2
    except SnapshotUnavailable as exc:
        payload = {"ok": False, "error_code": exc.code, "message": exc.message, "details": exc.details}
        _print_json(payload)
        return 4
    except GSBError as exc:
        _print_json(exc.to_dict())
        return 2


if __name__ == "__main__":
    sys.exit(main())
