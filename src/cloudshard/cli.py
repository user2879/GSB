from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .canonical import content_digest
from .demo_spec import build_demo_declaration
from .errors import CloudShardError, exit_code_for
from .ledger import Ledger
from .pipeline import check_batch_boundaries, execute_pipeline, replay_declared
from .recovery import recover_inflight
from .snapshot import atomic_write_json
from .spec import load_spec
from .verifier import verify_spec_file


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cloudshard", description="Local deterministic CloudShard platform")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate declaration JSON")
    validate.add_argument("--spec", required=True)

    init_sample = sub.add_parser("init-sample", help="write a deterministic local sample declaration")
    init_sample.add_argument("--out", required=True)
    init_sample.add_argument("--fixed-ports", action="store_true")

    run = sub.add_parser("run", help="start loopback nodes and run dispatch/revisions")
    run.add_argument("--spec", required=True)
    run.add_argument("--workdir", required=True)
    run.add_argument("--ledger")
    run.add_argument("--snapshot")
    run.add_argument("--no-second-revision", action="store_true")
    run.add_argument("--crash-before-commit")
    run.add_argument("--crash-marker")

    recover = sub.add_parser("recover", help="reclaim unknown dispatch occupancy from committed ledger")
    recover.add_argument("--ledger", required=True)

    verify = sub.add_parser("verify", help="independent ledger/spec replay verification")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--ledger", required=True)

    replay = sub.add_parser("replay", help="replay declared lease/adjudication events")
    replay.add_argument("--spec", required=True)
    replay.add_argument("--out")
    replay.add_argument("--batch-check", action="store_true")

    demo = sub.add_parser("demo", help="generate free-port sample and run full happy path")
    demo.add_argument("--workdir", default="artifacts/demo")

    acceptance = sub.add_parser("acceptance", help="run all local acceptance checks including SIGKILL recovery")
    acceptance.add_argument("--workdir", default="artifacts/acceptance")

    tk = sub.add_parser("tk", help="open optional Tkinter ledger/snapshot viewer")
    tk.add_argument("--ledger", required=True)
    tk.add_argument("--snapshot")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            spec = load_spec(args.spec)
            _print(
                {
                    "ok": True,
                    "code": "E_SPEC_OK",
                    "nodes": len(spec.nodes),
                    "shards": len(spec.shards),
                    "revisions": len(spec.revisions),
                    "events": len(spec.events),
                    "spec_digest": content_digest(spec.raw),
                }
            )
            return 0

        if args.command == "init-sample":
            declaration = build_demo_declaration(allocate_ports=not args.fixed_ports)
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            load_spec(out)
            _print({"ok": True, "out": str(out), "ports": [n["port"] for n in declaration["nodes"]]})
            return 0

        if args.command == "run":
            report = execute_pipeline(
                args.spec,
                args.workdir,
                ledger_path=args.ledger,
                snapshot_path=args.snapshot,
                run_second_revision=not args.no_second_revision,
                crash_before_commit=args.crash_before_commit,
                crash_marker=args.crash_marker,
            )
            _print({k: v for k, v in report.items() if k != "declared_ownership"})
            return 0

        if args.command == "recover":
            with Ledger(args.ledger) as ledger:
                marks = recover_inflight(ledger)
                digest = ledger.ledger_digest()
            _print({"ok": True, "recovery_marks": marks, "ledger_digest": digest})
            return 0

        if args.command == "verify":
            report = verify_spec_file(args.spec, args.ledger)
            _print(report)
            return 0

        if args.command == "replay":
            spec = load_spec(args.spec)
            result = replay_declared(spec, args.out)
            if args.batch_check:
                result["batch_boundaries"] = check_batch_boundaries(spec)
            if args.out:
                atomic_write_json(args.out, result)
            _print(
                {
                    "ok": True,
                    "ownership_records": len(result["ownership_sequence"]),
                    "decisions": len(result["decisions"]),
                    "out": args.out,
                    "batch_boundaries": result.get("batch_boundaries"),
                }
            )
            return 0

        if args.command == "demo":
            workdir = Path(args.workdir)
            if workdir.exists():
                shutil.rmtree(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            spec_path = workdir / "declaration.generated.json"
            declaration = build_demo_declaration()
            spec_path.write_text(json.dumps(declaration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report = execute_pipeline(spec_path, workdir)
            verify = verify_spec_file(spec_path, Path(report["ledger"]))
            batches = check_batch_boundaries(load_spec(spec_path))
            output = {"ok": True, "report": report, "verify": verify, "batch_boundaries": batches}
            atomic_write_json(workdir / "demo-result.json", output)
            _print({"ok": True, "workdir": str(workdir), "verify_ok": verify["ok"], "batch_all_equal": batches["all_equal"]})
            return 0

        if args.command == "acceptance":
            from .acceptance import run_acceptance

            result = run_acceptance(args.workdir)
            _print(result)
            return 0 if result.get("ok") else 1

        if args.command == "tk":
            from .tk_view import main as tk_main

            return tk_main(["--ledger", args.ledger] + (["--snapshot", args.snapshot] if args.snapshot else []))

        raise CloudShardError(f"unknown command {args.command!r}")
    except CloudShardError as exc:
        payload = exc.to_payload()
        _print(payload)
        return exit_code_for(exc)
    except KeyboardInterrupt:
        _print({"ok": False, "code": "E_INTERRUPTED"})
        return 130
    except Exception as exc:
        _print({"ok": False, "code": "E_INTERNAL", "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
