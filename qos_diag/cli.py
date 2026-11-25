from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .errors import QosDiagError, SpecError, VerifyError
from .model import bundle_digest, load_bundle


def _print_json(value: Any, *, stream: Any = None) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), file=stream or sys.stdout)


def _load_with_digest(spec_dir: str | Path) -> dict[str, Any]:
    bundle = load_bundle(spec_dir)
    bundle["spec_digest"] = bundle_digest(bundle)
    return bundle


def cmd_sample(args: argparse.Namespace) -> dict[str, Any]:
    from .samplegen import generate_all_samples

    paths = generate_all_samples(args.workspace, seed=args.seed)
    return {"ok": True, "samples": {key: str(value) for key, value in paths.items()}}


def cmd_validate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load_with_digest(args.spec_dir)
    return {
        "ok": True,
        "pipeline_count": len(bundle["pipelines"]),
        "shaper_count": len(bundle["shapers"]),
        "event_count": len(bundle["events"]),
        "spec_digest": bundle["spec_digest"],
    }


def cmd_diagnose(args: argparse.Namespace) -> dict[str, Any]:
    from .diagnosis import diagnose

    bundle = _load_with_digest(args.spec_dir)
    result = diagnose(bundle, forced_path=args.path)
    if args.output:
        atomic_write_json(args.output, result)
    return {
        "ok": True,
        "output": args.output,
        "active_path": result["path_selection"]["active_path"],
        "path_parity": result["path_parity"],
        "event_count": len(result["events"]),
        "selected_reason": result["path_selection"]["rule"],
    }


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    from .ledger import Ledger
    from .runtime import ShaperProcessManager, kill_stale_pid_files
    from .service import run_full_into_ledger

    bundle = _load_with_digest(args.spec_dir)
    ledger_path = Path(args.ledger)
    run_dir = Path(args.run_dir)
    kill_stale_pid_files(run_dir)
    ledger = Ledger.open(ledger_path, spec_digest=bundle["spec_digest"])
    try:
        with ShaperProcessManager(bundle, run_dir) as manager:
            outcome = run_full_into_ledger(
                bundle,
                ledger,
                manager,
                batch_size=args.batch_size,
                forced_path=args.path,
            )
        if not outcome["ok"]:
            return {"ok": False, "error": "run_failed", "details": outcome}
        if args.snapshot:
            atomic_write_json(args.snapshot, outcome["diagnosis"])
        return {
            "ok": True,
            "ledger": str(ledger_path),
            "ledger_replaced": ledger.replaced,
            "snapshot": args.snapshot,
            "events": outcome["events"],
            "calls": outcome["calls"],
            "governor": outcome["governor"],
        }
    finally:
        ledger.close()


def cmd_recover(args: argparse.Namespace) -> dict[str, Any]:
    from .ledger import Ledger
    from .runtime import ShaperProcessManager, kill_stale_pid_files
    from .service import recover_into_ledger
    from .verify import verify_ledger

    bundle = _load_with_digest(args.spec_dir)
    run_dir = Path(args.run_dir)
    kill_stale_pid_files(run_dir)
    ledger = Ledger.open(Path(args.ledger), spec_digest=bundle["spec_digest"])
    replacement = {"replaced": ledger.replaced, "reason": ledger.replacement_reason}
    try:
        with ShaperProcessManager(bundle, run_dir) as manager:
            recovery = recover_into_ledger(bundle, ledger, manager, forced_path=args.path)
    finally:
        ledger.close()
    verification = verify_ledger(args.spec_dir, args.ledger)
    return {"ok": bool(recovery["ok"] and verification["ok"]), "replacement": replacement, "recovery": recovery, "verification": verification}


def cmd_edit(args: argparse.Namespace) -> dict[str, Any]:
    from .ledger import Ledger
    from .runtime import ShaperProcessManager, kill_stale_pid_files
    from .service import commit_edit_revision
    from .verify import verify_ledger

    old_bundle = _load_with_digest(args.old_dir)
    new_bundle = _load_with_digest(args.new_dir)
    manifest_path = Path(args.new_dir) / "edit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    run_dir = Path(args.run_dir)
    kill_stale_pid_files(run_dir)
    ledger = Ledger.open(Path(args.ledger), spec_digest=new_bundle["spec_digest"])
    try:
        with ShaperProcessManager(new_bundle, run_dir) as manager:
            outcome = commit_edit_revision(
                old_bundle,
                new_bundle,
                ledger,
                manager,
                edit_manifest=manifest,
                forced_path=args.path,
            )
        if not outcome["ok"]:
            return {"ok": False, "error": "edit_recompute_failed", "details": outcome}
        if args.invalidation_out:
            atomic_write_json(args.invalidation_out, outcome["closure"])
    finally:
        ledger.close()
    verification = verify_ledger(args.new_dir, args.ledger)
    return {
        "ok": True,
        "revision": outcome["revision"],
        "closure": {
            "affected_event_seqs": outcome["closure"]["affected_event_seqs"],
            "affected_windows": outcome["closure"]["affected_windows"],
            "affected_stages": outcome["closure"]["affected_stage_windows"],
        },
        "unchanged_check": outcome["unchanged_check"],
        "verification": verification,
    }


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    from .verify import verify_ledger

    return verify_ledger(args.spec_dir, args.ledger, revision=args.revision)


def cmd_clean(args: argparse.Namespace) -> dict[str, Any]:
    from .drill import clean_generated_artifacts

    removed = clean_generated_artifacts(args.workspace)
    return {"ok": True, **removed}


def cmd_drill(args: argparse.Namespace) -> dict[str, Any]:
    from .drill import run_acceptance_drill

    report = run_acceptance_drill(args.workspace)
    if args.output:
        atomic_write_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qos-diag", description="Local QoS token-bucket-chain diagnostics.")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="generate deterministic JSON samples")
    sample.add_argument("--workspace", default=".")
    sample.add_argument("--seed", type=int, default=287)
    sample.set_defaults(func=cmd_sample)

    validate = sub.add_parser("validate", help="validate spec/event JSON")
    validate.add_argument("--spec-dir", required=True)
    validate.set_defaults(func=cmd_validate)

    diagnose = sub.add_parser("diagnose", help="replay deterministic diagnosis without subprocess calls")
    diagnose.add_argument("--spec-dir", required=True)
    diagnose.add_argument("--output")
    diagnose.add_argument("--path", choices=("auto", "event", "window"), default="auto")
    diagnose.set_defaults(func=cmd_diagnose)

    run = sub.add_parser("run", help="run real TCP shaper subprocesses and commit ledger")
    run.add_argument("--spec-dir", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--snapshot")
    run.add_argument("--run-dir", default="artifacts/run/cli")
    run.add_argument("--batch-size", type=int)
    run.add_argument("--path", choices=("auto", "event", "window"), default="auto")
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover", help="mark unknown batches and replay missing events")
    recover.add_argument("--spec-dir", required=True)
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--run-dir", default="artifacts/run/cli-recover")
    recover.add_argument("--path", choices=("auto", "event", "window"), default="auto")
    recover.set_defaults(func=cmd_recover)

    edit = sub.add_parser("edit", help="commit an edited-spec incremental revision")
    edit.add_argument("--old-dir", required=True)
    edit.add_argument("--new-dir", required=True)
    edit.add_argument("--ledger", required=True)
    edit.add_argument("--invalidation-out")
    edit.add_argument("--run-dir", default="artifacts/run/edit")
    edit.add_argument("--path", choices=("auto", "event", "window"), default="auto")
    edit.set_defaults(func=cmd_edit)

    verify = sub.add_parser("verify", help="independently replay and compare ledger")
    verify.add_argument("--spec-dir", required=True)
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--revision", type=int)
    verify.set_defaults(func=cmd_verify)

    clean = sub.add_parser("clean", help="kill recorded local shaper PIDs and remove generated artifacts")
    clean.add_argument("--workspace", default=".")
    clean.set_defaults(func=cmd_clean)

    drill = sub.add_parser("drill", help="run the full local acceptance drill")
    drill.add_argument("--workspace", default=".")
    drill.add_argument("--output", default="artifacts/reports/acceptance-drill.json")
    drill.set_defaults(func=cmd_drill)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except VerifyError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return 3
    except SpecError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return 2
    except QosDiagError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return 4
    if isinstance(result, dict) and result.get("ok") is False:
        _print_json(result, stream=sys.stderr)
        return 1
    _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
