from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .bundle import initial_state, load_bundle
from .errors import Codes
from .samples import generate_samples
from .session import InteractiveSession, canonical_ledger, field_report, render_final_state
from .snapshot import SnapshotStore, restore_payload_into_engine
from .engine import RenderEngine
from .validation import write_failure_list


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False))


def cmd_generate(args: argparse.Namespace) -> int:
    result = generate_samples(args.root)
    print(f"generated deterministic samples in {Path(args.root).resolve()}")
    print(f"sample files: {len(result['files'])} binary/image files")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    bundle, failures = load_bundle(root)
    output = Path(args.output)
    payload = write_failure_list(failures, output, extra={
        "suite": "valid-sample-bundle",
        "root": str(root.resolve()),
    })
    print(f"validation failures: {payload['count']}; report: {output}")
    if failures:
        for failure in failures:
            print(failure.code, failure.path, failure.message)
        return 2
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    root = Path(args.root)
    bundle, failures = load_bundle(root)
    if failures:
        write_failure_list(failures, Path(args.report).parent / "load_failures.json")
        for failure in failures:
            print(failure.code, failure.message, file=sys.stderr)
        return 2

    store = SnapshotStore(args.snapshot_dir) if args.snapshot_dir else None
    payload = store.load_current() if store else None
    if payload and isinstance(payload.get("state"), dict):
        state = payload["state"]
    else:
        state = initial_state(bundle)
    if args.tile_rows is not None:
        state["config"]["tile_rows"] = int(args.tile_rows)

    injections = []
    if args.injections:
        injection_doc = yaml.safe_load(Path(args.injections).read_text(encoding="utf-8"))
        injections = injection_doc.get("injections", [])

    engine = RenderEngine(
        state,
        injections=injections,
        workers=args.workers,
        task_delay_ms=args.task_delay_ms,
        output_dir=args.output_dir,
        product_checkpoints=args.product_checkpoints,
    )
    decision: dict[str, Any] = {"decision": "blank_full_render", "code": Codes.OK}
    if payload:
        decision = restore_payload_into_engine(engine, payload)

    if args.pid_file:
        pid_path = Path(args.pid_file)
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()), encoding="utf-8")

    result = engine.render_all(checkpoint=store.commit if store else None)
    if store:
        store.commit(engine)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "version": __version__,
        "restore_decision": decision,
        "summary": result["summary"],
        "failures": result["failures"],
        "completed_frames": result["completed_frames"],
        "max_observed_concurrency": engine.max_observed_concurrency,
        "worker_limit": engine.workers,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_failure_list(result["failures"], report_path.parent / "render_failures.json")
    _print_json({
        "status": result["status"],
        "completed_frames": result["completed_frames"],
        "failures": [(f.get("frame"), f.get("stage"), f.get("code")) for f in result["failures"]],
        "report": str(report_path),
        "output_dir": args.output_dir,
    })
    return 0


def cmd_run_commands(args: argparse.Namespace) -> int:
    from .acceptance import run_command_acceptance
    result = run_command_acceptance(
        Path(args.root),
        stream_path=args.stream,
        report_dir=Path(args.report_dir),
        workers=args.workers,
        budget_ms=args.budget_ms,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def cmd_verify(args: argparse.Namespace) -> int:
    from .acceptance import run_full_verification
    report = run_full_verification(Path(args.root), Path(args.artifacts_dir))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def cmd_gui(args: argparse.Namespace) -> int:
    from .tk_app import launch
    launch(Path(args.root))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="condrender", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-samples")
    p.add_argument("--root", default=".")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate")
    p.add_argument("--root", default=".")
    p.add_argument("--output", default="artifacts/validation/failure_list.json")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("render")
    p.add_argument("--root", default=".")
    p.add_argument("--snapshot-dir", default="artifacts/snapshots/default")
    p.add_argument("--output-dir", default="artifacts/render/default")
    p.add_argument("--report", default="artifacts/render/default/render_report.json")
    p.add_argument("--injections", default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--task-delay-ms", type=float, default=0.0)
    p.add_argument("--tile-rows", type=int, default=None)
    p.add_argument("--product-checkpoints", action="store_true")
    p.add_argument("--pid-file", default=None)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("run-commands")
    p.add_argument("--root", default=".")
    p.add_argument("--stream", default=None)
    p.add_argument("--report-dir", default="artifacts/command-acceptance")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--budget-ms", type=float, default=None)
    p.set_defaults(func=cmd_run_commands)

    p = sub.add_parser("verify")
    p.add_argument("--root", default=".")
    p.add_argument("--artifacts-dir", default="artifacts/verification")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("gui")
    p.add_argument("--root", default=".")
    p.set_defaults(func=cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
