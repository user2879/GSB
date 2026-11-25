from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_command_stream, load_config, validate_config
from .diagnostics import prepare_workspace, run_acceptance
from .errors import BusinessError
from .exchange import convert_text
from .generator import apply_plan_commands, generate_workspace, load_plan, append_keypoint_command
from .injection import run_injection_suite, write_injection_config
from .ledger import JsonlLedger, SidecarStore, atomic_write_json
from .pipeline import Pipeline
from .snapshot import SnapshotStore
from .validation import load_dataset
from .viewer import launch_viewer


def cmd_sample(args: argparse.Namespace) -> int:
    root = Path(args.root)
    cfg = prepare_workspace(root)
    print(json.dumps({"ok": True, "root": str(root), "records_planned": 54,
                      "config": str(root / "config" / "config.json")}, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root)
    cfg = load_config(args.config or root / "config" / "config.json")
    report = Pipeline(root, cfg).run(
        batch_budget=args.budget,
        kill_before_commit=args.kill_before_commit,
        continue_pending=not args.no_continue_pending,
    )
    print(json.dumps({
        "ok": True,
        "report": str(root / "reports" / "last_report.json"),
        "elapsed_ms": report["elapsed_ms"],
        "receipts": report["receipt_counts"],
        "cache_counts": report["cache_counts"],
        "budget": report["budget"],
        "codes": report["business_code_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    cfg = load_config(args.config or root / "config" / "config.json")
    dataset = load_dataset(root, cfg)
    result = {
        "ok": not dataset["failures"],
        "png_count": len(dataset["images"]),
        "failure_count": len(dataset["failures"]),
        "failures": dataset["failures"],
        "keypoint_fingerprint": dataset["keypoint_fingerprint"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def cmd_convert(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text(encoding="utf-8")
    ledger_dir = Path(args.ledger_dir or Path(args.input).parent)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger = JsonlLedger(ledger_dir / "conversion_ledger.jsonl")
    sidecars = SidecarStore(ledger_dir / "sidecar_store.json", ledger)
    output = convert_text(args.source, args.target, text, sidecars=sidecars, ledger=ledger)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


def cmd_commands(args: argparse.Namespace) -> int:
    root = Path(args.root)
    commands = load_command_stream(args.file)
    plan = load_plan(root)
    structural = [c for c in commands if c["op"] not in {"set_keypoint_version", "set_config"}]
    effects = apply_plan_commands(plan, structural)
    generate_workspace(root, plan)
    for cmd in commands:
        if cmd["op"] == "set_keypoint_version":
            append_keypoint_command(root, cmd)
            current = load_plan(root)
            current["global_version"] = int(current.get("global_version", 1)) + 1
            atomic_write_json(root / "input" / "sample_plan.json", current)
        elif cmd["op"] == "set_config":
            config_path = root / "config" / "config.json"
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            patch = cmd["patch"]
            if "fixed_point" in patch:
                cfg.setdefault("fixed_point", {}).update(patch.pop("fixed_point"))
            cfg.update(patch)
            validate_config(cfg)
            atomic_write_json(config_path, cfg)
    print(json.dumps({"ok": True, "effects": effects}, ensure_ascii=False, indent=2))
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    summary = run_acceptance(args.root)
    print(json.dumps({
        "ok": summary["ok"],
        "passed": summary["passed"],
        "total": summary["total"],
        "report": str(Path(args.root) / "acceptance_report.json"),
        "summary_md": str(Path(args.root) / "acceptance_summary.md"),
    }, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


def cmd_view(args: argparse.Namespace) -> int:
    return launch_viewer(args.root)


def cmd_recover(args: argparse.Namespace) -> int:
    summary = SnapshotStore(Path(args.root)).recovery_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("recoverable") else 1


def cmd_inject(args: argparse.Namespace) -> int:
    summary = run_injection_suite(args.output, args.root)
    print(json.dumps({"ok": summary["all_hits"], "hit_count": summary["hit_count"],
                      "total": summary["total"],
                      "report": str(Path(args.output) / "injection_report.json")},
                     ensure_ascii=False, indent=2))
    return 0 if summary["all_hits"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vertical_ocr", description="Deterministic vertical-layout digitization toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="generate deterministic sample workspace")
    sample.add_argument("--root", default="demo_workspace")
    sample.set_defaults(func=cmd_sample)

    run = sub.add_parser("run", help="run validation, vision, matching, reading and snapshot")
    run.add_argument("--root", default="demo_workspace")
    run.add_argument("--config")
    run.add_argument("--budget", type=int)
    run.add_argument("--kill-before-commit", action="store_true")
    run.add_argument("--no-continue-pending", action="store_true")
    run.set_defaults(func=cmd_run)

    validate = sub.add_parser("validate")
    validate.add_argument("--root", default="demo_workspace")
    validate.add_argument("--config")
    validate.set_defaults(func=cmd_validate)

    convert = sub.add_parser("convert")
    convert.add_argument("--from", dest="source", choices=["v1", "v2", "v3"], required=True)
    convert.add_argument("--to", dest="target", choices=["v1", "v2", "v3"], required=True)
    convert.add_argument("--input", required=True)
    convert.add_argument("--output")
    convert.add_argument("--ledger-dir")
    convert.set_defaults(func=cmd_convert)

    commands = sub.add_parser("apply-commands")
    commands.add_argument("--root", default="demo_workspace")
    commands.add_argument("--file", required=True)
    commands.set_defaults(func=cmd_commands)

    acceptance = sub.add_parser("acceptance")
    acceptance.add_argument("--root", default="artifacts/acceptance-run")
    acceptance.set_defaults(func=cmd_acceptance)

    view = sub.add_parser("view", help="optional Tkinter report/image browser")
    view.add_argument("--root", default="demo_workspace")
    view.set_defaults(func=cmd_view)

    recover = sub.add_parser("recover", help="show scope recoverable from committed snapshot")
    recover.add_argument("--root", default="demo_workspace")
    recover.set_defaults(func=cmd_recover)

    inject = sub.add_parser("inject", help="run isolated fault-injection suite")
    inject.add_argument("--root", default="artifacts/acceptance-run/baseline")
    inject.add_argument("--output", default="artifacts/injection-run")
    inject.set_defaults(func=cmd_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BusinessError as exc:
        payload = {"ok": False, "code": exc.code, "message": exc.message,
                   "field": exc.field, "details": exc.details}
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        root = getattr(args, "root", None)
        if root:
            root_path = Path(root)
            report_dir = root_path / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            failure_list = report_dir / "failure_list.json"
            failures = []
            if failure_list.exists():
                failures = json.loads(failure_list.read_text(encoding="utf-8")).get("failures", [])
            failures.append(exc.to_failure())
            atomic_write_json(failure_list, {"failures": failures})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
