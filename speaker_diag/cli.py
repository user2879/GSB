"""Command-line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .canonical import canonical_json
from .demo import run_demo
from .errors import DiagnosticError
from .orchestration import probe_workload_ports
from .receipts import context_for, normalize_receipt_file
from .samples import prepare_samples
from .spec import load_workload


def _print_error(exc: DiagnosticError) -> None:
    print(canonical_json(exc.as_dict()), file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speaker-diag", description="Speaker log batch attribution diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="generate built-in deterministic samples")
    prepare.add_argument("--samples", default="samples")

    probe = subparsers.add_parser("probe-ports", help="probe every declared loopback port")
    probe.add_argument("--config", required=True)

    validate = subparsers.add_parser("validate", help="strictly validate a workload declaration")
    validate.add_argument("--config", required=True)

    normalize = subparsers.add_parser("normalize", help="normalize one historical receipt")
    normalize.add_argument("--config", required=True)
    normalize.add_argument("--receipt", required=True)
    normalize.add_argument("--batch-id", required=True)

    demo = subparsers.add_parser("demo", help="run the complete local acceptance demonstration")
    demo.add_argument("--root", default=".")
    demo.add_argument("--samples", default=None)

    gui = subparsers.add_parser("gui", help="show a verification report with Tkinter")
    gui.add_argument("--report", default="artifacts/verification.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_samples(args.samples)
            print(canonical_json({"status": "ok", "manifest": str(Path(args.samples) / "manifest.json"), "digest": manifest["manifest_digest"]}))
            return 0
        if args.command == "probe-ports":
            result = probe_workload_ports(load_workload(args.config))
            print(canonical_json(result))
            return 0
        if args.command == "validate":
            spec = load_workload(args.config)
            print(canonical_json({"status": "ok", "spec_digest": spec.spec_digest, "windows": len(spec.window_ids), "requests": len(spec.requests)}))
            return 0
        if args.command == "normalize":
            spec = load_workload(args.config)
            normalized = normalize_receipt_file(args.receipt, context=context_for(spec, args.batch_id))
            print(normalized.canonical.decode("utf-8"))
            return 0
        if args.command == "demo":
            report = run_demo(Path(args.root), samples_root=args.samples)
            print(
                canonical_json(
                    {
                        "status": report["status"],
                        "checks": report["checks"],
                        "counts": report["counts"],
                        "anomaly_codes": report["anomalies"]["codes"],
                        "report": str(Path(args.root) / "artifacts" / "verification.json"),
                    }
                )
            )
            return 0
        if args.command == "gui":
            from .gui import main as gui_main

            return gui_main(["--report", args.report])
    except DiagnosticError as exc:
        _print_error(exc)
        return 2
    return 1
