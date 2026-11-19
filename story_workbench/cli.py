"""Command-line interface for sample generation, replay, demo, and GUI launch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import pretty_json, write_utf8
from .demo import run_demo
from .errors import E_NONDETERMINISTIC, WorkbenchError
from .injection import apply_injections, locate_first_flip, parse_injection_config
from .samples import generate_samples
from .spec import load_spec
from .timeline import NarrativeSession


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="story-workbench",
        description="Deterministic narrative branch, event cause, and invalidation workbench.",
    )
    parser.add_argument("--version", action="version", version="story-workbench 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="generate all samples and run the deterministic demo")
    demo_parser.add_argument("--out", default="artifacts/demo", help="output directory")

    generate_parser = subparsers.add_parser("generate-samples", help="write the fixed local sample set")
    generate_parser.add_argument("--out", default="samples", help="sample output directory")

    validate_parser = subparsers.add_parser("validate", help="parse and validate one YAML spec")
    validate_parser.add_argument("spec")

    run_parser = subparsers.add_parser("run", help="replay one sequence and write structured JSON")
    run_parser.add_argument("spec")
    run_parser.add_argument("--sequence", required=True)
    run_parser.add_argument("--out")

    inject_parser = subparsers.add_parser("inject", help="apply a JSON injection config and locate a flip")
    inject_parser.add_argument("spec")
    inject_parser.add_argument("--config", required=True)
    inject_parser.add_argument("--sequence", required=True)
    inject_parser.add_argument("--out")

    gui_parser = subparsers.add_parser("gui", help="open the Tkinter timeline workbench")
    gui_parser.add_argument("--spec", default="artifacts/demo/samples/normal.yaml")
    gui_parser.add_argument("--sequence", default=None)
    return parser


def _emit_json(value: Any, path: str | None) -> None:
    text = pretty_json(value)
    if path:
        write_utf8(Path(path), text)
    else:
        sys.stdout.write(text)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    args = _parser().parse_args(argv)
    try:
        if args.command == "demo":
            summary = run_demo(args.out)
            sys.stdout.write(pretty_json(summary))
            if not summary["byte_identical"]:
                raise WorkbenchError(
                    E_NONDETERMINISTIC,
                    "two in-memory demo builds were not byte-identical",
                    details=summary,
                )
            return 0
        if args.command == "generate-samples":
            paths = generate_samples(args.out)
            sys.stdout.write(
                pretty_json(
                    {
                        "schema": "narrative-workbench/sample-generation/v1",
                        "output_dir": str(Path(args.out)),
                        "files": [path.name for path in paths],
                    }
                )
            )
            return 0
        if args.command == "validate":
            spec = load_spec(args.spec)
            sys.stdout.write(
                pretty_json(
                    {
                        "schema": "narrative-workbench/validation/v1",
                        "valid": True,
                        "spec_id": spec.id,
                        "spec_hash": spec.spec_hash,
                        "nodes": len(spec.nodes),
                        "edges": len(spec.edges),
                        "samples": len(spec.samples),
                        "canonical_spec": spec.canonical_dict(),
                    }
                )
            )
            return 0
        if args.command == "run":
            spec = load_spec(args.spec)
            result = NarrativeSession(spec, args.sequence).run()
            _emit_json(result.as_dict(), args.out)
            return 0
        if args.command == "inject":
            spec = load_spec(args.spec)
            config_text = Path(args.config).read_text(encoding="utf-8")
            config = parse_injection_config(config_text)
            injected_spec, records = apply_injections(spec, config)
            baseline = NarrativeSession(spec, args.sequence).run()
            injected = NarrativeSession(injected_spec, args.sequence).run()
            report = locate_first_flip(
                baseline,
                injected,
                injection_records=records,
                injection_config=config,
            )
            report["baseline_timeline"] = baseline.as_dict()
            report["injected_timeline"] = injected.as_dict()
            _emit_json(report, args.out)
            return 0
        if args.command == "gui":
            from .gui import launch

            return launch(spec_path=args.spec, sequence_id=args.sequence)
        raise WorkbenchError("E_CLI_INVALID", f"unknown command {args.command!r}")
    except WorkbenchError as exc:
        sys.stderr.write(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        return 2
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write(
            json.dumps(
                {
                    "code": "E_CLI_RUNTIME",
                    "message": str(exc),
                    "location": {},
                    "details": {"exception": type(exc).__name__},
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
