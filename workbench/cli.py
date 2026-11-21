"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import AlignmentConfig
from .errors import WorkbenchError
from .generator import generate_dataset
from .injections import InjectionSet
from .pipeline import Workbench


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m workbench",
        description="Deterministic cross-modal alignment and retrieval diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate the deterministic local dataset")
    generate.add_argument("--samples", default="samples")
    generate.add_argument("--config", default="config")
    generate.add_argument("--seed", type=int, default=20261122)
    generate.add_argument("--sample-count", type=int, default=8)

    run = subparsers.add_parser("run", help="run the full pipeline")
    run.add_argument("--samples", default="samples")
    run.add_argument("--config", default="config/alignment.json")
    run.add_argument("--injections", default="config/injections.json")
    run.add_argument("--output", default="output")
    run.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    run.add_argument("--report", default=None)

    validate = subparsers.add_parser("validate-config", help="validate a configuration file")
    validate.add_argument("--config", default="config/alignment.json")
    validate.add_argument("--injections", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_dataset(
                args.samples,
                args.config,
                seed=args.seed,
                sample_count=args.sample_count,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "validate-config":
            config = AlignmentConfig.load(args.config)
            injections = (
                InjectionSet.load(args.injections) if args.injections else InjectionSet.empty()
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "config_hash": config.config_hash,
                        "quantization_level": config.quantization_level,
                        "effective_decimal_bits": config.effective_decimal_bits,
                        "injections": injections.audit(),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "run":
            config = AlignmentConfig.load(args.config)
            injections = InjectionSet.load(args.injections) if args.injections else InjectionSet.empty()
            workbench = Workbench(
                samples_dir=args.samples,
                config=config,
                injections=injections,
                output_dir=args.output,
                mode=args.mode,
                report_path=args.report,
            )
            report = workbench.run()
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "mode": report["mode"],
                        "reproducibility_sha256": report["reproducibility_sha256"],
                        "business_codes": report["business_codes"],
                        "retrieval_quality": {
                            key: report["retrieval_quality"][key]
                            for key in (
                                "queries",
                                "self_hit_at_k_rate",
                                "top1_self_hit_rate",
                                "mrr",
                            )
                        },
                        "failure_count": len(report["failures"]),
                        "report_path": str(Path(args.report) if args.report else Path(args.output) / "report.json"),
                        "snapshot": {
                            "status": report["snapshot"]["status"],
                            "current_global_version": report["snapshot"]["current_global_version"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        parser.error("unknown command")
    except WorkbenchError as exc:
        print(
            json.dumps(
                {"status": "error", "code": exc.code, "message": exc.message, "details": exc.details},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 0
