"""Command-line interface for deterministic sample generation and recognition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import RecognitionEngine
from .errors import AppError
from .runtime import validate_runtime
from .samples import generate_samples
from .util import atomic_write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ancient_vlayout")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic local samples")
    generate.add_argument("--samples", default="samples", help="sample output directory")
    generate.add_argument("--seed", type=int, default=20261124)

    run = subparsers.add_parser("run", help="recognize a local sample directory")
    run.add_argument("--samples", default="samples")
    run.add_argument("--config", default=None)
    run.add_argument("--state-dir", default=".ancient-layout-state")
    run.add_argument("--output", default="artifacts")
    run.add_argument("--injections", action="store_true", help="apply the independent fault plan")
    run.add_argument("--once", action="store_true", help="execute one budget batch only")
    run.add_argument("--force-rebuild", action="store_true")
    run.add_argument("--max-batches", type=int, default=100)

    validate = subparsers.add_parser("validate-runtime", help="check imported package versions")
    validate.add_argument("--json", action="store_true")
    return parser


def _summary(result: Any) -> dict[str, Any]:
    images = result.recognition.get("images", [])
    chars = sum(len(column["chars"]) for image in images for column in image["columns"])
    recognized = sum(
        1
        for image in images
        for column in image["columns"]
        for char in column["chars"]
        if char.get("recognized_char")
    )
    return {
        "snapshot": result.report.get("snapshot", {}),
        "images": len(images),
        "characters": chars,
        "recognized": recognized,
        "failures": len(result.recognition.get("failures", [])),
        "pending": len(result.recognition.get("pending", [])),
        "cache_hits": result.report.get("cache", {}).get("hit_count", 0),
        "recomputed": result.report.get("cache", {}).get("recomputed_count", 0),
        "injection_hits": [
            item["id"]
            for item in result.diagnostics.get("injections", {}).get("items", [])
            if item.get("hit")
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            manifest = generate_samples(args.samples, seed=args.seed)
            print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.command == "validate-runtime":
            report = validate_runtime()
            if args.json:
                print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
            else:
                print(json.dumps(report["packages"], ensure_ascii=False, sort_keys=True))
                for item in report["diagnostics"]:
                    print(f"{item['code']}: {item['message']}", file=sys.stderr)
            return 0 if report["ok"] else 2
        if args.command == "run":
            engine = RecognitionEngine(
                args.samples,
                config_path=args.config,
                state_dir=args.state_dir,
                injection_mode=args.injections,
                force_rebuild=args.force_rebuild,
            )
            result = engine.run() if args.once else engine.settle(max_batches=args.max_batches)
            output = Path(args.output)
            output.mkdir(parents=True, exist_ok=True)
            atomic_write_json(output / "recognition.json", result.recognition)
            atomic_write_json(output / "report.json", result.report)
            atomic_write_json(output / "diagnostics.json", result.diagnostics)
            atomic_write_json(output / "snapshot_manifest.json", result.snapshot_manifest)
            atomic_write_json(output / "run_summary.json", _summary(result))
            print(json.dumps(_summary(result), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
    except AppError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"E_APPLICATION: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
