"""Command-line entry point for the deterministic reranking bench."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_data
from .engine import run_experiment


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def clean_known_outputs(output_dir: Path, report_name: str = "report.json") -> None:
    for name in (
        ".tmp-snapshot",
        ".snapshot-old",
        "snapshot",
        report_name,
        f".{report_name}.tmp",
    ):
        _remove_path(output_dir / name)


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            canonical_data(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temp, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rerank-lab",
        description="Deterministic fixed-point reranking and reranking bench",
    )
    parser.add_argument("--sample-dir", required=True, help="sample directory")
    parser.add_argument("--output-dir", required=True, help="output/snapshot directory")
    parser.add_argument(
        "--injections",
        default=None,
        help="optional independent injection manifest (JSON or YAML)",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="do not remove previous snapshot/report before running",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="disable stage snapshot commits (in-memory evaluation only)",
    )
    parser.add_argument(
        "--report-name",
        default="report.json",
        help="report file name under the output directory (default: report.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = not args.keep_output
    if clean:
        clean_known_outputs(output_dir, args.report_name)
    try:
        report = run_experiment(
            args.sample_dir,
            output_dir,
            injections_path=args.injections,
            clean=clean,
            enable_snapshots=not args.no_snapshots,
        )
    except Exception as exc:
        report = {
            "schema_version": "1.0",
            "tool": "deterministic-rerank-lab",
            "summary": {
                "status": "failed",
                "error_codes": ["E_INPUT_INVALID"],
                "detail": f"{type(exc).__name__}: {exc}",
            },
            "failures": [
                {
                    "code": "E_INPUT_INVALID",
                    "stage": "load",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
        report_text = _json_text(report)
        report_path = output_dir / args.report_name
        _atomic_write(report_path, report_text)
        sys.stdout.write(report_text)
        return 2

    report_text = _json_text(report)
    report_path = output_dir / args.report_name
    _atomic_write(report_path, report_text)
    sys.stdout.write(report_text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
