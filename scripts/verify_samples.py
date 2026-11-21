#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from news_diag.engine import RunnerOptions, run_workbench  # noqa: E402
from news_diag.io_utils import atomic_write_text, json_text  # noqa: E402


def _run(news: Path, config: Path, injections: Path, serial: bool):
    return run_workbench(
        RunnerOptions(
            news_path=news,
            config_path=config,
            injections_path=injections,
            serial=serial,
        )
    ).report


def _summary(report: dict) -> dict:
    dominant = Counter(
        item["effect"]["dominant_component"]
        for item in report["injections"]["all"]
        if item["status"] == "applied"
        and isinstance(item.get("effect"), dict)
        and item["effect"].get("selection_flip")
    )
    return {
        "documents": len(report["documents"]),
        "failures": len(report["failures"]),
        "failure_codes": dict(
            sorted(Counter(item["code"] for item in report["failures"]).items())
        ),
        "injections_applied": len(report["injections"]["applied"]),
        "injections_not_applied": len(report["injections"]["not_applied"]),
        "injection_selection_flip_dominant_components": dict(sorted(dominant.items())),
        "coverage_boundary_hits": sum(
            item["statistics"]["coverage_boundary_hit_count"] for item in report["documents"]
        ),
        "boundary_sentences": sum(
            item["statistics"]["boundary_sentence_count"] for item in report["documents"]
        ),
        "suppressed_replacements": sum(
            item["statistics"]["suppressed_replacement_count"] for item in report["documents"]
        ),
        "rewrite_table_warnings": len(report["rewrite_table_warnings"]),
        "rewrite_errors": len(report["rewrite_errors"]),
        "complete": report["completion"]["complete"],
        "dependencies": report["dependencies"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deterministic sample execution")
    parser.add_argument("--news", type=Path, default=ROOT / "samples")
    parser.add_argument("--config", type=Path, default=ROOT / "samples/config.yaml")
    parser.add_argument("--injections", type=Path, default=ROOT / "samples/injections.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/sample_report.json")
    args = parser.parse_args()

    concurrent = _run(args.news, args.config, args.injections, serial=False)
    serial = _run(args.news, args.config, args.injections, serial=True)
    repeated = _run(args.news, args.config, args.injections, serial=False)
    concurrent_text = json_text(concurrent)
    serial_text = json_text(serial)
    repeated_text = json_text(repeated)
    atomic_write_text(args.report, concurrent_text)
    result = _summary(concurrent)
    result.update(
        {
            "report": str(args.report.relative_to(ROOT)) if args.report.is_relative_to(ROOT) else str(args.report),
            "sha256_concurrent": hashlib.sha256(concurrent_text.encode("utf-8")).hexdigest(),
            "sha256_serial": hashlib.sha256(serial_text.encode("utf-8")).hexdigest(),
            "sha256_repeated": hashlib.sha256(repeated_text.encode("utf-8")).hexdigest(),
            "concurrent_equals_serial": concurrent_text == serial_text,
            "concurrent_equals_repeated": concurrent_text == repeated_text,
        }
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
