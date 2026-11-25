from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .diagnostics import StructuredLogger
from .errors import LedgerError
from .injections import inject_dangling_reference, inject_snapshot_digest_mismatch, inject_spec_corruption
from .samples import generate_sample_directory
from .verification import run_verification


INJECTION_ALIASES = {
    "spec": "spec",
    "spec_corrupt": "spec",
    "E_SPEC": "spec",
    "ref": "ref",
    "ref_dangling": "ref",
    "E_REF": "ref",
    "snapshot": "snapshot",
    "snapshot_digest": "snapshot",
    "E_SNAPSHOT": "snapshot",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic receipt ledger workspace")
    parser.add_argument("--sample-dir", default="samples/local", help="local sample specification directory")
    parser.add_argument("--runtime-dir", default=".runtime", help="SQLite, snapshots and logs directory")
    parser.add_argument("--generate-samples", action="store_true", help="generate fixed-seed local samples")
    parser.add_argument("--verify", action="store_true", help="run full local acceptance verification")
    parser.add_argument("--ui", action="store_true", help="launch the Tkinter diagnostics console")
    parser.add_argument(
        "--inject",
        choices=sorted(INJECTION_ALIASES),
        help="inject one of spec corruption, dangling reference or snapshot digest mismatch",
    )
    return parser


def run_injection(case_name: str, sample_dir: Path, runtime_dir: Path) -> int:
    case = INJECTION_ALIASES[case_name]
    logger = StructuredLogger(runtime_dir / "logs" / "injections.jsonl")
    try:
        if case == "spec":
            inject_spec_corruption(runtime_dir)
        elif case == "ref":
            inject_dangling_reference(sample_dir)
        else:
            inject_snapshot_digest_mismatch(runtime_dir)
    except LedgerError as exc:
        record = logger.log_error(exc, injection=case)
        print(json.dumps({"injection": case, "reported_code": exc.code, "record": record}, ensure_ascii=False, indent=2))
        return 2
    logger.log("E_UNEXPECTED", "injection did not fail", injection=case)
    return 3


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_dir = Path(args.sample_dir)
    runtime_dir = Path(args.runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_samples:
        path = generate_sample_directory(sample_dir)
        print(json.dumps({"generated": str(path)}, ensure_ascii=False))
        return 0

    if args.inject:
        return run_injection(args.inject, sample_dir, runtime_dir)

    if args.verify:
        logger = StructuredLogger(runtime_dir / "logs" / "verification.jsonl")
        try:
            report = run_verification(sample_dir, runtime_dir / "verify")
        except LedgerError as exc:
            logger.log_error(exc, stage="verification")
            print(json.dumps({"ok": False, **exc.to_dict()}, ensure_ascii=False, indent=2))
            return 2
        output = Path("artifacts") / "verification-result.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        logger.log("VERIFY_OK", "local verification passed", report=str(output))
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"\nVerification report: {output}")
        return 0

    if args.ui:
        from .ui import launch_ui

        return launch_ui(sample_dir, runtime_dir)

    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
