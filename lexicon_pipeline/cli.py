"""Command line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .generator import generate_samples
from .runner import check_sample_formats, execute_local_delivery
from .snapshot import SnapshotManager
from .validation import load_and_validate_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic lexicon/POS ledger pipeline")
    parser.add_argument("--root", default=".", help="project root containing samples/ and configs/")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="generate deterministic samples")
    sub.add_parser("validate", help="validate declarations and exchange samples")
    sub.add_parser("run", help="run baseline, edit stream, incremental/full equivalence and report")
    status = sub.add_parser("snapshot-status", help="inspect/recover a snapshot directory")
    status.add_argument("--snapshot-dir", default="artifacts/snapshot")
    status.add_argument("--lexicon-revision", default=None)
    status.add_argument("--expected-version", type=int, default=None)
    crash = sub.add_parser("crash-demo", help="write staging and send SIGKILL to this process")
    crash.add_argument("--snapshot-dir", default="artifacts/crash-demo-snapshot")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "generate":
        paths = generate_samples(root / "samples")
        print(f"generated {len(paths)} files under {root / 'samples'}")
        return 0

    if args.command == "validate":
        load_and_validate_inputs(root / "samples")
        checks = check_sample_formats(root)
        print(json.dumps({"declarations": "ok", "exchange_samples": checks}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        report = execute_local_delivery(root)
        print(
            json.dumps(
                {
                    "report": str(root / "artifacts/diagnostic-report.json"),
                    "equivalence": report["equivalence"],
                    "receipt_counts": report["receipt_counts"],
                    "business_code_counts": report["business_code_counts"],
                    "precise_reuse_hits": report["precise_reuse_demo"]["incremental_cache"]["hit_total"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "snapshot-status":
        config = load_config(root / "configs/local.json")
        manager = SnapshotManager(root / args.snapshot_dir)
        manifest = manager.load_current_manifest()
        revision = args.lexicon_revision
        if revision is None and manifest is not None:
            revision = manifest.get("lexicon_revision")
        if revision is None:
            revision = "status-check"
        decision = manager.startup_decision(config["config_hash"], revision, args.expected_version)
        print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))
        return 0 if decision["action"] == "resume" else 2

    if args.command == "crash-demo":
        manager = SnapshotManager(root / args.snapshot_dir)
        manager.simulate_forced_kill_after_staging({"demo": {"status": "partial-staging-before-atomic-commit"}})
        return 137

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
