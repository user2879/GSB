from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .commands import load_command_stream
from .config import RecognitionConfig
from .runner import run_command_sequence
from .samplegen import generate_samples
from .snapshot import SnapshotStore
from .validation import validate_command_sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layout-workbench",
        description="Deterministic synthetic layout text detection/recognition workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="write deterministic PNG/truth samples")
    generate.add_argument("--data-dir", type=Path, default=Path("data"))

    run = subparsers.add_parser("run", help="execute an incremental command stream")
    run.add_argument("--data-dir", type=Path, default=Path("data"))
    run.add_argument("--config", type=Path, default=Path("config/recognition.json"))
    run.add_argument("--commands", type=Path, default=Path("config/commands.json"))
    run.add_argument("--work-dir", type=Path, default=Path("artifacts/run"))
    run.add_argument("--base-dir", type=Path, default=Path.cwd())

    validate = subparsers.add_parser(
        "validate", help="execute incremental and cold full runs after every command"
    )
    validate.add_argument("--data-dir", type=Path, default=Path("data"))
    validate.add_argument("--config", type=Path, default=Path("config/recognition.json"))
    validate.add_argument("--commands", type=Path, default=Path("config/commands.json"))
    validate.add_argument("--work-dir", type=Path, default=Path("artifacts/validation"))
    validate.add_argument("--base-dir", type=Path, default=Path.cwd())

    snapshot = subparsers.add_parser("snapshot-check", help="verify a whole-document snapshot")
    snapshot.add_argument("--snapshot-dir", type=Path, default=Path("artifacts/run/snapshots"))
    snapshot.add_argument("--config", type=Path, default=Path("config/recognition.json"))
    snapshot.add_argument("--cache-dir", type=Path, default=None)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate-samples":
        written = [str(path) for path in generate_samples(args.data_dir)]
        _print({"generated": written})
        return 0
    if args.command == "run":
        config = RecognitionConfig.load(args.config)
        commands = load_command_stream(args.commands)
        index = run_command_sequence(
            data_dir=args.data_dir,
            config=config,
            commands=commands,
            base_dir=args.base_dir,
            work_dir=args.work_dir,
        )
        _print(index)
        return 0
    if args.command == "validate":
        config = RecognitionConfig.load(args.config)
        commands = load_command_stream(args.commands)
        report = validate_command_sequence(
            data_dir=args.data_dir,
            config=config,
            commands=commands,
            base_dir=args.base_dir,
            work_dir=args.work_dir,
        )
        _print(
            {
                "passed": report["passed"],
                "first_overall_divergence": report["first_overall_divergence"],
                "snapshot_verification": report["snapshot_verification"],
                "snapshot_restore": report["snapshot_restore"],
                "concurrency_equivalence": report["concurrency_equivalence"],
                "validation_report": str(args.work_dir / "validation.json"),
            }
        )
        return 0 if report["passed"] else 1
    if args.command == "snapshot-check":
        config = RecognitionConfig.load(args.config)
        store = SnapshotStore(args.snapshot_dir, config.config_hash)
        document = store.load()
        if document is None:
            _print({"valid": False, "error": "snapshot missing"})
            return 1
        try:
            verification = store.verify(expected_config_hash=document.get("config_hash"))
        except Exception as exc:  # noqa: BLE001 - CLI reports verification failures
            _print({"valid": False, "error": str(exc)})
            return 1
        verification["document_digest"] = store.last_document_digest
        if args.cache_dir is not None:
            from .cache import CacheStore
            from .snapshot import snapshot_cache_audit

            verification["cache_audit"] = snapshot_cache_audit(
                document, CacheStore(args.cache_dir)
            )
        _print(verification)
        return 0
    raise AssertionError("unreachable")
