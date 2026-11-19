from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from .canonical import canonical_json
from .config import BudgetConfig, FingerprintConfig
from .demo import run_demo
from .errors import WorkbenchError
from .generator import GeneratorConfig, generate_materials
from .search import search
from .workbench import Workbench


def _load_json_argument(value: str | None) -> dict | None:
    if value is None:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _print_json(value: object) -> None:
    print(canonical_json(value))


def _budget_from_args(args: argparse.Namespace) -> BudgetConfig | None:
    value = _load_json_argument(getattr(args, "budget_json", None))
    return BudgetConfig.from_dict(value) if value is not None else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpw",
        description="Deterministic versioned audio fingerprint retrieval workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the complete reproducible demonstration")
    demo.add_argument("--root", default="artifacts/demo")
    demo.add_argument("--seed", type=int, default=20261119)

    generate = subparsers.add_parser("generate", help="generate deterministic WAV material and truth")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=20261119)
    generate.add_argument("--track-count", type=int, default=7)

    versions = subparsers.add_parser("versions", help="list committed state/index hashes")
    versions.add_argument("--root", required=True)

    imp = subparsers.add_parser("import", help="incrementally add one WAV track")
    imp.add_argument("--root", required=True)
    imp.add_argument("--track-id", required=True)
    imp.add_argument("--audio", required=True)
    imp.add_argument("--budget-json")

    delete = subparsers.add_parser("delete", help="incrementally remove one track")
    delete.add_argument("--root", required=True)
    delete.add_argument("--track-id", required=True)

    config = subparsers.add_parser("config", help="change fingerprint config and/or threshold")
    config.add_argument("--root", required=True)
    config.add_argument("--fingerprint-config-json")
    config.add_argument("--threshold", type=float)

    undo = subparsers.add_parser("undo", help="undo the latest operation through a new state")
    undo.add_argument("--root", required=True)

    revert = subparsers.add_parser("revert", help="restore a historical state without rebuilding")
    revert.add_argument("--root", required=True)
    revert.add_argument("--version", type=int, required=True)

    query = subparsers.add_parser("search", help="fingerprint and search one query clip")
    query.add_argument("--root", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--version", type=int)
    query.add_argument("--budget-json")

    recover = subparsers.add_parser("recover", help="open, recover orphan state, and close cleanly")
    recover.add_argument("--root", required=True)

    verify = subparsers.add_parser(
        "verify-version",
        help="rebuild a historical index in memory and compare its hashes",
    )
    verify.add_argument("--root", required=True)
    verify.add_argument("--version", type=int, required=True)

    crash = subparsers.add_parser("_crash-import", help=argparse.SUPPRESS)
    crash.add_argument("--root", required=True)
    crash.add_argument("--track-id", required=True)
    crash.add_argument("--audio", required=True)
    crash.add_argument("--crash-after-shards", type=int, default=2)
    return parser


def _open_workbench(root: str, budget: BudgetConfig | None = None) -> Workbench:
    return Workbench(root, budget=budget)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            run_demo(args.root, seed=args.seed)
            return 0
        if args.command == "generate":
            generated = generate_materials(
                args.out,
                GeneratorConfig(seed=args.seed, track_count=args.track_count),
            )
            _print_json(
                {
                    "manifest_sha256": generated["manifest_sha256"],
                    "tracks": {
                        key: Path(value).name for key, value in sorted(generated["tracks"].items())
                    },
                    "queries": {
                        key: Path(value).name for key, value in sorted(generated["queries"].items())
                    },
                }
            )
            return 0
        if args.command == "_crash-import":
            target_write = args.crash_after_shards + 1
            force_signal = getattr(signal, "SIGKILL", signal.SIGTERM)

            def crash_hook(event: str, index: int, temporary: Path) -> None:
                if event == "shard_temp" and index == target_write:
                    os.kill(os.getpid(), force_signal)

            workbench = Workbench(args.root, crash_hook=crash_hook)
            workbench.import_track(args.track_id, args.audio)
            return 70

        budget = _budget_from_args(args)
        workbench = _open_workbench(args.root, budget)
        try:
            if args.command == "versions":
                _print_json({"current_version": workbench.current.version, "versions": workbench.version_chain()})
            elif args.command == "import":
                state = workbench.import_track(args.track_id, args.audio)
                _print_json(
                    {
                        "version": state.version,
                        "state_digest": state.state_digest,
                        "index_hash": state.manifest_hash,
                    }
                )
            elif args.command == "delete":
                state = workbench.delete_track(args.track_id)
                _print_json(
                    {
                        "version": state.version,
                        "state_digest": state.state_digest,
                        "index_hash": state.manifest_hash,
                    }
                )
            elif args.command == "config":
                config_data = _load_json_argument(args.fingerprint_config_json)
                fingerprint_config = (
                    FingerprintConfig.from_dict(config_data) if config_data is not None else None
                )
                state = workbench.update_config(
                    fingerprint_config=fingerprint_config,
                    similarity_threshold=args.threshold,
                )
                _print_json(
                    {
                        "version": state.version,
                        "state_digest": state.state_digest,
                        "index_hash": state.manifest_hash,
                    }
                )
            elif args.command == "undo":
                state = workbench.undo()
                _print_json(
                    {
                        "version": state.version,
                        "state_digest": state.state_digest,
                        "index_hash": state.manifest_hash,
                    }
                )
            elif args.command == "revert":
                state = workbench.revert(args.version)
                _print_json(
                    {
                        "version": state.version,
                        "state_digest": state.state_digest,
                        "index_hash": state.manifest_hash,
                    }
                )
            elif args.command == "search":
                _print_json(
                    search(
                        workbench,
                        args.query,
                        budget=budget,
                        version=args.version,
                    )
                )
            elif args.command == "recover":
                _print_json(workbench.recovery_report)
            elif args.command == "verify-version":
                _print_json(workbench.reconstruct_version(args.version))
            else:
                parser.error(f"unknown command: {args.command}")
        finally:
            workbench.close()
        return 0
    except WorkbenchError as exc:
        print(canonical_json(exc.to_dict()), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
