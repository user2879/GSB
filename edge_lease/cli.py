"""Command line entry points: sample generation, run, replay and verify."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .errors import PlatformError
from .logging_setup import JsonlLogger
from .pipeline import PipelineConfig, load_inputs, run_pipeline
from .replay import replay
from .ledger import Ledger


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def generate_samples(workdir: Path, base_port: int = 47000) -> Dict[str, Path]:
    """Create deterministic registry, shard declaration and two manifest versions."""
    workdir.mkdir(parents=True, exist_ok=True)
    registry = {
        "nodes": [
            {
                "node_id": "edge-a",
                "host": "127.0.0.1",
                "port": base_port,
                "per_node_quota": 2,
                "dispatch_timeout_ms": 1500,
                "retry_policy": "retry",
                "max_retries": 1,
                "fail_mode": "none",
            },
            {
                "node_id": "edge-b",
                "host": "127.0.0.1",
                "port": base_port + 1,
                "per_node_quota": 1,
                "dispatch_timeout_ms": 1500,
                "retry_policy": "none",
                "max_retries": 0,
                "fail_mode": "none",
            },
            {
                "node_id": "edge-c",
                "host": "127.0.0.1",
                "port": base_port + 2,
                "per_node_quota": 2,
                "dispatch_timeout_ms": 800,
                "retry_policy": "retry",
                "max_retries": 2,
                "fail_mode": "none",
            },
        ]
    }
    shards = {
        "paths": [
            {
                "path": "media/intro.bin",
                "length": 128,
                "shards": [
                    {"start": 0, "end": 48},
                    {"start": 48, "end": 96},
                    {"start": 96, "end": 128},
                ],
            },
            {
                "path": "media/body.bin",
                "length": 200,
                "shards": [
                    {"start": 0, "end": 100},
                    {"start": 100, "end": 200},
                ],
            },
            {
                "path": "media/index.json",
                "length": 64,
                "shards": [{"start": 0, "end": 32}, {"start": 32, "end": 64}],
            },
        ]
    }
    version1 = {
        "version": 1,
        "entries": [
            {
                "path": "media/intro.bin",
                "weight": 0.125,
                "targets": ["edge-a", "edge-b"],
                "strategy": "shard",
                "priority": 1,
            },
            {
                "path": "media/body.bin",
                "weight": "0.375",
                "targets": ["edge-a", "edge-c"],
                "strategy": "full",
                "priority": 2,
            },
            {
                "path": "media/index.json",
                "weight": 0.5,
                "targets": ["edge-b", "edge-c"],
                "strategy": "shard",
                "priority": 3,
                "derived": {
                    "sources": [
                        ["media/intro.bin", 0.6],
                        ["media/body.bin", "0.4"],
                    ]
                },
            },
        ],
    }
    version2 = {
        "version": 2,
        "entries": [
            {
                "path": "media/intro.bin",
                "weight": 0.125,
                "targets": ["edge-a", "edge-b"],
                "strategy": "shard",
                "priority": 1,
            },
            {
                "path": "media/body.bin",
                "weight": "0.375",
                "targets": ["edge-a", "edge-c"],
                "strategy": "full",
                "priority": 2,
            },
            {
                "path": "media/index.json",
                "weight": 0.5,
                "targets": ["edge-a", "edge-b", "edge-c"],
                "strategy": "shard",
                "priority": 3,
                "derived": {
                    "sources": [
                        ["media/intro.bin", 0.7],
                        ["media/body.bin", "0.4"],
                    ]
                },
            },
        ],
    }
    paths = {
        "registry": workdir / "nodes.json",
        "shards": workdir / "shards.json",
        "manifest": workdir / "manifest.json",
    }
    _write_json(paths["registry"], registry)
    _write_json(paths["shards"], shards)
    _write_json(paths["manifest"], {"versions": [version1, version2]})
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-lease",
        description="edge-cache shard scheduling and lease-ledger verification",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-samples", help="write deterministic sample files")
    gen.add_argument("--workdir", required=True, type=Path)
    gen.add_argument("--base-port", type=int, default=47000)

    run = sub.add_parser("run", help="run the full dispatch pipeline")
    run.add_argument("--workdir", required=True, type=Path)
    run.add_argument("--registry", required=True, type=Path)
    run.add_argument("--shards", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--scale", type=int, default=1000)
    run.add_argument("--seed", type=int, default=20261123)
    run.add_argument("--global-limit", type=int, default=3)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument(
        "--inject",
        choices=["corrupt_manifest", "force_cycle", "bad_grid"],
        default=None,
    )
    run.add_argument("--report", type=Path, default=None)

    replay_cmd = sub.add_parser("replay", help="reopen ledger and verify replay")
    replay_cmd.add_argument("--workdir", required=True, type=Path)
    replay_cmd.add_argument("--registry", required=True, type=Path)
    replay_cmd.add_argument("--shards", required=True, type=Path)
    replay_cmd.add_argument("--manifest", required=True, type=Path)
    replay_cmd.add_argument("--scale", type=int, default=1000)
    replay_cmd.add_argument("--seed", type=int, default=20261123)
    replay_cmd.add_argument("--report", type=Path, default=None)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        paths = generate_samples(args.workdir, args.base_port)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
        return 0
    if args.command == "run":
        config = PipelineConfig(
            workdir=args.workdir,
            registry_path=args.registry,
            shards_path=args.shards,
            manifest_path=args.manifest,
            scale=args.scale,
            content_seed=args.seed,
            global_limit=args.global_limit,
            batch_size=args.batch_size,
            inject=args.inject,
        )
        try:
            result = run_pipeline(config)
        except PlatformError as exc:
            logger = JsonlLogger(args.workdir / "events.jsonl")
            logger.error("pipeline_failure", **exc.to_dict())
            print(json.dumps(exc.to_dict(), indent=2))
            return 2
        if args.report:
            _write_json(args.report, result)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "replay":
        config = PipelineConfig(
            workdir=args.workdir,
            registry_path=args.registry,
            shards_path=args.shards,
            manifest_path=args.manifest,
            scale=args.scale,
            content_seed=args.seed,
            global_limit=1,
        )
        registry, _decls, manifests = load_inputs(config)
        logger = JsonlLogger(args.workdir / "events.jsonl")
        ledger = Ledger(args.workdir / "lease.sqlite", logger=logger)
        try:
            report = replay(ledger, manifests, args.seed, args.scale)
        except PlatformError as exc:
            logger.error("replay_failure", **exc.to_dict())
            print(json.dumps(exc.to_dict(), indent=2))
            return 2
        finally:
            ledger.close()
        payload = report.to_dict()
        if args.report:
            _write_json(args.report, payload)
        print(json.dumps(payload, indent=2))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
