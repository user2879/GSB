"""Command-line entry point for the local forensic workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .demo import _crash_child, read_config_file, run_demo
from .errors import ForensicError
from .pipeline import execute
from .runcontrol import inspect_run, record_recovery
from .source import SourceSpec, ensure_source
from .state import SnapshotStore
from .util import atomic_write_json


def _json_print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _workspace(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _spec_from_args(args: argparse.Namespace) -> SourceSpec:
    return SourceSpec(
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration_seconds=args.duration,
        seed=args.seed,
        gop_size=args.gop_size,
    )


def _summary(result: Any) -> dict[str, Any]:
    conclusion = result.graph.nodes["conclusion"]
    return {
        "ok": True,
        "snapshot_version": result.commit.version if result.commit else None,
        "snapshot_hash": result.commit.snapshot_hash if result.commit else None,
        "conclusion_hash": result.conclusion_hash,
        "integrity_status": conclusion.payload["integrity_status"],
        "overall_suspicion": conclusion.payload["overall_suspicion"],
        "changed_parameters": result.audit["changed_parameters"],
        "recomputed_count": len(result.audit["recomputed_nodes"]),
        "reused_count": len(result.audit["reused_nodes"]),
        "recomputed_nodes": result.audit["recomputed_nodes"],
        "reuse_counts": result.audit["reuse_counts"],
        "full_verified": result.full_verified,
        "run_control": inspect_run(result.run_session.run_dir)
        if result.run_session is not None
        else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forensic-workbench",
        description="Incremental deterministic video-forensics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic source material")
    generate.add_argument("--workspace", required=True)
    generate.add_argument("--width", type=int, default=640)
    generate.add_argument("--height", type=int, default=360)
    generate.add_argument("--fps", type=int, default=30)
    generate.add_argument("--duration", type=int, default=60)
    generate.add_argument("--seed", type=int, default=20261119)
    generate.add_argument("--gop-size", type=int, default=30)
    generate.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run", help="execute one incremental pipeline run")
    run.add_argument("--workspace", required=True)
    run.add_argument("--config", required=True, help="JSON config; use generate/status to bootstrap")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--run-id")
    run.add_argument("--expected-base-version", type=int)
    run.add_argument("--expected-base-hash")
    run.add_argument("--no-full-verify", action="store_true")
    run.add_argument("--full-recompute", action="store_true")
    run.add_argument("--skip-source-hash-verify", action="store_true")

    status = subparsers.add_parser("status", help="show committed snapshot state")
    status.add_argument("--workspace", required=True)

    config_parser = subparsers.add_parser("config", help="emit the default pipeline config")
    config_parser.add_argument("--workspace", required=True)
    config_parser.add_argument("--output")

    recover = subparsers.add_parser("recover", help="recover committed state and quarantine orphans")
    recover.add_argument("--workspace", required=True)

    demo = subparsers.add_parser("demo", help="run the complete acceptance demonstration")
    demo.add_argument("--workspace", required=True)
    demo.add_argument("--width", type=int, default=640)
    demo.add_argument("--height", type=int, default=360)
    demo.add_argument("--fps", type=int, default=30)
    demo.add_argument("--duration", type=int, default=60)
    demo.add_argument("--seed", type=int, default=20261119)
    demo.add_argument("--gop-size", type=int, default=30)
    demo.add_argument("--skip-crash-recovery", action="store_true")

    crash = subparsers.add_parser("_crash-child", help=argparse.SUPPRESS)
    crash.add_argument("--workspace", required=True)
    crash.add_argument("--run-dir", required=True)
    crash.add_argument("--config", required=True)
    crash.add_argument("--expected-base-version", type=int, required=True)
    crash.add_argument("--expected-base-hash", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            workspace = _workspace(args.workspace)
            if args.overwrite:
                from .source import generate_source

                source = generate_source(workspace / "source", _spec_from_args(args), overwrite=True)
            else:
                source = ensure_source(workspace / "source", _spec_from_args(args))
            _json_print(
                {
                    "ok": True,
                    "source": {
                        "spec": source.spec.to_dict(),
                        "frame_count": source.frame_count,
                        "gop_count": source.gop_count,
                        "video_sha256": source.video_sha256,
                        "truth_sha256": source.truth_sha256,
                        "pixel_content_sha256": source.content_sha256,
                        "manifest_sha256": source.manifest_digest,
                        "shot_boundaries": sorted(source.shot_boundaries),
                    },
                }
            )
            return 0

        if args.command == "config":
            workspace = _workspace(args.workspace)
            source = ensure_source(workspace / "source")
            config = source.default_pipeline_config()
            output = (
                _workspace(args.output)
                if args.output
                else workspace / "configs" / "default.json"
            )
            atomic_write_json(output, config.to_dict())
            _json_print(
                {
                    "ok": True,
                    "config_path": str(output),
                    "config": config.to_dict(),
                }
            )
            return 0

        if args.command == "run":
            workspace = _workspace(args.workspace)
            config = read_config_file(Path(args.config))
            result = execute(
                workspace,
                config,
                expected_base_version=args.expected_base_version,
                expected_base_hash=args.expected_base_hash,
                verify_full=not args.no_full_verify,
                full_recompute=args.full_recompute,
                verify_source_hashes=not args.skip_source_hash_verify,
                run_dir=_workspace(args.run_dir),
                run_id=args.run_id,
            )
            _json_print(_summary(result))
            return 0

        if args.command == "status":
            workspace = _workspace(args.workspace)
            store = SnapshotStore(workspace)
            snapshot = store.load()
            source_manifest = workspace / "source" / "source_manifest.json"
            source_info: dict[str, Any] | None = None
            if source_manifest.exists():
                try:
                    source = ensure_source(workspace / "source")
                    source_info = {
                        "frame_count": source.frame_count,
                        "video_sha256": source.video_sha256,
                        "truth_sha256": source.truth_sha256,
                        "pixel_content_sha256": source.content_sha256,
                    }
                except ForensicError as exc:
                    source_info = exc.to_dict()["error"]
            if snapshot is None:
                _json_print({"ok": True, "committed": False, "source": source_info})
            else:
                conclusion = snapshot.graph.nodes["conclusion"]
                _json_print(
                    {
                        "ok": True,
                        "committed": True,
                        "version": snapshot.version,
                        "snapshot_hash": snapshot.snapshot_hash,
                        "snapshot_file": str(snapshot.path),
                        "node_count": len(snapshot.graph.nodes),
                        "conclusion_hash": conclusion.content_hash,
                        "integrity_status": conclusion.payload["integrity_status"],
                        "overall_suspicion": conclusion.payload["overall_suspicion"],
                        "param_versions": snapshot.graph.param_versions,
                        "source": source_info,
                    }
                )
            return 0

        if args.command == "recover":
            workspace = _workspace(args.workspace)
            store = SnapshotStore(workspace)
            state_recovery = store.recover_orphans()
            run_recoveries = []
            recovered_runs_root = workspace / "recovery-runs"
            if recovered_runs_root.exists():
                for run_dir in sorted(path for path in recovered_runs_root.iterdir() if path.is_dir()):
                    status = inspect_run(run_dir)
                    if status["status"] == "unclean_termination":
                        path = record_recovery(
                            run_dir,
                            {
                                "clean_exit": False,
                                "reason": "recovered from latest committed snapshot",
                                "run_status": status,
                                "state_recovery": state_recovery["report"],
                            },
                        )
                        run_recoveries.append(str(path))
            snapshot = SnapshotStore(workspace).load()
            _json_print(
                {
                    "ok": True,
                    "state_recovery": state_recovery,
                    "run_recovery_reports": run_recoveries,
                    "committed": snapshot is not None,
                    "version": None if snapshot is None else snapshot.version,
                    "snapshot_hash": None if snapshot is None else snapshot.snapshot_hash,
                    "conclusion_hash": None
                    if snapshot is None
                    else snapshot.graph.nodes["conclusion"].content_hash,
                }
            )
            return 0

        if args.command == "demo":
            report = run_demo(
                _workspace(args.workspace),
                spec=_spec_from_args(args),
                skip_crash_recovery=args.skip_crash_recovery,
            )
            _json_print(report)
            return 0

        if args.command == "_crash-child":
            config = read_config_file(Path(args.config))
            _crash_child(
                _workspace(args.workspace),
                _workspace(args.run_dir),
                config,
                expected_base_version=args.expected_base_version,
                expected_base_hash=args.expected_base_hash,
            )
            return 0
    except ForensicError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return exc.exit_code
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
