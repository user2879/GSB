from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .cache import ArtifactStore
from .config import load_config, load_json
from .conversion import convert_plane
from .engine import (
    RemixEngine,
    compare_signatures,
    render_signature,
)
from .formats import parse_package
from .fingerprint import digest_json
from .generate import DEFAULT_SEED, generate_workspace
from .injections import load_injections
from .models import DEFAULT_TRACKS
from .sequence import run_edit_sequence
from .snapshots import SnapshotManager


def _atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "clips": len(report.get("clips", [])),
        "rendered_clips": len(
            [clip for clip in report.get("clips", []) if clip.get("status") == "rendered"]
        ),
        "failures": len(report.get("failures", [])),
        "cache": report.get("cache_summary"),
        "snapshot_version": report.get("snapshots", {}).get("global_version"),
        "concurrency_matches": report.get("concurrency_comparison", {}).get("matches"),
        "report_signature": report.get("render_signature_sha256"),
    }


def _write_report(args: argparse.Namespace, report: dict[str, Any], workdir: Path) -> Path:
    report_path = Path(args.report) if args.report else workdir / "report.json"
    _atomic_write_json(report_path, report)
    return report_path


def command_generate(args: argparse.Namespace) -> int:
    manifest = generate_workspace(args.root, seed=args.seed)
    print(
        json.dumps(
            {
                "status": "generated",
                "root": str(Path(args.root).resolve()),
                "seed": manifest["seed"],
                "clips": [item["clip_id"] for item in manifest["clips"]],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _serial_comparison(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    injections,
    report: dict[str, Any],
    workdir: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="serial_compare_", dir=str(workdir)) as temp:
        serial = RemixEngine(
            samples_root=args.samples_root,
            config=config,
            workdir=temp,
            injections=injections,
            force_workers=1,
            quiet=args.quiet,
        ).execute()
        if serial.stopped or serial.report is None:
            return {
                "matches": False,
                "status": "serial_comparison_stopped",
                "fields_compared": [],
            }
        comparison = compare_signatures(
            render_signature(report, workdir),
            render_signature(serial.report, Path(temp)),
        )
        comparison.update(
            {
                "mode": "configured thread pool vs single-thread blank-workspace render",
                "parallel_workers": report.get("thread_pool_size"),
                "serial_workers": 1,
                "cache_event_order_matches": (
                    report.get("cache_event_order")
                    == serial.report.get("cache_event_order")
                ),
                "failure_order_matches": (
                    report.get("failure_order") == serial.report.get("failure_order")
                ),
                "serial_cache_summary": serial.report.get("cache_summary"),
            }
        )
        return comparison


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    injections = load_injections(args.injections)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    result = RemixEngine(
        samples_root=args.samples_root,
        config=config,
        workdir=workdir,
        injections=injections,
        force_workers=args.threads,
        quiet=args.quiet,
    ).execute()
    if result.stopped:
        print(
            json.dumps(
                {
                    "status": "stopped_after_stage_commit",
                    "exit_code": 0,
                    "progress": str(result.workdir / "progress.json"),
                    "report_written": False,
                },
                sort_keys=True,
            )
        )
        return 0
    assert result.report is not None
    report = result.report
    if args.skip_serial_compare:
        report["concurrency_comparison"] = {
            "matches": None,
            "status": "skipped_by_cli",
        }
    else:
        report["concurrency_comparison"] = _serial_comparison(
            args=args,
            config=config,
            injections=injections,
            report=report,
            workdir=workdir,
        )
    report["render_signature_sha256"] = digest_json(
        render_signature(report, workdir)
    )
    report_path = _write_report(args, report, workdir)
    print(
        json.dumps(
            {**_summary(report), "report": str(report_path.resolve())},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    comparison = report.get("concurrency_comparison", {})
    return 0 if comparison.get("matches") is not False else 1


def command_run_sequence(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    edits_data = load_json(args.edits)
    if not isinstance(edits_data, dict) or not isinstance(edits_data.get("edits"), list):
        raise ValueError("edit sequence must be an object with an edits array")
    injections = load_injections(args.injections)
    manifest_path = Path(args.samples_root) / "manifest.json"
    sample_manifest = load_json(manifest_path) if manifest_path.is_file() else {"clips": []}
    workdir = Path(args.workdir)
    result = run_edit_sequence(
        samples_root=args.samples_root,
        base_config=config,
        edits=edits_data["edits"],
        workdir=workdir,
        sample_manifest=sample_manifest,
        injections=injections,
        quiet=args.quiet,
    )
    if result.get("stopped"):
        print(
            json.dumps(
                {
                    "status": "stopped_after_stage_commit",
                    "exit_code": 0,
                    "progress": str(workdir / "progress.json"),
                    "report_written": False,
                },
                sort_keys=True,
            )
        )
        return 0
    report = result["report"]
    report_path = _write_report(args, report, workdir)
    print(
        json.dumps(
            {**_summary(report), "report": str(report_path.resolve())},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["edit_sequence"]["all_edits_match_full"] else 1


def command_convert(args: argparse.Namespace) -> int:
    plane = parse_package(
        args.clip_id,
        args.source,
        args.source_format,
        expected_tracks=tuple(args.expected_tracks),
    )
    report = convert_plane(
        plane,
        args.target_format,
        args.out,
        source_package=args.source,
    )
    print(
        json.dumps(
            {
                "status": "converted",
                "destination": str(Path(args.out).resolve()),
                "loss_count": report["loss_count"],
                "report": str((Path(args.out) / "conversion_report.json").resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def command_recover_status(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    cache_root = workdir / "cache" / "artifacts"
    probe = ArtifactStore(cache_root)
    manager = SnapshotManager(
        workdir / "snapshots",
        artifact_exists=lambda key: probe.path_for(key).is_file(),
    )
    print(
        json.dumps(
            {
                "recovery": manager.recovery,
                "status": manager.status(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remix-workbench",
        description="Deterministic multiformat stem remix workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-samples", help="generate deterministic local samples and configs"
    )
    generate.add_argument("--root", default=".")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.set_defaults(function=command_generate)

    run = subparsers.add_parser("run", help="run one incremental remix render")
    run.add_argument("--samples-root", default="samples")
    run.add_argument("--config", required=True)
    run.add_argument("--workdir", default="artifacts/run")
    run.add_argument("--report")
    run.add_argument("--injections")
    run.add_argument("--threads", type=int)
    run.add_argument("--skip-serial-compare", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(function=command_run)

    sequence = subparsers.add_parser(
        "run-sequence", help="apply edits incrementally and compare to blank full renders"
    )
    sequence.add_argument("--samples-root", default="samples")
    sequence.add_argument("--config", required=True)
    sequence.add_argument("--edits", required=True)
    sequence.add_argument("--workdir", default="artifacts/sequence")
    sequence.add_argument("--report")
    sequence.add_argument("--injections")
    sequence.add_argument("--quiet", action="store_true")
    sequence.set_defaults(function=command_run_sequence)

    convert = subparsers.add_parser("convert", help="convert one package between v1/v2/v3")
    convert.add_argument("--source", required=True)
    convert.add_argument("--source-format", required=True, choices=["v1", "v2", "v3"])
    convert.add_argument("--target-format", required=True, choices=["v1", "v2", "v3"])
    convert.add_argument("--out", required=True)
    convert.add_argument("--clip-id", default="clip")
    convert.add_argument("--expected-tracks", nargs="+", default=list(DEFAULT_TRACKS))
    convert.set_defaults(function=command_convert)

    recover = subparsers.add_parser(
        "recover-status", help="inspect only complete atomic snapshots"
    )
    recover.add_argument("--workdir", required=True)
    recover.set_defaults(function=command_recover_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (ValueError, OSError, RuntimeError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
