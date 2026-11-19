from __future__ import annotations

import argparse
import json
import select
import shlex
import sys
from pathlib import Path
from typing import Any

from .codec import parse_v1_package, parse_v2_package
from .config import PipelineConfig
from .conversion import v1_to_v2, v2_to_v1
from .interactive import InteractiveEngine
from .io_utils import atomic_write_json, read_json
from .pipeline import SampleLayout, build_session
from .snapshot import GracefulShutdown, SnapshotStore
from .verification import run_verification


def _sample_layout(sample: str | Path) -> SampleLayout:
    return SampleLayout.discover(sample)


def command_generate(args: argparse.Namespace) -> int:
    from tools.generate_samples import generate

    manifest = generate(Path(args.out))
    print(
        json.dumps(
            {
                "generated": str(Path(args.out).resolve()),
                "seed": manifest["seed"],
                "files": len(manifest["files"]),
            }
        )
    )
    return 0


def command_run(args: argparse.Namespace) -> int:
    report = run_verification(args.sample, args.work)
    atomic_write_json(args.out, report)
    print(
        json.dumps(
            {
                "report": str(Path(args.out).resolve()),
                "overall_pass": report["overall_pass"],
                "failure_codes": report["failure_codes"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["overall_pass"] else 1


def command_convert(args: argparse.Namespace) -> int:
    layout = _sample_layout(args.sample)
    config = PipelineConfig.load(layout.config_path)
    session = build_session(args.sample, config=config)
    output = Path(args.out)
    if args.direction == "v1-to-v2":
        bundle = parse_v1_package(
            layout.v1_path,
            config=config,
            metadata=session.metadata_document,
        )
        report = v1_to_v2(bundle, output, config=config)
    else:
        bundle = parse_v2_package(layout.v2_path, config=config)
        sidecar = (
            Path(args.sidecar)
            if args.sidecar
            else output.with_name(f"{output.stem}.sidecar.json")
        )
        report = v2_to_v1(bundle, output, sidecar, config=config)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["comparison"]["pixel_equal"] else 1


def _parse_interactive_line(line: str) -> dict[str, Any] | None:
    tokens = shlex.split(line)
    if not tokens:
        return None
    command = tokens[0]
    if command in {"quit", "exit"}:
        return None
    if command == "threshold" and len(tokens) == 2:
        return {"type": "set_params", "iou_match_threshold": float(tokens[1])}
    if command == "retention" and len(tokens) == 3:
        return {
            "type": "set_params",
            "retention_mode": tokens[1],
            "min_coverage": float(tokens[2]),
        }
    if command == "formats" and len(tokens) == 2:
        return {"type": "set_formats", "formats": tokens[1].split(",")}
    if command == "metadata" and len(tokens) == 4:
        return {
            "type": "replace_metadata",
            "instance_key": tokens[1],
            "confidence": float(tokens[2]),
            "crowd": tokens[3].lower() in {"1", "true", "yes"},
        }
    if command == "fuse":
        return {"type": "fuse"}
    if line.lstrip().startswith("{"):
        return json.loads(line)
    raise ValueError(f"cannot parse interactive command: {line}")


def _load_commands(path: str | Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = read_json(path)
    if isinstance(payload, dict):
        payload = payload.get("commands", [])
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError("interactive command file must contain a JSON list")
    return payload


def _readline_until_signal(shutdown: GracefulShutdown, prompt: str) -> str | None:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    while not shutdown.requested:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if not ready:
            continue
        line = sys.stdin.readline()
        if line == "":
            return None
        return line.rstrip("\n")
    return None


def command_interactive(args: argparse.Namespace) -> int:
    layout = _sample_layout(args.sample)
    config = PipelineConfig.load(layout.config_path)
    session = build_session(args.sample, config=config)
    engine = InteractiveEngine(session)
    shutdown = GracefulShutdown()
    shutdown.install()
    snapshot_root = Path(args.snapshots or layout.snapshot_dir)
    store = SnapshotStore(snapshot_root)
    commands = _load_commands(args.commands)
    processed = 0
    try:
        if commands is not None:
            iterator = iter(commands)
            for item in iterator:
                if shutdown.requested:
                    break
                response = engine.submit(item)
                print(json.dumps(response, ensure_ascii=False), flush=True)
                processed = response["command_seq"]
                if processed % config.snapshot_commit_interval == 0:
                    engine.wait_command(processed)
                    store.commit(
                        session,
                        report_payload={"command_seq": processed},
                    )
        else:
            while not shutdown.requested:
                line = _readline_until_signal(shutdown, "pano> ")
                if line is None:
                    break
                if shutdown.requested:
                    break
                if line.strip() in {"status", "preview"}:
                    print(json.dumps(engine.latest_preview(), ensure_ascii=False))
                    continue
                parsed = _parse_interactive_line(line)
                if parsed is None:
                    break
                response = engine.submit(parsed)
                print(json.dumps(response, ensure_ascii=False), flush=True)
                processed = response["command_seq"]
                if processed % config.snapshot_commit_interval == 0:
                    engine.wait_command(processed)
                    store.commit(
                        session,
                        report_payload={"command_seq": processed},
                    )
        responses = engine.wait_all(timeout=120)
        for response in responses:
            if response is not None and response["status"] == "pending":
                raise RuntimeError("command remained pending after final wait")
        store.commit(
            session,
            report_payload={
                "phase": "shutdown",
                "processed_commands": processed,
                "graceful_signal": shutdown.requested,
            },
        )
    finally:
        engine.close()
        shutdown.restore()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pano-pipeline",
        description="Deterministic panoramic segmentation mask pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate deterministic sample")
    generate_parser.add_argument("--out", default="samples")
    generate_parser.set_defaults(handler=command_generate)

    run_parser = subparsers.add_parser("run", help="run full validation pipeline")
    run_parser.add_argument("--sample", default="samples")
    run_parser.add_argument("--out", default="artifacts/verification_report.json")
    run_parser.add_argument("--work", default="artifacts/verification")
    run_parser.set_defaults(handler=command_run)

    verify_parser = subparsers.add_parser("verify", help="alias for full verification")
    verify_parser.add_argument("--sample", default="samples")
    verify_parser.add_argument("--out", default="artifacts/verification_report.json")
    verify_parser.add_argument("--work", default="artifacts/verification")
    verify_parser.set_defaults(handler=command_run)

    convert_parser = subparsers.add_parser("convert", help="convert between mask formats")
    convert_parser.add_argument("--sample", default="samples")
    convert_parser.add_argument(
        "--direction",
        choices=("v1-to-v2", "v2-to-v1"),
        required=True,
    )
    convert_parser.add_argument("--out", required=True)
    convert_parser.add_argument("--sidecar")
    convert_parser.set_defaults(handler=command_convert)

    interactive_parser = subparsers.add_parser(
        "interactive",
        help="adjust threshold/retention and return budgeted previews",
    )
    interactive_parser.add_argument("--sample", default="samples")
    interactive_parser.add_argument("--commands")
    interactive_parser.add_argument("--snapshots")
    interactive_parser.set_defaults(handler=command_interactive)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        code = getattr(exc, "code", "E_INTERNAL")
        print(
            json.dumps(
                {"business_code": code, "message": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
