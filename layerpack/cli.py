"""Command-line interface for deterministic compression and acceptance flows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .engine import LayerPackEngine, ShutdownController
from .interaction import CompressionSession
from .loader import load_inputs
from .samplegen import DEFAULT_SEED, generate_samples
from .snapshot import SnapshotStore


def _json_dump(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
    )


def _read_commands(path: str) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid command JSON on line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"command on line {line_number} must be an object")
        commands.append(value)
    if not commands:
        raise ValueError("command file contains no commands")
    return commands


def _extract_core(report: dict[str, Any]) -> Any:
    if "result" in report and isinstance(report["result"], dict):
        return report["result"]
    if "core" in report:
        return report["core"]
    return report


def _emit_frame_if_needed(frame: Any, *, quiet: bool) -> None:
    if not quiet:
        print(_json_dump(frame.to_dict()), flush=True)


def _install_shutdown() -> ShutdownController:
    controller = ShutdownController()
    controller.install()
    return controller


def _command_validate(args: argparse.Namespace) -> int:
    loaded, config = load_inputs(
        args.weights, args.calibration, args.config
    )
    payload = {
        "status": "failed" if loaded.failures else "ok",
        "layers": [
            {
                "name": name,
                "shape": list(loaded.layers[name].shape),
                "content_version": loaded.layers[name].content_version,
            }
            for name in sorted(loaded.layers)
        ],
        "calibration_sample_ids": [
            sample.sample_id for sample in loaded.calibration.samples
        ],
        "config_hash": config.config_hash,
        "invalid_layers": sorted(config.invalid_layers),
        "failures": [item.to_dict() for item in loaded.failures],
        "failure_codes": sorted({item.code for item in loaded.failures}),
    }
    print(_json_dump(payload, pretty=args.pretty))
    return 2 if loaded.failures else 0


def _command_run(args: argparse.Namespace) -> int:
    shutdown = _install_shutdown()
    try:
        engine = LayerPackEngine(
            args.weights,
            args.calibration,
            args.config,
            args.state_dir,
            use_cache=not args.no_cache,
            shutdown=shutdown,
        )
        summary = engine.run(full_rebuild=args.full)
    finally:
        shutdown.restore()
    payload = summary.to_dict()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(_json_dump(payload, pretty=True) + "\n", encoding="utf-8")
    print(_json_dump(payload, pretty=args.pretty))
    return summary.exit_code


def _command_commands(args: argparse.Namespace) -> int:
    shutdown = _install_shutdown()
    try:
        commands = _read_commands(args.commands)
        session = CompressionSession(
            args.weights,
            args.calibration,
            args.config,
            args.state_dir,
            reset_session=args.reset,
            shutdown=shutdown,
        )
        session.initialize(wait=True)
        frames = []
        final = None
        for command in commands:
            frame = session.submit_command(
                command, latency_budget_ms=args.latency_budget_ms
            )
            frames.append(frame)
            _emit_frame_if_needed(frame, quiet=args.quiet_frames)
            if frame.payload.get("frame_type") == "interim":
                frame = session.wait_for_final(frame)
                frames.append(frame)
                _emit_frame_if_needed(frame, quiet=args.quiet_frames)
            final = frame
        assert final is not None
        exported: dict[str, str] | None = None
        if args.export_final:
            exported = session.export_final_state(args.export_final)
        full_comparison: dict[str, Any] | None = None
        if args.full_compare:
            comparison_dir = tempfile.mkdtemp(prefix="layerpack-full-compare-")
            full_dir = tempfile.mkdtemp(prefix="layerpack-full-state-")
            try:
                paths = session.export_final_state(comparison_dir)
                full = LayerPackEngine(
                    paths["weights"],
                    paths["calibration"],
                    paths["config"],
                    full_dir,
                    use_cache=False,
                    write_reports=False,
                ).run(full_rebuild=True)
                incremental_core = final.payload["result"]["result"]
                equal = incremental_core == full.comparable()
                full_comparison = {
                    "equal": equal,
                    "incremental_manifest": (
                        incremental_core.get("deployment_manifest") or {}
                    ).get("manifest_sha256"),
                    "full_manifest": (
                        full.comparable().get("deployment_manifest") or {}
                    ).get("manifest_sha256"),
                    "full_status": full.status,
                }
            finally:
                shutil.rmtree(comparison_dir, ignore_errors=True)
                shutil.rmtree(full_dir, ignore_errors=True)
        summary_payload = {
            "status": final.payload.get("status"),
            "frame_type": final.payload.get("frame_type"),
            "persisted": final.payload.get("persisted"),
            "command_count": len(commands),
            "frame_count": len(frames),
            "final": final.to_dict(),
            "exported_final": exported,
            "full_comparison": full_comparison,
        }
        print(_json_dump(summary_payload, pretty=args.pretty))
        if full_comparison is not None and not full_comparison["equal"]:
            return 1
        if "result" in final.payload:
            return int(final.payload["result"].get("exit_code", 0))
        return 2 if final.payload.get("status") == "failed" else 0
    finally:
        shutdown.restore()


def _command_apply(args: argparse.Namespace) -> int:
    commands: list[dict[str, Any]] = []
    if args.command:
        value = json.loads(args.command)
        if isinstance(value, list):
            commands.extend(value)
        else:
            commands.append(value)
    if args.command_file:
        commands.extend(_read_commands(args.command_file))
    if not commands:
        raise ValueError("provide --command or --command-file")
    # Reuse the same accepted frame flow without requiring a temporary file.
    shutdown = _install_shutdown()
    try:
        session = CompressionSession(
            args.weights,
            args.calibration,
            args.config,
            args.state_dir,
            reset_session=args.reset,
            shutdown=shutdown,
        )
        session.initialize(wait=True)
        frames = []
        final = None
        for command in commands:
            frame = session.submit_command(
                command, latency_budget_ms=args.latency_budget_ms
            )
            frames.append(frame)
            _emit_frame_if_needed(frame, quiet=args.quiet_frames)
            if frame.payload.get("frame_type") == "interim":
                frame = session.wait_for_final(frame)
                frames.append(frame)
                _emit_frame_if_needed(frame, quiet=args.quiet_frames)
            final = frame
        assert final is not None
        exported = (
            session.export_final_state(args.export_final)
            if args.export_final
            else None
        )
        print(
            _json_dump(
                {
                    "status": final.payload.get("status"),
                    "persisted": final.payload.get("persisted"),
                    "command_count": len(commands),
                    "final": final.to_dict(),
                    "exported_final": exported,
                },
                pretty=args.pretty,
            )
        )
        if "result" in final.payload:
            return int(final.payload["result"].get("exit_code", 0))
        return 2 if final.payload.get("status") == "failed" else 0
    finally:
        shutdown.restore()


def _command_compare(args: argparse.Namespace) -> int:
    try:
        left = json.loads(Path(args.left).read_text(encoding="utf-8"))
        right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(_json_dump({"equal": False, "error": str(exc)}))
        return 2
    left_core = _extract_core(left)
    right_core = _extract_core(right)
    equal = left_core == right_core
    payload = {"equal": equal, "field_count": len(left_core) if isinstance(left_core, dict) else None}
    if not equal and args.show_diff:
        import difflib

        left_lines = _json_dump(left_core, pretty=True).splitlines()
        right_lines = _json_dump(right_core, pretty=True).splitlines()
        payload["diff"] = list(
            difflib.unified_diff(
                left_lines, right_lines, fromfile=args.left, tofile=args.right, lineterm=""
            )
        )[: args.max_diff_lines]
    print(_json_dump(payload, pretty=args.pretty))
    return 0 if equal else 1


def _command_status(args: argparse.Namespace) -> int:
    snapshot = SnapshotStore(args.state_dir)
    payload = {
        "state_dir": str(Path(args.state_dir).resolve()),
        "restore_point": snapshot.restore_point(),
    }
    latest_report = Path(args.state_dir) / "reports" / "latest.json"
    if latest_report.is_file():
        try:
            report = json.loads(latest_report.read_text(encoding="utf-8"))
            payload["latest_report_status"] = report.get("status")
            payload["latest_snapshot_version"] = report.get("snapshot_version")
        except (OSError, json.JSONDecodeError):
            pass
    pointer = Path(args.state_dir) / "session" / "current.json"
    if pointer.is_file():
        try:
            payload["session_pointer"] = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    print(_json_dump(payload, pretty=True))
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    paths = generate_samples(args.out, args.seed)
    print(_json_dump(paths, pretty=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layerpack",
        description="Deterministic incremental quantization and deployment package builder",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    validate = subparsers.add_parser("validate", help="validate sample inputs")
    validate.add_argument("--weights", required=True)
    validate.add_argument("--calibration", required=True)
    validate.add_argument("--config", required=True)
    validate.add_argument("--pretty", action="store_true")
    validate.set_defaults(func=_command_validate)

    run = subparsers.add_parser("run", help="run all phases")
    run.add_argument("--weights", required=True)
    run.add_argument("--calibration", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--state-dir", required=True)
    run.add_argument("--full", action="store_true", help="ignore all cached artifacts")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--output")
    run.add_argument("--pretty", action="store_true")
    run.set_defaults(func=_command_run)

    commands = subparsers.add_parser("commands", help="apply a JSONL command sequence")
    commands.add_argument("--commands", required=True)
    commands.add_argument("--weights", required=True)
    commands.add_argument("--calibration", required=True)
    commands.add_argument("--config", required=True)
    commands.add_argument("--state-dir", required=True)
    commands.add_argument("--reset", action="store_true")
    commands.add_argument("--latency-budget-ms", type=int)
    commands.add_argument("--export-final")
    commands.add_argument("--full-compare", action="store_true")
    commands.add_argument("--quiet-frames", action="store_true")
    commands.add_argument("--pretty", action="store_true")
    commands.set_defaults(func=_command_commands)

    apply = subparsers.add_parser("apply", help="apply one or more JSON commands")
    apply.add_argument("--command", help="JSON object or array")
    apply.add_argument("--command-file", help="JSONL command file")
    apply.add_argument("--weights", required=True)
    apply.add_argument("--calibration", required=True)
    apply.add_argument("--config", required=True)
    apply.add_argument("--state-dir", required=True)
    apply.add_argument("--reset", action="store_true")
    apply.add_argument("--latency-budget-ms", type=int)
    apply.add_argument("--export-final")
    apply.add_argument("--quiet-frames", action="store_true")
    apply.add_argument("--pretty", action="store_true")
    apply.set_defaults(func=_command_apply)

    compare = subparsers.add_parser("compare", help="compare incremental/full reports")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--show-diff", action="store_true")
    compare.add_argument("--max-diff-lines", type=int, default=120)
    compare.add_argument("--pretty", action="store_true")
    compare.set_defaults(func=_command_compare)

    status = subparsers.add_parser("status", help="show latest valid recovery point")
    status.add_argument("--state-dir", required=True)
    status.set_defaults(func=_command_status)

    recover = subparsers.add_parser("recover", help="alias for status/recovery point")
    recover.add_argument("--state-dir", required=True)
    recover.set_defaults(func=_command_status)

    generate = subparsers.add_parser("generate-samples", help="regenerate deterministic samples")
    generate.add_argument("--out", default="samples")
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.set_defaults(func=_command_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(_json_dump({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
