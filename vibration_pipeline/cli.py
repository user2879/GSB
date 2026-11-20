from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any, Sequence

from .data import load_yaml
from .engine import StopController, run_pipeline


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return load_yaml(path)


def _install_signal_handlers(controller: StopController) -> dict[int, Any]:
    previous: dict[int, Any] = {}
    candidates = [getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)]
    for candidate in candidates:
        if candidate is None:
            continue

        def handler(signum: int, _frame: Any) -> None:
            controller.request(signum)
            print(f"received signal {signum}; finishing current stage", file=sys.stderr, flush=True)

        try:
            previous[int(candidate)] = signal.getsignal(candidate)
            signal.signal(candidate, handler)
        except (OSError, ValueError):
            pass
    return previous


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vibration-pipeline",
        description="Deterministic incremental anomaly-detection pipeline for synthetic vibration segments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run the four-stage pipeline")
    run_parser.add_argument("--config", default="config/pipeline.yaml", help="pipeline YAML")
    run_parser.add_argument("--data", default="data/devices.yaml", help="device waveform YAML")
    run_parser.add_argument("--injections", default=None, help="independent injection YAML")
    run_parser.add_argument("--edits", default=None, help="segment edit overlay YAML")
    run_parser.add_argument("--snapshot-dir", default=None, help="override snapshot directory")
    run_parser.add_argument("--report", default="artifacts/report.json", help="JSON report path")
    run_parser.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    run_parser.add_argument("--quiet", action="store_true", help="do not print the summary line")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "run":
        parser.error("unsupported command")
        return 2
    try:
        config_path = Path(args.config).resolve()
        data_path = Path(args.data).resolve()
        config_spec = load_yaml(config_path)
        data_spec = load_yaml(data_path)
        injection_path = Path(args.injections).resolve() if args.injections else None
        edit_path = Path(args.edits).resolve() if args.edits else None
        injection_spec = _load_optional(injection_path)
        edits_spec = _load_optional(edit_path)
        configured_snapshot = config_spec.get("snapshot", {}).get("dir", "artifacts/snapshot") if isinstance(config_spec.get("snapshot"), dict) else "artifacts/snapshot"
        snapshot_path = (
            Path(args.snapshot_dir).resolve()
            if args.snapshot_dir
            else _resolve(str(configured_snapshot), config_path.parent.parent if config_path.parent.name == "config" else config_path.parent)
        )
        report_path = Path(args.report).resolve()
        controller = StopController()
        previous_handlers = _install_signal_handlers(controller)
        try:
            report = run_pipeline(
                config_spec,
                data_spec,
                injection_spec,
                snapshot_dir=snapshot_path,
                mode=args.mode,
                edits_spec=edits_spec,
                report_path=report_path,
                stop_controller=controller,
            )
        finally:
            _restore_signal_handlers(previous_handlers)
        if report["snapshot"]["stale_code"]:
            print(
                f"{report['snapshot']['stale_code']}: "
                + ", ".join(report["snapshot"]["stale_reasons"]),
                file=sys.stderr,
            )
        if not args.quiet:
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "report": str(report_path),
                        "segments": report["segment_count"],
                        "failures": report["failure_count"],
                        "reused": report["reuse_count"],
                        "recomputed": report["recompute_count"],
                        "boundaries": len(report["boundary_segments"]),
                        "injection_hits": sum(1 for item in report["injection_hits"] if item["hit"]),
                        "result_fingerprint": report["result_fingerprint"],
                        "exit_code": 0,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
