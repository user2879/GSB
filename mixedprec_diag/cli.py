from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .deterministic import generate_artifacts, write_manifest
from .engine import DiagnosticSession, RunRequest, ensure_writable_directory, write_report
from .errors import E_CONFIG, E_DATA, E_IO, E_LATENCY, E_OOM, E_PRESSURE, DiagnosticError


EXIT_CODES = {
    E_CONFIG: 2,
    E_DATA: 3,
    E_OOM: 4,
    E_IO: 6,
    E_PRESSURE: 5,
    E_LATENCY: 7,
}


DEFAULT_CONFIGS = (
    Path("configs/float32.json"),
    Path("configs/fixed.json"),
    Path("configs/scaled.json"),
)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=184, help="integer PCG64 random seed")
    parser.add_argument("--samples", type=int, default=48, help="number of deterministic samples")
    parser.add_argument("--input-dim", type=int, default=32, help="input feature dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layer dimension")


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report-dir", default="reports", help="writable run-report directory")
    parser.add_argument("--snapshot-dir", default="snapshots", help="writable session snapshot directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mixedprec-diag",
        description="Deterministic local mixed-precision inference and block-memory diagnostic workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute one validated inference configuration")
    run_parser.add_argument("--config", required=True, help="local inference configuration JSON")
    run_parser.add_argument("--expected-manifest", help="optional digest manifest; mismatch aborts with E_DATA")
    _add_generation_args(run_parser)
    _add_output_args(run_parser)

    demo_parser = subparsers.add_parser("demo", help="run float32, fixed-point and scale-perturbation versions")
    demo_parser.add_argument("--config-dir", default="configs", help="directory containing the three demo configs")
    demo_parser.add_argument("--expected-manifest", help="optional digest manifest for every demo run")
    _add_generation_args(demo_parser)
    _add_output_args(demo_parser)

    ui_parser = subparsers.add_parser("ui", help="launch the Tkinter diagnostic workbench")
    ui_parser.add_argument("--config", default="configs/float32.json", help="initial configuration file")
    ui_parser.add_argument("--auto-run", action="store_true", help="run the initial configuration after startup")
    _add_generation_args(ui_parser)
    _add_output_args(ui_parser)

    manifest_parser = subparsers.add_parser("manifest", help="write deterministic data/parameter digests")
    manifest_parser.add_argument("--output", default="configs/digest_manifest.json")
    _add_generation_args(manifest_parser)

    validate_parser = subparsers.add_parser("validate-config", help="strictly validate one config JSON")
    validate_parser.add_argument("--config", required=True)
    return parser


def _request_from_args(args: argparse.Namespace) -> RunRequest:
    return RunRequest(
        seed=args.seed,
        sample_count=args.samples,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
    )


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def _report_brief(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report["run_id"],
        "config_version": report["config_version"],
        "precision": report["config"]["precision"],
        "preallocation": report["config"]["preallocation"],
        "status": report["summary"]["status"],
        "codes": report["summary"]["codes"],
        "batches": len(report["batches"]),
        "samples": len(report["samples"]),
        "peak_used_blocks": report["memory"]["peak_used_blocks"],
        "output_sha256": report["path_output_sha256"],
        "replacement": report["replacement"],
        "ui_latency_p95_ms": report["ui_latency"]["aggregates"].get("__all__", {}).get("p95_ms"),
    }


def _exercise_query_fast_path(session: DiagnosticSession) -> None:
    session.query_view()
    view = session.current_report
    if view and view.get("batches"):
        first_batch = view["batches"][0]["batch_id"]
        session.query_batch(first_batch)
        session.query_samples_for_batch(first_batch)


def command_run(args: argparse.Namespace) -> int:
    session = DiagnosticSession(
        _request_from_args(args),
        report_dir=args.report_dir,
        snapshot_dir=args.snapshot_dir,
    )
    report = session.run_config_path(args.config, expected_manifest_path=args.expected_manifest)
    _exercise_query_fast_path(session)
    session.write_session_summary()
    _print_json(_report_brief(session.current_report or report))
    return _report_exit_code(session.current_report or report)


def command_demo(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir)
    config_paths = [config_dir / path.name for path in DEFAULT_CONFIGS]
    for path in config_paths:
        if not path.exists():
            raise DiagnosticError(E_CONFIG, "demo configuration is missing", "$", {"path": str(path)})

    session = DiagnosticSession(
        _request_from_args(args),
        report_dir=args.report_dir,
        snapshot_dir=args.snapshot_dir,
    )
    summaries: list[dict[str, Any]] = []
    for path in config_paths:
        report = session.run_config_path(path, expected_manifest_path=args.expected_manifest)
        _exercise_query_fast_path(session)
        summaries.append(_report_brief(session.current_report or report))

    session_summary = session.write_session_summary()
    demo_summary = {
        "schema": "mixedprec-diagnostic-demo/v1",
        "request": session.request.to_dict(),
        "artifact_digests": session_summary["artifact_digests"],
        "runs": summaries,
        "final_session": session_summary,
        "verification": {
            "all_replays_verified": all(
                report["memory"]["replay"]["verified"]
                for report in _load_version_reports(session.report_dir)
            ),
            "versions_monotonic": _versions_monotonic(summaries),
            "whole_replacement": all(item["replacement"]["replaced_whole"] for item in summaries),
        },
    }
    write_report(Path(args.report_dir) / "demo_summary.json", demo_summary)
    _print_json(demo_summary)
    if summaries:
        for code in (E_OOM, E_PRESSURE, E_LATENCY):
            if any(code in item["codes"] for item in summaries):
                return EXIT_CODES[code]
    return 0


def _report_exit_code(report: dict[str, Any]) -> int:
    codes = set(report.get("summary", {}).get("codes", []))
    for code in (E_OOM, E_PRESSURE, E_LATENCY):
        if code in codes:
            return EXIT_CODES[code]
    return 0


def _load_version_reports(report_dir: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(report_dir.glob("version_*.json")):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def _versions_monotonic(summaries: list[dict[str, Any]]) -> bool:
    versions = [int(item["config_version"]) for item in summaries]
    return all(left < right for left, right in zip(versions, versions[1:]))


def command_manifest(args: argparse.Namespace) -> int:
    artifacts = generate_artifacts(args.seed, args.samples, args.input_dim, args.hidden_dim)
    write_manifest(artifacts, args.output)
    _print_json(
        {
            "output": str(args.output),
            "data_sha256": artifacts.data_sha256,
            "params_sha256": artifacts.params_sha256,
        }
    )
    return 0


def command_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _print_json({"status": "VALID", "config": config.to_dict()})
    return 0


def command_ui(args: argparse.Namespace) -> int:
    try:
        from .ui import launch
    except ModuleNotFoundError as exc:
        if exc.name == "tkinter":
            raise DiagnosticError(
                E_IO,
                "Tkinter is unavailable. On Windows install the standard Python 3.11 distribution and verify `python -c \"import tkinter; tkinter.Tk()\"`.",
                "$.tkinter",
            ) from exc
        raise
    try:
        launch(
            request=_request_from_args(args),
            initial_config=args.config,
            report_dir=args.report_dir,
            snapshot_dir=args.snapshot_dir,
            auto_run=args.auto_run,
        )
    except Exception as exc:  # Tk raises TclError when no graphical display is available
        raise DiagnosticError(
            E_IO,
            "Tkinter could not open a graphical display; verify a Windows desktop session and `python -c \"import tkinter; tkinter.Tk()\"`.",
            "$.display",
            {"exception": type(exc).__name__, "message": str(exc)},
        ) from exc
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return command_run(args)
        if args.command == "demo":
            return command_demo(args)
        if args.command == "manifest":
            return command_manifest(args)
        if args.command == "validate-config":
            return command_validate(args)
        if args.command == "ui":
            return command_ui(args)
        parser.error("unknown command")
    except DiagnosticError as exc:
        _print_json({"status": "ERROR", "error": exc.to_dict()})
        return EXIT_CODES.get(exc.code, 1)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
