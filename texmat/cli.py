"""Command-line entry points for the texmat component."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_generation_config, load_materials
from .engine import TextureEngine
from .errors import ConfigInvalid, TexmatError
from .loader import scan_textures
from .report import build_report


def _read_material_raw(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(args: argparse.Namespace) -> int:
    sample_dir = Path(args.sample_dir)
    try:
        config = load_generation_config(sample_dir / args.config)
        materials = load_materials(sample_dir / args.materials, config)
    except ConfigInvalid as exc:
        payload = {"status": "failed", "code": exc.code, "message": exc.message, "details": exc.details}
        _emit(args.report, payload)
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    material_raw = _read_material_raw(sample_dir / args.materials)
    textures, decode_failures = scan_textures(sample_dir, config.active_tier)
    if not textures:
        payload = {"status": "failed", "code": "E_TEXTURE_DECODE",
                   "message": "no decodable PNG textures found",
                   "failures": [failure.to_dict() for failure in decode_failures]}
        _emit(args.report, payload)
        return 3

    engine = TextureEngine(config, materials, textures, args.snapshot_dir, material_raw,
                           strict_resume=args.strict_resume,
                           stop_after_commits=args.stop_after_batches)
    if args.handle_signals:
        engine.install_signal_handlers()
    if args.stall_seconds > 0:
        import threading

        def _stall() -> None:
            time.sleep(args.stall_seconds)
            engine._stop_requested.set()  # test hook: emulate a graceful exit signal

        threading.Thread(target=_stall, daemon=True).start()

    run = engine.run_to_completion()
    total_runs = 1
    max_runs = max(100, len(config.noise_layers) * len(materials) * len(("d", "r", "m", "n")) * 4)
    previous_version = -1
    while not run.complete and not run.stopped_by_signal and total_runs < max_runs:
        if run.manifest_version == previous_version:
            break
        previous_version = run.manifest_version
        engine = TextureEngine(config, materials, textures, args.snapshot_dir, material_raw,
                               strict_resume=args.strict_resume,
                               stop_after_commits=args.stop_after_batches)
        if args.handle_signals:
            engine.install_signal_handlers()
        run = engine.run_to_completion()
        total_runs += 1
    run.batches = total_runs
    diagnostics = engine.build_diagnostics() if run.complete else None
    report = build_report(engine, run, decode_failures, diagnostics,
                          report_path=args.report)
    print(json.dumps({"status": report["status"], "batches": report["batches"],
                      "hits": report["cache"]["hits"], "failures": len(report["failures"]),
                      "manifest_version": report["manifest_version"]}, ensure_ascii=False))
    if run.stopped_by_signal:
        return 0
    if not run.complete and run.failures and all(
            item.get("code") == "E_POINT_BUDGET_PENDING" for item in run.failures):
        return 0
    return 0 if run.complete else 1


def _emit(report_path: str | None, payload: dict) -> None:
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="texmat", description="Deterministic texture/material renderer")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run (or resume) a deterministic render")
    run.add_argument("--sample-dir", required=True)
    run.add_argument("--config", default="generation_config.json")
    run.add_argument("--materials", default="materials.json")
    run.add_argument("--snapshot-dir", required=True)
    run.add_argument("--report", default=None)
    run.add_argument("--strict-resume", action="store_true")
    run.add_argument("--handle-signals", action="store_true")
    run.add_argument("--stall-seconds", type=float, default=0.0,
                     help="test hook: request a graceful stop after N seconds")
    run.add_argument("--stop-after-batches", type=int, default=None,
                     help="deterministic hook: stop after N committed tasks")
    run.set_defaults(func=run_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TexmatError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
