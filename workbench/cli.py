"""Command line entry point with cooperative exit-signal handling."""

from __future__ import annotations

import argparse
import dataclasses
import signal
import sys
import tempfile
from pathlib import Path

from .config import load_config
from .errors import WorkbenchError
from .pipeline import InterruptFlag, run_pipeline
from .report import build_report, render_strip, write_run_artifacts
from .sample_gen import sample_paths
from .store import SnapshotStore
from .utils import png_decode


def install_signal_handlers(interrupt: InterruptFlag) -> None:
    def _handler(signum, _frame):
        interrupt.trigger(signal.Signals(signum).name)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # non-main thread or unsupported platform
            pass


def _load_final_images(config, report):
    store = SnapshotStore(config.snapshots_dir)
    images = {}
    for sample in report["samples"]:
        if sample["status"] != "ok":
            continue
        png_key = sample["png_key"]
        images[sample["sample"]] = store.object_path(png_key, "png").read_bytes()
    return images


def execute(config_path: str, run_mode: str = "parallel", reset: bool = False,
            serial_check: bool = False) -> int:
    try:
        config = load_config(config_path)
    except WorkbenchError as exc:
        print(f"[workbench] {exc.code}: {exc.message} {exc.context}", file=sys.stderr)
        return 2

    interrupt = InterruptFlag()
    install_signal_handlers(interrupt)

    try:
        report = run_pipeline(config, run_mode=run_mode, reset=reset, interrupt=interrupt)
        serial_report = None
        if serial_check and not interrupt.set:
            # Isolated snapshot store => the serial run genuinely recomputes
            # every stage instead of reusing the parallel run's artifacts.
            with tempfile.TemporaryDirectory(prefix="wb-serial-") as verify_dir:
                verify_config = dataclasses.replace(
                    config,
                    snapshots_dir=verify_dir,
                    raw={**config.raw, "snapshots_dir": verify_dir},
                )
                serial_report = run_pipeline(verify_config, run_mode="serial", reset=True)
        report_doc = build_report(report, serial_report)

        final_pngs = _load_final_images(config, report)
        strip_image = None
        if final_pngs:
            preview_names = sorted(final_pngs)[:4]
            tiles, labels = [], []
            tier = config.tier()
            for name in preview_names:
                source = png_decode(sample_paths(config, name, tier)["png"].read_bytes())
                tiles.extend([source, png_decode(final_pngs[name])])
                labels.extend([f"{name}-src", f"{name}-edit"])
            strip_image = render_strip(tiles, labels)
        run_dir = Path(config.outputs_dir) / report["run_id"]
        artifacts = write_run_artifacts(run_dir, report_doc, strip_image, final_pngs)
    except WorkbenchError as exc:
        print(f"[workbench] {exc.code}: {exc.message} {exc.context}", file=sys.stderr)
        return 3

    print(f"[workbench] run={report['run_id']} tier={report['tier']} mode={run_mode}")
    print(f"[workbench] completed={len(report['completed_samples'])} "
          f"failures={len(report['failures'])} interrupted={report['interrupted']}")
    print(f"[workbench] artifacts: {artifacts['report']}")
    if report["interrupted"]:
        print("[workbench] shutdown signal honored at sample/stage boundary; progress committed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generative editing diagnostic workbench")
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument("--mode", choices=("parallel", "serial"), default="parallel")
    parser.add_argument("--reset", action="store_true", help="ignore existing snapshot and rebuild")
    parser.add_argument("--serial-check", action="store_true",
                        help="re-run serially and include field comparison in the report")
    args = parser.parse_args(argv)
    return execute(args.config, run_mode=args.mode, reset=args.reset,
                   serial_check=args.serial_check)


if __name__ == "__main__":
    raise SystemExit(main())
