"""Command-line entry points."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

from .audio import decode_wav_24, write_wav_24
from .deconvolution import recompute_channel_ir
from .demo import run_demo
from .errors import WorkbenchError
from .hashing import canonical_json, content_hash, write_bytes_atomic, write_text_atomic
from .models import MeasurementRun, config_from_mapping
from .report import build_report, render_markdown
from .simulation import estimate_noise_sigma
from .snapshot import SnapshotStore


def _demo(args: argparse.Namespace) -> int:
    manifest = run_demo(args.output, verify_determinism=args.verify_determinism)
    print(canonical_json({
        "output": str(Path(args.output).resolve()),
        "steps": len(manifest["steps"]),
        "error_codes": sorted({event["code"] for event in manifest["error_chain"]}),
        "final_snapshot_hash": manifest["final_snapshot_hash"],
        "final_version_stamp": manifest["final_version_stamp"],
        "determinism_check": manifest.get("determinism_check"),
    }, pretty=True), end="")
    return 0


def _report(args: argparse.Namespace) -> int:
    snapshot = SnapshotStore(args.snapshot).load()
    report = build_report(snapshot)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output, canonical_json(report, pretty=True))
    if args.markdown:
        write_text_atomic(Path(args.markdown), render_markdown(report))
    print(report["report_hash"])
    return 0


def _recompute(args: argparse.Namespace) -> int:
    snapshot = SnapshotStore(args.snapshot).load()
    config = config_from_mapping(snapshot.data["config"])
    wav_bytes = Path(args.wav).read_bytes()
    recorded, fmt = decode_wav_24(wav_bytes, expected_sample_rate=48_000, expected_channels=4, expected_bit_depth=24)
    source = np.load(args.source, allow_pickle=False).astype(np.float64)
    if source.ndim != 1 or source.size != recorded.shape[0]:
        raise WorkbenchError(
            "E_AUDIO_FORMAT",
            "source reference and WAV frame counts differ",
            stage="recompute",
            details={"source_samples": int(source.size), "wav_frames": int(recorded.shape[0])},
        )
    truth = snapshot.data["sync"]["truth"]
    sync_markers = tuple(dict(item) for item in truth)
    sweep_starts = tuple(
        int(item["sweep_start_sample"]) for item in truth
    )
    run = MeasurementRun(
        config=config.measurement,
        source=source,
        recorded=recorded,
        wav_bytes=wav_bytes,
        sync_markers=sync_markers,
        sweep_starts=sweep_starts,
        source_hash=content_hash({"source": source}),
        recorded_hash=content_hash({"recorded": recorded}),
        wav_hash=content_hash(wav_bytes),
        noise_sigma_estimate=estimate_noise_sigma(recorded, config.measurement),
    )
    channels = range(4) if args.channel is None else [int(args.channel)]
    results = []
    for channel_index in channels:
        result = recompute_channel_ir(run, config, channel_index)
        results.append(
            {
                "channel_index": result.channel_index,
                "arrival_sample": result.arrival_sample,
                "peak_sample": result.peak_sample,
                "peak_level_db": result.peak_level_db,
                "nfft": result.nfft,
                "window": result.window,
                "truncation_samples": result.truncation_samples,
                "origin_shift_samples": result.origin_shift_samples,
                "window_starts": result.window_starts,
                "normalization_factor": result.normalization_factor,
                "ir_hash": result.ir_hash,
            }
        )
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(result.ir, dtype="<f8"), allow_pickle=False)
        write_bytes_atomic(Path(args.output_dir) / f"channel_{channel_index}_ir.npy", buffer.getvalue())
    summary = {
        "schema": "acoustic-workbench.recompute/1.0",
        "snapshot_hash": snapshot.content_hash,
        "wav_format": {
            "sample_rate": fmt.sample_rate,
            "channels": fmt.channels,
            "bit_depth": fmt.bit_depth,
            "frames": fmt.total_frames,
        },
        "channels": results,
    }
    _write = Path(args.output_dir) / "recompute_summary.json"
    write_text_atomic(_write, canonical_json(summary, pretty=True))
    print(canonical_json(summary, pretty=True), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="acoustic-workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the fixed deterministic demonstration")
    demo.add_argument("--output", default="output/demo")
    demo.add_argument("--verify-determinism", action="store_true")
    demo.set_defaults(func=_demo)

    report = sub.add_parser("report", help="render a report from a snapshot")
    report.add_argument("snapshot")
    report.add_argument("--output", default="output/report.json")
    report.add_argument("--markdown", default="output/report.md")
    report.set_defaults(func=_report)

    recompute = sub.add_parser("recompute", help="independently recompute one channel IR")
    recompute.add_argument("snapshot")
    recompute.add_argument("--wav", required=True)
    recompute.add_argument("--source", required=True)
    recompute.add_argument("--channel", type=int)
    recompute.add_argument("--output-dir", default="output/recompute")
    recompute.set_defaults(func=_recompute)

    gui = sub.add_parser("gui", help="launch the PyQt6 workbench")
    gui.add_argument("snapshot", nargs="?")
    gui.add_argument("--output", default="output/gui")
    gui.add_argument("--auto-run", action="store_true")
    gui.set_defaults(func=lambda args: __import__("acoustic_workbench.gui", fromlist=["main"]).main(
        ([args.snapshot] if args.snapshot else []) + ["--output", args.output] + (["--auto-run"] if args.auto_run else [])
    ))

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        print(canonical_json({"error": exc.to_dict()}, pretty=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
