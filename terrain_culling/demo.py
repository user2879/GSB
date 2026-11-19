"""Single-command deterministic demo entry point.

Run::

    python -m terrain_culling.demo --out artifacts/out

It generates both 512 and 2048 heightfields, executes the fixed camera track
through the full cull/streaming pipeline, and writes per-frame reports plus
content hashes. Running it twice produces byte-identical outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np

from . import __version__
from .config import DEFAULT_CONFIG
from .camera import Camera, generate_track
from .terrain import TerrainDataset
from .engine import CullingEngine
from .gl_context import probe_egl
from .diagnostics import canonical_bytes, content_hash, summarize_frame, find_first_divergence


def run_size(size: int, frame_count: int, out_dir: str, config) -> dict:
    dataset = TerrainDataset.generate(size)
    storage = dataset.save(out_dir)
    camera = Camera(config.depth_width, config.depth_height,
                    config.near, config.far, config.fov_rad)
    track = generate_track(frame_count, size)
    engine = CullingEngine(dataset, camera, config)

    reports = []
    for frame_index, camera_frame in enumerate(track):
        report = engine.run_frame(frame_index, camera_frame)
        reports.append(report)

    # Determinism self-check: replay the identical track in a fresh engine.
    replay_engine = CullingEngine(
        TerrainDataset.generate(size), camera, config)
    replay_reports = [replay_engine.run_frame(i, frame)
                      for i, frame in enumerate(generate_track(frame_count, size))]
    divergence = find_first_divergence(reports, replay_reports,
                                       "first_run", "replay")

    report_path = os.path.join(out_dir, f"frames_{size}.json")
    report_payload = canonical_bytes(reports)
    with open(report_path, "wb") as fh:
        fh.write(report_payload)

    summaries = [summarize_frame(r) for r in reports]
    summary_path = os.path.join(out_dir, f"summary_{size}.json")
    summary_payload = canonical_bytes({
        "size": size,
        "frame_count": frame_count,
        "storage": storage,
        "frames": summaries,
        "degradation_sequence": engine.policy.sequence_log,
        "final_budget": engine.ledger.usage(),
        "self_divergence_check": divergence,
    })
    with open(summary_path, "wb") as fh:
        fh.write(summary_payload)

    return {
        "size": size,
        "report_path": os.path.basename(report_path),
        "summary_path": os.path.basename(summary_path),
        "frames_report_sha256": hashlib_hash(report_payload),
        "summary_sha256": hashlib_hash(summary_payload),
        "height_sha256": storage["height_sha256"],
        "manifest_sha256": storage["manifest_sha256"],
        "block_count": len(dataset.blocks),
        "degradation_steps": engine.policy.step,
    }


def hashlib_hash(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic terrain culling demo")
    parser.add_argument("--out", default="artifacts/out", help="output directory")
    parser.add_argument("--frames", type=int, default=16, help="frames per terrain")
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 2048])
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    if not os.access(args.out, os.W_OK):
        print(f"output directory {args.out} is not writable", file=sys.stderr)
        return 2

    gl_probe = probe_egl()
    preflight = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "package_version": __version__,
        "offscreen_gl": gl_probe,
        "config": DEFAULT_CONFIG.to_dict(),
        "sizes": list(args.sizes),
        "frames": args.frames,
    }
    preflight_path = os.path.join(args.out, "preflight.json")
    with open(preflight_path, "wb") as fh:
        fh.write(canonical_bytes(preflight))

    results = []
    for size in args.sizes:
        result = run_size(size, args.frames, args.out, DEFAULT_CONFIG)
        results.append(result)
        print(f"[size={size}] blocks={result['block_count']} "
              f"frames_sha256={result['frames_report_sha256'][:16]} "
              f"degradation_steps={result['degradation_steps']}")

    manifest = {"preflight": preflight, "results": results,
                "overall_sha256": content_hash(results)}
    run_manifest_path = os.path.join(args.out, "run_manifest.json")
    with open(run_manifest_path, "wb") as fh:
        fh.write(canonical_bytes(manifest))
    print(f"run_manifest_sha256={manifest['overall_sha256']}")
    print(f"offscreen_gl_available={gl_probe.get('available')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
