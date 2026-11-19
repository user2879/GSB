"""CLI demo entry: run the fixed camera track and emit deterministic reports."""

from __future__ import annotations

import argparse

import yaml

from .camera import CameraTrack
from .pipeline import run_pipeline, write_outputs
from .shading import ShadingParams


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=257, choices=(257, 1025, 4097))
    parser.add_argument("--frames", type=int, default=None,
                        help="override track frame count")
    parser.add_argument("--budget", default="configs/budget.yaml")
    parser.add_argument("--track", default="configs/camera_track.yaml")
    parser.add_argument("--out", default="out")
    parser.add_argument("--no-gl", action="store_true", help="skip GL, force software path")
    args = parser.parse_args(argv)

    with open(args.budget, "r", encoding="utf-8") as fh:
        budget = yaml.safe_load(fh)
    track = CameraTrack.from_yaml(args.track)
    if args.frames:
        track.frames = args.frames
    # track keyframes are authored for the 4097 tier; scale xy to smaller tiers
    track = track.scaled((args.size - 1) / 4096.0)
    shading = ShadingParams.from_dict(budget["shading"]) if "shading" in budget else None

    result = run_pipeline(
        size=args.size,
        seed=budget.get("seed", 20261119),
        track=track,
        tri_budget=budget["triangle_budget"],
        mem_budget=budget["memory_budget_bytes"],
        tau=budget["tau_pixels"],
        shading=shading or ShadingParams(),
        prefer_gl=not args.no_gl,
    )
    write_outputs(result, args.out)
    summary = result["summary"]
    print(f"[demo] size={args.size} frames={track.frames} "
          f"run_hash={summary['run_hash'][:16]} gl={summary['gl_status']}")
    print(f"[demo] residency: {summary['residency_audit']}")
    return summary


if __name__ == "__main__":
    main()
