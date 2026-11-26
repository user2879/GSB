"""Command line entry points for samples, rendering and inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app import build_engine
from .config import RenderConfig
from .formats import read_v1, read_v2, write_v1, write_v2
from .pipeline import render_all
from .samplegen import generate_sample_bundle


def _cmd_samples(args) -> None:
    cfg = RenderConfig()
    path = generate_sample_bundle(args.out, cfg=cfg,
                                  frame_count=args.frames)
    print(f"samples written to {path}")


def _cmd_render(args) -> None:
    engine, checks = build_engine(args.bundle, args.cache,
                                  max_workers=args.workers)
    rejected = [item.as_dict() for item in checks if not item.accepted]
    result = render_all(engine)
    print(json.dumps({
        "rejected_checks": rejected,
        "completed_frames": result.completed_frames,
        "consumed_budget": result.consumed_budget,
        "pending_events": result.pending_events,
        "failures": result.failures,
    }, indent=2))
    if args.export:
        Path(args.export).mkdir(parents=True, exist_ok=True)
        records = engine.cache.frame_records()
        write_v2(Path(args.export) / "frames.v2.jsonl", records)
        write_v1(Path(args.export) / "frames.v1.txt", records)
        print(f"exported v1/v2 frame records to {args.export}")


def _cmd_roundtrip(args) -> None:
    if args.input.endswith(".txt"):
        records = read_v1(args.input)
        write_v2(args.output + ".jsonl", records)
        print(f"v1 -> v2: {len(records)} records")
    else:
        records = read_v2(args.input)
        write_v1(args.output + ".txt", records)
        print(f"v2 -> v1: {len(records)} records")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fxrender")
    sub = parser.add_subparsers(dest="command", required=True)

    samples = sub.add_parser("samples")
    samples.add_argument("--out", required=True)
    samples.add_argument("--frames", type=int, default=6)
    samples.set_defaults(func=_cmd_samples)

    render = sub.add_parser("render")
    render.add_argument("--bundle", required=True)
    render.add_argument("--cache", required=True)
    render.add_argument("--workers", type=int, default=None)
    render.add_argument("--export", default=None)
    render.set_defaults(func=_cmd_render)

    trip = sub.add_parser("roundtrip")
    trip.add_argument("--input", required=True)
    trip.add_argument("--output", required=True)
    trip.set_defaults(func=_cmd_roundtrip)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
