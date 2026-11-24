"""Command line entry point for the diarization component.

Examples:
  python run_diariz.py synth --input samples --output output/run
  python run_diariz.py diarize --wav samples/synthetic_3spk.wav \
      --config samples/diar_config.json --output output/run
  python run_diariz.py visualize --summary output/summary.json \
      --html output/timeline.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from diariz.cache import CacheStore, atomic_write_json
from diariz.config import default_config, load_config
from diariz.pipeline import discover_wavs, process_audio, stable_audio_key
from diariz.synth import SynthSpec, synthesize
from diariz.visualize import show_timeline


def _serializable(result: dict) -> dict:
    from dataclasses import asdict
    return {
        "audio": result["audio"],
        "window_count": result["window_count"],
        "clustering": {
            "assignments": result["clustering"].assignments,
            "clusters": {k: asdict(v)
                         for k, v in result["clustering"].clusters.items()},
            "boundary_windows": result["clustering"].boundary_windows,
            "merges": [asdict(m) for m in result["clustering"].merges],
        },
        "segments": [asdict(s) for s in result["segmentation"].segments],
    }


def cmd_synth(args) -> int:
    os.makedirs(args.input, exist_ok=True)
    spec = SynthSpec(duration_s=args.duration,
                     speaker_count=args.speakers, seed=args.seed)
    path = os.path.join(args.input, f"synthetic_{args.speakers}spk.wav")
    truth = synthesize(spec, path)
    print(f"synthesized {path}: {len(truth.segments)} truth segments, "
          f"{len(truth.overlaps)} overlap regions")
    return 0


def cmd_diarize(args) -> int:
    cfg = load_config(args.config) if args.config else default_config()
    os.makedirs(args.output, exist_ok=True)
    cache = CacheStore(os.path.join(args.output, "cache"))
    wavs = [args.wav] if args.wav else discover_wavs(args.input)
    if not wavs:
        print("no wav files found")
        return 2
    results = []
    for wav in wavs:
        result = process_audio(wav, cfg, args.output, cache=cache)
        payload = _serializable(result)
        out = os.path.join(args.output,
                           f"{stable_audio_key(wav)}_result.json")
        atomic_write_json(out, payload)
        results.append(payload)
        print(f"{os.path.basename(wav)}: windows={result['window_count']} "
              f"clusters={len(result['clustering'].clusters)} "
              f"segments={len(result['segmentation'].segments)}")
    cache.flush_index()
    return 0


def cmd_visualize(args) -> int:
    from diariz.visualize import load_summary
    summary = load_summary(args.summary)
    status = show_timeline(summary, args.html)
    print(json.dumps(status, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Speaker diarization component")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="synthesize deterministic WAV")
    synth.add_argument("--input", default="samples")
    synth.add_argument("--duration", type=float, default=6.0)
    synth.add_argument("--speakers", type=int, default=3, choices=(2, 3))
    synth.add_argument("--seed", type=int, default=20240601)
    synth.set_defaults(func=cmd_synth)

    diar = sub.add_parser("diarize", help="run diarization")
    diar.add_argument("--wav")
    diar.add_argument("--input", default="samples")
    diar.add_argument("--config")
    diar.add_argument("--output", default="output/run")
    diar.set_defaults(func=cmd_diarize)

    vis = sub.add_parser("visualize", help="render timeline")
    vis.add_argument("--summary", required=True)
    vis.add_argument("--html", default="output/timeline.html")
    vis.set_defaults(func=cmd_visualize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
