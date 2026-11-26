"""Command line entry points for generation and pipeline execution."""

import argparse
import json
import os
import signal
import sys

from .pipeline import Pipeline, ShutdownFlag
from .snapshot import SnapshotStore
from .synth import generate_demo, relabel_transcript


def _install_signals(flag):
    def handler(signum, frame):  # pragma: no cover - exercised via script
        flag.request(f"signal_{signum}")
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    return flag


def cmd_generate(args):
    paths = generate_demo(args.samples, args.configs, args.label_version)
    print(json.dumps({"generated": paths}, indent=2))


def cmd_run(args):
    shutdown = _install_signals(ShutdownFlag())
    pipeline = Pipeline(
        args.samples, args.config, args.snapshots,
        tier_name=args.tier, max_batches=args.max_batches,
        shutdown=shutdown)
    result = pipeline.run()
    print(json.dumps({
        "mode": result["mode"],
        "manifest": result["manifest"],
        "segments": [
            {"segment_id": s["segment_id"], "label": s["label"],
             "confidence": s["confidence"], "boundary": s["boundary"],
             "suppressed": s["suppressed"]}
            for s in result["segments"]
        ],
        "state": result["state"],
        "audit_counts": {
            "reused": len(result["audit"]["reused"]),
            "computed": len(result["audit"]["computed"]),
        },
    }, indent=2))


def cmd_relabel(args):
    transcript_path = os.path.join(args.samples, "dialogue_demo.json")
    with open(transcript_path, "r", encoding="utf-8") as handle:
        transcript = json.load(handle)
    changed = args.changed.split(",") if args.changed else []
    new = relabel_transcript(transcript, args.version, changed)
    with open(transcript_path, "w", encoding="utf-8") as handle:
        json.dump(new, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"label_version": args.version,
                      "changed_segments": changed}, indent=2))


def cmd_snapshot(args):
    store = SnapshotStore(args.snapshots)
    print(json.dumps({
        "current": store.current_path(),
        "manifest": store.read_json("manifest.json"),
    }, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(prog="vdu")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--samples", default="samples")
    gen.add_argument("--configs", default="configs")
    gen.add_argument("--label-version", type=int, default=3)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run")
    run.add_argument("--samples", default="samples")
    run.add_argument("--config", default="configs/calibration.json")
    run.add_argument("--snapshots", default="snapshots")
    run.add_argument("--tier", default="coarse",
                     choices=["coarse", "fine", "reference"])
    run.add_argument("--max-batches", type=int, default=None)
    run.set_defaults(func=cmd_run)

    rel = sub.add_parser("relabel")
    rel.add_argument("--samples", default="samples")
    rel.add_argument("--version", type=int, required=True)
    rel.add_argument("--changed", default="")
    rel.set_defaults(func=cmd_relabel)

    snap = sub.add_parser("snapshot")
    snap.add_argument("--snapshots", default="snapshots")
    snap.set_defaults(func=cmd_snapshot)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
