"""Command line entry point for the ASR scheduling pipeline."""

import argparse
import os
import sys

from . import config as config_mod
from . import fingerprints as fp
from .pipeline import Orchestrator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Deterministic parallel ASR shard pipeline")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--score-config", required=True)
    parser.add_argument("--schedule-config", required=True)
    parser.add_argument("--phoneme-map", required=True)
    parser.add_argument("--injections", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--report", required=True)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--jitter-ms", type=float, default=0.0)
    parser.add_argument("--max-runs", type=int, default=1,
                        help="resume automatically until no pending shard remains")
    return parser.parse_args(argv)


def _resolve_workdir(args):
    if args.workdir:
        return args.workdir
    return os.path.join(args.sample_dir, ".asr_work")


def _injection_hits(orchestrator, injections):
    hits = []
    for injection_id, spec in sorted(injections.items()):
        matched = [
            failure for failure in orchestrator.failures
            if failure.get("injection_id") == injection_id
        ]
        hits.append({
            "injection_id": injection_id,
            "type": spec["type"],
            "audio": spec["audio"],
            "shard_index": spec["shard_index"],
            "hit": bool(matched),
            "business_code": matched[0]["code"] if matched else None,
        })
    return hits


def run_once(args):
    score_params = config_mod.load_score_params(args.score_config)
    schedule_config = config_mod.load_schedule_config(args.schedule_config)
    phoneme_map = config_mod.load_phoneme_map(args.phoneme_map)
    injections = config_mod.load_injections(args.injections)

    orchestrator = Orchestrator(
        args.sample_dir, score_params, schedule_config, phoneme_map,
        injections, _resolve_workdir(args), serial=args.serial,
        jitter_ms=args.jitter_ms,
    )
    orchestrator.prepare()
    remaining = orchestrator.run()
    transcripts = orchestrator.stitch()
    hits = _injection_hits(orchestrator, injections)
    report = orchestrator.build_report(
        transcripts, remaining, hits
    )
    return orchestrator, report, schedule_config, injections


def main(argv=None):
    args = parse_args(argv)
    last_report = None
    for run_index in range(max(1, args.max_runs)):
        orchestrator, report, schedule_config, injections = run_once(args)
        last_report = report
        if not report["pending_queue"]:
            break
        # Pending shards persist via the snapshot/checkpoint cache; next run
        # consumes a fresh global budget and processes them first.
    fp.atomic_write_json(args.report, last_report)

    if last_report["shards_failed"]:
        return 3
    if last_report["pending_queue"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
