"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import BusinessError
from .runner import run_full_reference, run_stream


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic streaming ASR data editor")
    sub = parser.add_subparsers(dest="action", required=True)

    run_p = sub.add_parser("run", help="run an interactive command stream")
    run_p.add_argument("--input", required=True, help="sample input directory")
    run_p.add_argument("--output", required=True, help="report/snapshot directory")
    run_p.add_argument("--commands", help="command stream JSON file")
    run_p.add_argument("--faults", help="fault injection JSON file")

    full_p = sub.add_parser("full", help="one-shot full decode reference")
    full_p.add_argument("--input", required=True)
    full_p.add_argument("--output", required=True)
    full_p.add_argument("--overlap-ms", type=int)
    full_p.add_argument("--stable-frames", type=int)
    full_p.add_argument("--backoff-scale", type=float, default=1.0)

    args = parser.parse_args(argv)
    try:
        if args.action == "run":
            commands = _load_json(args.commands) if args.commands else None
            faults = _load_json(args.faults) if args.faults else None
            report = run_stream(args.input, args.output, commands, faults)
            print(json.dumps(report["summary"], sort_keys=True, ensure_ascii=False))
        else:
            reference = run_full_reference(
                args.input, args.output, overlap_ms=args.overlap_ms,
                min_consistent_frames=args.stable_frames,
                backoff_scale=args.backoff_scale)
            print(json.dumps({name: item["text"] for name, item in reference.items()},
                             sort_keys=True, ensure_ascii=False))
    except BusinessError as exc:
        print(json.dumps({"code": exc.code, "message": exc.message},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
