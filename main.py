#!/usr/bin/env python3
"""CLI entry point for the windowed clustering diarization component."""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diarization.pipeline import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Windowed clustering speaker diarization")
    parser.add_argument("--samples", default="samples",
                        help="directory of 16 kHz mono 16-bit PCM WAV files")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--injections", default="injections.json")
    parser.add_argument("--output", default="output")
    parser.add_argument("--serial", action="store_true",
                        help="disable the thread pool (single-threaded run)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        with open(args.injections, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print("E_CONFIG: %s" % exc, file=sys.stderr)
        return 2
    return run_pipeline(args.samples, config, manifest, args.output,
                        serial=args.serial)


if __name__ == "__main__":
    sys.exit(main())
