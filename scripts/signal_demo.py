#!/usr/bin/env python3
"""Deterministic worker that installs signal handlers and runs one batch.

Used by tests/test_signal_handling.py to prove graceful shutdown: on SIGTERM
the engine marks the run as signaled and never publishes a partial snapshot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsseg.config import SegmentationConfig, load_config
from rsseg.engine import IncrementalEngine, install_signal_handlers, request_shutdown
from rsseg.runner import build_inputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--signal-before-start", action="store_true")
    args = parser.parse_args()
    install_signal_handlers()
    if args.signal_before_start:
        request_shutdown()
    cfg = load_config(Path(args.sample_dir) / "segmentation_config.json").to_dict()
    cfg["recompute_budget"] = 2
    inputs = build_inputs(args.sample_dir, args.work_dir,
                          config_override=SegmentationConfig.from_dict(cfg).to_dict())
    report = IncrementalEngine(inputs).run()
    print(report.snapshot_version)
    print("signaled" if report.signaled else "normal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
