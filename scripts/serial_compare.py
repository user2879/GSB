"""Run concurrent and serial passes and emit a serial comparison summary."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ab_workbench.config import load_config
from ab_workbench.loader import load_injections
from ab_workbench.pipeline import Workbench

COMPARED = ("group_stats", "estimation", "attribution", "failures", "bucketing")


def _run(args, concurrent, tag):
    config = load_config(args.config)
    bench = Workbench(
        config=config,
        snapshot_dir=os.path.join(args.workdir, tag, "snapshot"),
        cache_dir=os.path.join(args.workdir, tag, "cache"),
        tmp_dir=os.path.join(args.workdir, tag, "tmp"),
        concurrent=concurrent,
    )
    injections = load_injections(args.injections) if args.injections else []
    return bench.run(args.log, args.attributes, injections=injections).to_dict()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--injections")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    concurrent = _run(args, True, "concurrent")
    serial = _run(args, False, "serial")
    mismatches = [field for field in COMPARED if concurrent[field] != serial[field]]
    summary = {
        "concurrent_mode": concurrent["mode"],
        "serial_mode": serial["mode"],
        "concurrency": concurrent["concurrency"],
        "compared_fields": list(COMPARED),
        "mismatches": mismatches,
        "identical": not mismatches,
        "serial_group_order": list(serial["group_stats"].keys()),
        "serial_attribution_order": [layer["attribute_key"] for layer in serial["attribution"]],
        "serial_failure_codes": [f["code"] for f in serial["failures"]],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, sort_keys=True, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
