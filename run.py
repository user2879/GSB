#!/usr/bin/env python3
"""CLI entry for the tabular clustering & density estimation workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tabcluster.pipeline import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--data", default="config/data_description.yaml")
    ap.add_argument("--injections", default="config/injections.yaml")
    ap.add_argument("--workdir", default="out")
    args = ap.parse_args()

    summary = run(args.config, args.data, args.injections, args.workdir)
    budget = summary.get("budget", {})
    print(f"experiment      : {summary.get('experiment')}")
    print(f"candidates      : {len(summary.get('candidates', []))}")
    print(f"budget consumed : {budget.get('consumed')} / "
          f"{budget.get('global_iterations')}")
    print(f"todo (pending)  : {budget.get('todo')}")
    print(f"failures        : {len(summary.get('failures', []))}")
    for f in summary.get("failures", []):
        print(f"  - {f['code']}: {f['message']} "
          f"[stage={f.get('stage')}, candidate={f.get('candidate_key')}]")
    print(f"ranking         : {summary.get('ranking')}")
    print(f"summary written : {Path(args.workdir) / 'outputs' / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
