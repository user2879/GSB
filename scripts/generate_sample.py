#!/usr/bin/env python3
"""Write deterministic local JSON sample specifications."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsb.sample import build_changed_sample_spec, build_sample_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="samples/diagnostic_spec.json")
    parser.add_argument("--changed-output", default="samples/diagnostic_spec_v2.json")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(build_sample_spec(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.changed_output).write_text(
        json.dumps(build_changed_sample_spec(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(args.changed_output)


if __name__ == "__main__":
    main()
