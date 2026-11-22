#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stemscope.injection import apply_injection_case, load_injection_manifest, run_injection_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a StemScope input fault injection")
    parser.add_argument("--sample-root", default="sample_data")
    parser.add_argument("--manifest", default="sample_data/injection_manifest.json")
    parser.add_argument("--case", required=True, help="case id or 'all'")
    parser.add_argument("--out", required=True)
    parser.add_argument("--parse", action="store_true", help="run parser and print diagnostics")
    args = parser.parse_args()
    manifest = load_injection_manifest(args.manifest)
    cases = manifest["cases"]
    selected = cases if args.case == "all" else [case for case in cases if case["id"] == args.case]
    if not selected:
        raise SystemExit(f"unknown injection case {args.case!r}")
    results = []
    for case in selected:
        destination = Path(args.out) if len(selected) == 1 else Path(args.out) / case["id"]
        apply_injection_case(args.sample_root, destination, case)
        if args.parse:
            results.append(run_injection_case(args.sample_root, destination, case))
        else:
            results.append({"case_id": case["id"], "applied_to": str(destination)})
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all(item.get("passed", True) for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
