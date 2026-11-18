"""Full-rebuild entry point: ignores existing cache and recomputes from scratch."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ab_workbench.config import load_config
from ab_workbench.loader import load_injections
from ab_workbench.pipeline import Workbench


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attributes", required=True)
    parser.add_argument("--injections")
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    bench = Workbench(
        config=config,
        snapshot_dir=args.snapshot_dir,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
        concurrent=True,
    )
    bench.cache.clear_all()
    bench.snapshot.reset()
    injections = load_injections(args.injections) if args.injections else []
    result = bench.run(
        args.log, args.attributes, injections=injections
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, sort_keys=True, ensure_ascii=False, indent=2)
    print(f"full rebuild report -> {args.report} (exit_code={result.exit_code})")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
