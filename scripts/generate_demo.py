#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from edge_shard_diag.demo import bootstrap_repository_demo  # noqa: E402
from edge_shard_diag.spec import load_spec_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local edge-shard demo data")
    parser.add_argument("--root", default=str(ROOT), help="repository/output root")
    parser.add_argument("--check", action="store_true", help="load and strong-validate after generation")
    args = parser.parse_args()
    paths = bootstrap_repository_demo(args.root)
    result = {"ok": True, "seed": 407, "paths": {key: str(value) for key, value in paths.items()}}
    if args.check:
        bundle = load_spec_bundle(Path(args.root) / "config" / "demo")
        result["validated"] = {"nodes": len(bundle.nodes), "shards": len(bundle.shards), "digest": bundle.content_digest}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
