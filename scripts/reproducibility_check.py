"""两次独立运行比对：地图/清单字节级一致、路径哈希一致。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tactical_pathbench.encoding import canonical_encoding, write_text_file
from tactical_pathbench.workbench import build_workbench


def snapshot(seed: int):
    wb = build_workbench(seed=seed)
    routes = wb.solve_batch()
    return {
        "manifest": canonical_encoding(wb.sample.manifest_payload()),
        "map_hash": wb.grid.content_hash(),
        "partition_signature": wb.grid.region_partition_signature(),
        "path_hashes": {key: route.path_hash() for key, route in sorted(routes.items())},
        "costs": {key: route.cost for key, route in sorted(routes.items())},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    first = snapshot(args.seed)
    second = snapshot(args.seed)
    equal = {
        "manifest_byte_identical": first["manifest"] == second["manifest"],
        "map_hash_identical": first["map_hash"] == second["map_hash"],
        "partition_signature_identical": first["partition_signature"] == second["partition_signature"],
        "path_hashes_identical": first["path_hashes"] == second["path_hashes"],
        "costs_identical": first["costs"] == second["costs"],
    }
    payload = {"seed": args.seed, "equal": equal, "all_equal": all(equal.values()),
               "map_hash": first["map_hash"], "path_hashes": first["path_hashes"]}
    write_text_file(
        ROOT / "artifacts" / "reports" / "reproducibility-report.json",
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
