"""Generate deterministic local sample declarations (no network, no accounts).

Outputs under samples/:
  clusters.json     - cluster declarations with loopback ports and fixed seeds
  tasks.json        - configuration dispatch tasks
  revisions.json    - linear config revision chain with derived references
  seed_params.json  - fixed-seed parameter file
  revisions_amended.json - amended chain used to demonstrate invalidation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CLUSTERS = {
    "clusters": [
        {"cluster_id": "cluster-alpha", "host": "127.0.0.1", "port": 19001,
         "nodes": ["alpha-node-1", "alpha-node-2"], "seed": 20240101,
         "sample_rows": 64},
        {"cluster_id": "cluster-bravo", "host": "127.0.0.1", "port": 19002,
         "nodes": ["bravo-node-1", "bravo-node-2", "bravo-node-3"],
         "seed": 20240102, "sample_rows": 80},
        {"cluster_id": "cluster-charlie", "host": "127.0.0.1", "port": 19003,
         "nodes": ["charlie-node-1"], "seed": 20240103,
         "sample_rows": 48},
    ]
}

TASKS = {
    "tasks": [
        {"task_id": "task-ratelimit", "keys": ["rate_limit", "burst"],
         "targets": ["cluster-alpha", "cluster-bravo"], "caliber": 1},
        {"task_id": "task-timeout", "keys": ["timeout_ms", "retries"],
         "targets": ["cluster-bravo", "cluster-charlie"], "caliber": 2},
        {"task_id": "task-weighted", "keys": ["rate_limit", "timeout_ms"],
         "targets": ["cluster-alpha", "cluster-charlie"], "caliber": 3},
    ]
}

REVISIONS = {
    "revisions": [
        {"version": 1, "parent": None,
         "values": {"rate_limit": 100, "timeout_ms": 2000},
         "strategy": "canary", "derives": []},
        {"version": 2, "parent": 1,
         "values": {"burst": 50, "retries": 3},
         "strategy": "rolling", "derives": [1]},
        {"version": 3, "parent": 2,
         "values": {"rate_limit": 150},
         "strategy": "rolling", "derives": [1]},
        {"version": 4, "parent": 3,
         "values": {"timeout_ms": 2500},
         "strategy": "full", "derives": [2]},
        {"version": 5, "parent": 4,
         "values": {"burst": 75},
         "strategy": "full", "derives": [2, 3]},
        {"version": 6, "parent": 5,
         "values": {"retries": 5},
         "strategy": "rolling", "derives": [4]},
    ]
}

SEED_PARAMS = {
    "numpy": "1.26.4",
    "generator": "numpy.random.PCG64",
    "value_span": 1000,
    "group_count": 4,
    "integer_fixed_point": True,
    "seeds": {item["cluster_id"]: item["seed"]
              for item in CLUSTERS["clusters"]},
}


def build_amended() -> dict:
    amended = json.loads(json.dumps(REVISIONS))
    for rev in amended["revisions"]:
        if rev["version"] == 3:
            rev["values"] = {"rate_limit": 220}
        if rev["version"] == 4:
            rev["strategy"] = "rolling"
        if rev["version"] == 5:
            rev["derives"] = [2, 4]
    return amended


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic samples")
    parser.add_argument("--out", default="samples")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "clusters.json": CLUSTERS,
        "tasks.json": TASKS,
        "revisions.json": REVISIONS,
        "seed_params.json": SEED_PARAMS,
        "revisions_amended.json": build_amended(),
    }
    for name, payload in files.items():
        with open(out / name, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True,
                      ensure_ascii=False)
            handle.write("\n")
    print(f"wrote {len(files)} sample files to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
