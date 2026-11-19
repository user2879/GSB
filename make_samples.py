"""Generate deterministic local sample tables and query declarations."""

import json
import os
import random


def build_tables(path):
    rng = random.Random(42)
    rows = 60
    ts, grp, part, val = [], [], [], []
    null_bitmap = [0] * rows
    clock = 1_700_000_000
    for i in range(rows):
        clock += rng.randint(1, 4)
        ts.append(clock)
        grp.append(["alpha", "beta", "gamma"][i % 3])
        part.append(["p0", "p1"][i % 2])
        val.append(round(rng.uniform(-50.0, 150.0), 3))
    null_bitmap[17] = 1  # one null value cell
    val[17] = 0.0
    doc = {"tables": [{
        "name": "events",
        "timestamp_column": "ts",
        "columns": [
            {"name": "ts", "type": "int64", "data": ts},
            {"name": "grp", "type": "string", "data": grp},
            {"name": "part", "type": "string", "data": part},
            {"name": "val", "type": "float64", "data": val,
             "null_bitmap": null_bitmap},
        ],
    }]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    return doc


def build_queries(path, scale=1000):
    doc = {
        "queries": [
            {"id": "agg_by_grp", "kind": "aggregate", "table": "events",
             "group_by": ["grp"], "value_column": "val",
             "aggregates": ["count", "sum", "min", "max", "mean"]},
            {"id": "win_by_part", "kind": "window", "table": "events",
             "partition_by": ["part"], "value_column": "val",
             "interval": {"preceding": 12, "following": 0},
             "max_rows": 5,
             "out_of_order": {"policy": "recompute", "max_lateness": 6}},
        ],
        "session": {"scale": scale, "truncation": "toward_zero",
                    "batch_size": 7, "max_out_of_order_ratio": 0.5},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    return doc


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sample")
    os.makedirs(out_dir, exist_ok=True)
    build_tables(os.path.join(out_dir, "tables.json"))
    build_queries(os.path.join(out_dir, "queries.json"))
    print(f"samples written to {out_dir}")


if __name__ == "__main__":
    main()
