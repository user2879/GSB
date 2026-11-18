"""Generate deterministic local sample data, configs, edit commands and
injection config. No network, no external datasets. Fixed seed = 204."""

import json
import os
import random

SEED = 204
DIM = 8
ROWS_PER_BLOCK = 4
NUM_BLOCKS = 6
BASE_ID = 1000


def rand_vector(rng):
    return [round(rng.uniform(-0.6, 0.6), 6) for _ in range(DIM)]


def main(outdir):
    rng = random.Random(SEED)
    data_dir = os.path.join(outdir, "data")
    configs_dir = os.path.join(outdir, "configs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    # ---- table.jsonl -----------------------------------------------------
    records = []
    for block in range(NUM_BLOCKS):
        for slot in range(ROWS_PER_BLOCK):
            records.append({
                "word_id": BASE_ID + block * ROWS_PER_BLOCK + slot,
                "block_index": block,
                "vector": rand_vector(rng),
            })
    table_path = os.path.join(data_dir, "table.jsonl")
    with open(table_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    # ---- requests.json ---------------------------------------------------
    all_ids = [BASE_ID + i for i in range(NUM_BLOCKS * ROWS_PER_BLOCK)]
    sizes = [4, 5, 3, 6, 4, 5, 8, 3, 4, 5, 3, 3]
    requests = []
    for index, size in enumerate(sizes):
        ids = rng.sample(all_ids, size)
        requests.append({"request_id": f"req-{index:04d}", "ids": ids})
    # Deterministic cross-references so edits exercise invalidation.
    requests[0]["ids"][0] = 1003            # hit by replace_row
    requests[1]["ids"][0] = 1009            # hit by replace_block(2)
    requests[4]["ids"][0] = 1005            # hit by nonfinite injection
    requests[11]["ids"][0] = 1008           # hit by replace_block(2)
    # Ids 1025/1026 only exist after append_block.
    requests[10]["ids"] = [1025, 1026, 1003]
    with open(os.path.join(data_dir, "requests.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"requests": requests}, fh, indent=2)

    # ---- edits.json ------------------------------------------------------
    edits = [
        {"op": "replace_row", "word_id": 1003, "vector": rand_vector(rng)},
        {"op": "replace_block", "block_index": 2, "rows": [
            {"word_id": 1008 + i, "vector": rand_vector(rng)}
            for i in range(ROWS_PER_BLOCK)]},
        {"op": "append_block", "rows": [
            {"word_id": 1024 + i, "vector": rand_vector(rng)}
            for i in range(ROWS_PER_BLOCK)]},
        {"op": "set_quant_level", "quant_level": 32768},
        {"op": "set_agg_mode", "agg_mode": "mean"},
        {"op": "replace_row", "word_id": 1026, "vector": rand_vector(rng)},
    ]
    with open(os.path.join(data_dir, "edits.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"commands": edits}, fh, indent=2)

    # ---- config.json -----------------------------------------------------
    config = {
        "block_size": 4096,
        "rows_per_block": ROWS_PER_BLOCK,
        "dim": DIM,
        "quant_level": 65536,
        "frac_bits": 12,
        "agg_mode": "sum",
        "max_concurrency": 4,
        "global_point_budget": 100000,
        "per_request_quota": 64,
        "table_path": "../data/table.jsonl",
        "requests_path": "../data/requests.json",
        "snapshot_dir": "../snapshots",
        "report_path": "../reports/report.json",
    }
    with open(os.path.join(configs_dir, "config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    # ---- injection.json --------------------------------------------------
    injection = {"injections": [
        {"id": "INJ-001", "type": "nonfinite_vector", "stage": "validate",
         "word_id": 1005},
        {"id": "INJ-002", "type": "id_out_of_range", "stage": "gather",
         "request_index": 2, "out_of_range_id": 999999},
        {"id": "INJ-003", "type": "stale_snapshot_layout",
         "stage": "snapshot"},
        {"id": "INJ-004", "type": "zero_quota", "stage": "schedule",
         "request_index": 3},
    ]}
    with open(os.path.join(configs_dir, "injection.json"), "w",
              encoding="utf-8") as fh:
        json.dump(injection, fh, indent=2)

    # Injection run uses a separate snapshot chain and report file.
    inj_config = dict(config)
    inj_config["snapshot_dir"] = "../snapshots_inj"
    inj_config["report_path"] = "../reports/injection_report.json"
    with open(os.path.join(configs_dir, "config_inj.json"), "w",
              encoding="utf-8") as fh:
        json.dump(inj_config, fh, indent=2)

    print(f"sample data written under {outdir}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    main(os.path.abspath(target))
