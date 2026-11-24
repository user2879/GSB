"""Generate deterministic local sample declarations and historical receipts.

Run: python samples/generate_samples.py
Outputs stay fully offline inside the samples directory.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_diagnostic.canonical import sha256_hex

HERE = os.path.dirname(os.path.abspath(__file__))

INPUT_DIM = 6
HIDDEN_DIM = 4
REQUEST_COUNT = 11


def write(name: str, payload) -> None:
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("wrote", os.path.relpath(path, HERE))


def main() -> None:
    requests = [
        {
            "request_id": f"req-{index:03d}",
            "arrival_seq": index,
            "input_dim": INPUT_DIM,
            "seed": 1000 + index,
        }
        for index in range(REQUEST_COUNT)
    ]
    workers = [
        {
            "worker_id": "worker-a",
            "host": "127.0.0.1",
            "port": 5101,
            "seed": 42,
            "param_version": 1,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "batch_size": 3,
            "max_in_flight": 1,
            "retries": 3,
        },
        {
            "worker_id": "worker-b",
            "host": "127.0.0.1",
            "port": 5102,
            "seed": 99,
            "param_version": 1,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "batch_size": 4,
            "max_in_flight": 2,
            "retries": 2,
        },
    ]
    budget = {
        "global_max_in_flight": 2,
        "retry_limit": 3,
        "cost_budget_ms": 100000,
    }
    new_workers = [
        dict(workers[0], seed=4242, param_version=2),
        dict(workers[1]),
    ]

    base_digest = sha256_hex({"sample": "historical-receipt"})
    r1_receipts = [
        {
            "format_version": "r1",
            "batch_id": "batch-worker-a-0000-sample",
            "request_id": "req-000",
            "output_digest": base_digest,
        },
        {
            "format_version": "r1",
            "batch_id": "batch-worker-a-0000-sample",
            "request_id": "req-001",
            "output_digest": sha256_hex({"sample": "r1", "id": "req-001"}),
        },
    ]
    r2_receipts = [
        {
            "format_version": "r2",
            "batch_id": "batch-worker-b-0000-sample",
            "request_id": "req-004",
            "output_digest": sha256_hex({"sample": "r2", "id": "req-004"}),
            "param_version": 1,
            "elapsed_ms": 7,
            "worker_id": "worker-b",
        }
    ]
    r3_receipts = [
        {
            "format_version": "r3",
            "batch_id": "batch-worker-a-0001-sample",
            "request_id": "req-005",
            "output_digest": sha256_hex({"sample": "r3", "id": "req-005"}),
            "param_version": 1,
            "elapsed_ms": 11,
            "worker_id": "worker-a",
            "rank": 0,
            "duplicate_ref": None,
            "param_digest": sha256_hex({"sample": "param", "worker": "worker-a"}),
        },
        {
            "format_version": "r3",
            "batch_id": "batch-worker-a-0001-sample",
            "request_id": "req-006",
            "output_digest": sha256_hex({"sample": "r3", "id": "req-006"}),
            "param_version": 1,
            "elapsed_ms": 12,
            "worker_id": "worker-a",
            "rank": 1,
            "duplicate_ref": "req-005",
            "param_digest": sha256_hex({"sample": "param", "worker": "worker-a"}),
        },
    ]
    write("requests.json", requests)
    write("workers.json", workers)
    write("budget.json", budget)
    write("workers_v2.json", new_workers)
    write("receipts_r1.json", r1_receipts)
    write("receipts_r2.json", r2_receipts)
    write("receipts_r3.json", r3_receipts)


if __name__ == "__main__":
    main()
