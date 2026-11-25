"""Generate deterministic local sample declarations, requests and receipts."""
from __future__ import annotations

import json
import os

from grayroute.batching import build_batch_context, plan_batches
from grayroute.canonical import short_hash
from grayroute.receipt import encode_receipt, normalize_raw
from grayroute.sharding import attribute_all
from grayroute.spec import load_spec

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(ROOT, "config")
DATA = os.path.join(ROOT, "data")
MODEL_DIM = 8
HIDDEN_DIM = 16


def summary(seed: str) -> str:
    return short_hash(f"param|{seed}", 16)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    os.makedirs(CONFIG, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    backends = [
        {"backend_id": "stable-a", "host": "127.0.0.1", "port": 18091,
         "model_dim": MODEL_DIM, "hidden_dim": HIDDEN_DIM,
         "quantization": "int8", "param_format": "r2",
         "param_summary": summary("stable-v1"), "role": "stable"},
        {"backend_id": "stable-b", "host": "127.0.0.1", "port": 18092,
         "model_dim": MODEL_DIM, "hidden_dim": HIDDEN_DIM,
         "quantization": "int8", "param_format": "r2",
         "param_summary": summary("stable-v1"), "role": "stable"},
        {"backend_id": "gray-a", "host": "127.0.0.1", "port": 18093,
         "model_dim": MODEL_DIM, "hidden_dim": HIDDEN_DIM,
         "quantization": "int4", "param_format": "r3",
         "param_summary": summary("gray-v2"), "role": "gray"},
        {"backend_id": "gray-b", "host": "127.0.0.1", "port": 18094,
         "model_dim": MODEL_DIM, "hidden_dim": HIDDEN_DIM,
         "quantization": "int4", "param_format": "r3",
         "param_summary": summary("gray-v3"), "role": "gray"},
    ]
    write_json(os.path.join(CONFIG, "backends.json"), backends)

    rules = [
        {"rule_id": "route-main", "revisions": [
            {"revision": 1, "weight_permyriad": 3000, "batch_size": 3,
             "param_version_seq": 1, "param_summary": summary("rule-v1"),
             "backend_ids": ["stable-a", "stable-b", "gray-a"]},
            {"revision": 2, "weight_permyriad": 7000, "batch_size": 2,
             "param_version_seq": 2, "param_summary": summary("rule-v2"),
             "backend_ids": ["stable-a", "stable-b", "gray-b"]},
        ]},
    ]
    write_json(os.path.join(CONFIG, "rules.json"), rules)
    write_json(os.path.join(CONFIG, "merge_rules.json"), {
        "order": ["param_version_seq", "format_version", "merge_seq"],
        "direction": ["desc", "desc", "asc"],
    })

    # 12 requests; first half on revision 1, second half on revision 2
    requests = []
    for index in range(12):
        revision = 1 if index < 6 else 2
        vector = [round((index * 7 + k * 3) % 11 / 10.0, 4)
                  for k in range(MODEL_DIM)]
        requests.append({
            "request_id": f"req-{index:03d}",
            "arrival_seq": index,
            "client_key": f"client-{index % 5}",
            "rule_id": "route-main",
            "rule_revision": revision,
            "vector": vector,
        })
    with open(os.path.join(DATA, "requests.ndjson"), "w",
              encoding="utf-8") as handle:
        for row in requests:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # invalidation fixture: worker for gray-b param version 2 gets invalidated
    write_json(os.path.join(DATA, "invalidate.json"), [
        {"worker_id": "worker-gray-b", "param_version_seq": 2},
    ])
    print("samples generated")


if __name__ == "__main__":
    main()
