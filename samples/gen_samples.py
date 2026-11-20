"""Generate deterministic sample declarations and request streams (seeded)."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DIM = 8


def main() -> None:
    rng = np.random.default_rng(20261121)

    backends = {
        "backends": [
            {"model_id": "stable-a", "host": "127.0.0.1", "port": 18081,
             "params": [round(float(x), 6)
                        for x in rng.normal(0, 1, DIM)],
             "max_concurrency": 2, "batch_size": 4},
            {"model_id": "canary-b", "host": "127.0.0.1", "port": 18082,
             "params": [round(float(x), 6)
                        for x in rng.normal(0, 1, DIM)],
             "max_concurrency": 2, "batch_size": 4},
            {"model_id": "stable-c", "host": "127.0.0.1", "port": 18083,
             "params": [round(float(x), 6)
                        for x in rng.normal(0, 1, DIM)],
             "max_concurrency": 3, "batch_size": 4},
            {"model_id": "canary-d", "host": "127.0.0.1", "port": 18084,
             "params": [round(float(x), 6)
                        for x in rng.normal(0, 1, DIM)],
             "max_concurrency": 1, "batch_size": 4},
        ]
    }
    with open(os.path.join(HERE, "backends.json"), "w", encoding="utf-8") as fh:
        json.dump(backends, fh, indent=2, ensure_ascii=False)

    rules = {"rules": [
        {"rule_id": "r-chat", "prefix": "chat:",
         "stable_backend": "stable-a", "canary_backend": "canary-b",
         "canary_weight_bp": 2500, "batch_size": 4},
        {"rule_id": "r-code", "prefix": "code:",
         "stable_backend": "stable-c", "canary_backend": "canary-d",
         "canary_weight_bp": 1000, "batch_size": 4},
    ]}
    with open(os.path.join(HERE, "rules.json"), "w", encoding="utf-8") as fh:
        json.dump(rules, fh, indent=2, ensure_ascii=False)

    # Weight-change variant: canary promoted from 25% to 60% on r-chat.
    rules_v2 = {"rules": [
        dict(rules["rules"][0], canary_weight_bp=6000),
        rules["rules"][1],
    ]}
    with open(os.path.join(HERE, "rules_weight_change.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rules_v2, fh, indent=2, ensure_ascii=False)

    def write_requests(path: str, count: int, clients: int) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for seq in range(count):
                prefix = "chat:" if seq % 3 else "code:"
                record = {
                    "seq": seq,
                    "client_key": f"{prefix}client-{rng.integers(0, clients)}",
                    "inputs": [round(float(x), 6)
                               for x in rng.normal(0, 1, DIM)],
                    "expected_stage": seq // 4,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    write_requests(os.path.join(HERE, "requests_small.ndjson"), 8, 6)
    write_requests(os.path.join(HERE, "requests_large.ndjson"), 96, 48)
    print("samples generated under", HERE)


if __name__ == "__main__":
    main()
