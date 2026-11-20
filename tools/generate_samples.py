"""Generate the deterministic local sample set (no network access)."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    config = {
        "version": 1,
        "borrow_ttl_ms": 1500,
        "queue_timeout_ms": 5000,
        "main_timeout_ms": 1200,
        "mirror_timeout_ms": 450,
        "pools": [
            {
                "pool_id": "stable",
                "host": "127.0.0.1",
                "port": 19081,
                "max_connections": 3,
                "max_requests_per_connection": 4,
                "idle_timeout_ms": 1200,
                "lendable_to": ["gray"],
            },
            {
                "pool_id": "gray",
                "host": "127.0.0.1",
                "port": 19082,
                "max_connections": 2,
                "max_requests_per_connection": 4,
                "idle_timeout_ms": 1200,
                "lendable_to": [],
            },
            {
                "pool_id": "mirror",
                "host": "127.0.0.1",
                "port": 19083,
                "max_connections": 2,
                "max_requests_per_connection": 3,
                "idle_timeout_ms": 900,
                "lendable_to": [],
            },
        ],
        "rules": [
            {
                "rule_id": "catalog",
                "prefix": "/api/catalog",
                "stable_pool": "stable",
                "gray_pool": "gray",
                "mirror_pool": "mirror",
                "gray_weight_bp": 6500,
                "mirror_sample_bp": 3500,
            },
            {
                "rule_id": "checkout",
                "prefix": "/api/checkout",
                "stable_pool": "stable",
                "gray_pool": "gray",
                "mirror_pool": "mirror",
                "gray_weight_bp": 3000,
                "mirror_sample_bp": 1500,
            },
        ],
    }
    (SAMPLES / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pools_document = {
        key: value
        for key, value in config.items()
        if key != "rules"
    }
    (SAMPLES / "pools.json").write_text(
        json.dumps(pools_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (SAMPLES / "rules.json").write_text(
        json.dumps({"rules": config["rules"]}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    client_keys = [
        "client-alpha",
        "client-beta",
        "client-gamma",
        "client-delta",
        "client-epsilon",
        "client-zeta",
        "client-eta",
        "client-theta",
        "client-iota",
        "client-kappa",
        "client-lambda",
        "client-mu",
    ]
    paths = ["/api/catalog/items", "/api/catalog/search", "/api/checkout/cart", "/api/checkout/order"]
    lines = []
    for index in range(36):
        request = {
            "request_id": f"sample-{index + 1:03d}",
            "path": f"{paths[index % len(paths)]}/{index + 1}",
            "client_key": client_keys[(index * 5 + 2) % len(client_keys)],
            "timestamp_ms": 1_760_000_000_000 + index * 7,
        }
        lines.append(json.dumps(request, ensure_ascii=False, sort_keys=True))
    (SAMPLES / "requests.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "snapshots").mkdir(exist_ok=True)
    print(f"wrote deterministic samples under {SAMPLES}")


if __name__ == "__main__":
    main()
