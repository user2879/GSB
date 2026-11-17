"""Deterministic local sample generation (no network).

Running this module twice produces byte identical sample files. NumPy is used
with a fixed seed purely to make the synthetic stream reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20261118
STREAM_ROWS = 200

RULES = [
    {
        "rule_id": "rule-orders",
        "prefix": "/api/orders",
        "stable_target": "orders-v1",
        "gray_target": "orders-v2-canary",
        "gray_weight": 3333,
        "mirror_rate": 2500,
        "gray_retries": 2,
        "mirror_timeout_ms": 800,
        "mirror_target": "orders-mirror",
    },
    {
        "rule_id": "rule-users",
        "prefix": "/api/users",
        "stable_target": "users-v1",
        "gray_target": "users-v2-canary",
        "gray_weight": 1111,
        "mirror_rate": 700,
        "gray_retries": 1,
        "mirror_timeout_ms": 500,
        "mirror_target": "users-mirror",
    },
    {
        "rule_id": "rule-orders-deep",
        "prefix": "/api/orders/refund",
        "stable_target": "refund-v1",
        "gray_target": "refund-v2-canary",
        "gray_weight": 600,
        "mirror_rate": 9000,
        "gray_retries": 0,
        "mirror_timeout_ms": 1200,
        "mirror_target": "refund-mirror",
    },
]

# Exponents cover the full supported range; k=0 and k=1 demonstrate visible
# truncation while k>=4 is exact for every integer per-10000 weight.
SCALE_EXPONENTS = [0, 1, 2, 4, 6]

PATHS = [
    "/api/orders", "/api/orders/123", "/api/orders/refund",
    "/api/orders/refund/9", "/api/users", "/api/users/profile",
]


def generate(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    (base / "snapshots").mkdir(exist_ok=True)
    (base / "logs").mkdir(exist_ok=True)

    rule_path = base / "rules.json"
    rule_path.write_text(
        json.dumps({"rules": RULES}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    rng = np.random.default_rng(SEED)
    client_keys = [f"client-{i:03d}" for i in range(40)]
    rows: list[str] = []
    timestamp = 1_700_000_000_000
    for index in range(STREAM_ROWS):
        path = PATHS[int(rng.integers(0, len(PATHS)))]
        client = client_keys[int(rng.integers(0, len(client_keys)))]
        timestamp += int(rng.integers(1, 25))  # strictly non decreasing
        row = {
            "request_id": f"req-{index:05d}",
            "path": path,
            "client_key": client,
            "timestamp_ms": timestamp,
        }
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    stream_path = base / "stream.ndjson"
    stream_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    grid_path = base / "scale_exponents.json"
    grid_path.write_text(
        json.dumps({"scale_exponents": SCALE_EXPONENTS}, indent=2) + "\n",
        encoding="utf-8")
    return {"rules": rule_path, "stream": stream_path, "grid": grid_path}


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "samples"
    paths = generate(target)
    for name, path in paths.items():
        print(f"{name}: {path}")
