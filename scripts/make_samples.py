from __future__ import annotations

import json
from pathlib import Path


MiB = 1024 * 1024


def object_entry(
    key: str,
    total_bytes: int,
    shard_size: int,
    *,
    concurrency: int,
    lease_ms: int,
    renew_ms: int,
    seed_base: int,
) -> dict:
    shard_count = (total_bytes + shard_size - 1) // shard_size
    return {
        "object_key": key,
        "total_bytes": total_bytes,
        "shard_size": shard_size,
        "shard_count": shard_count,
        "shard_seeds": [seed_base + index * 17 for index in range(shard_count)],
        "concurrency_limit": concurrency,
        "lease_policy": {
            "lease_ms": lease_ms,
            "renew_interval_ms": renew_ms,
            "max_concurrent_leases": concurrency,
        },
        "batch_size": 1024 * 1024 if shard_size >= 1024 * 1024 else 64 * 1024,
    }


def small_spec() -> dict:
    return {
        "session": {
            "global_lease_quota": 3,
            "throughput_budget_bytes": 2 * MiB,
            "scale_exponent": 6,
            "tie_threshold_work": 32,
            "worker_threads": 3,
            "batch_size": 64 * 1024,
        },
        "objects": [
            object_entry("demo/alpha.bin", 256 * 1024, 64 * 1024, concurrency=2, lease_ms=500, renew_ms=100, seed_base=1001),
            object_entry("demo/beta.bin", 256 * 1024, 64 * 1024, concurrency=2, lease_ms=500, renew_ms=100, seed_base=2001),
        ],
    }


def demo_spec() -> dict:
    return {
        "session": {
            "global_lease_quota": 8,
            "throughput_budget_bytes": 320 * MiB,
            "scale_exponent": 6,
            "tie_threshold_work": 32,
            "worker_threads": 4,
            "batch_size": 1024 * 1024,
        },
        "objects": [
            object_entry("scale/tiny-1m.bin", 1 * MiB, 1 * MiB, concurrency=1, lease_ms=2000, renew_ms=400, seed_base=10_001),
            object_entry("scale/medium-32m.bin", 32 * MiB, 2 * MiB, concurrency=3, lease_ms=2000, renew_ms=400, seed_base=20_001),
            object_entry("scale/large-256m.bin", 256 * MiB, 8 * MiB, concurrency=4, lease_ms=5000, renew_ms=1000, seed_base=30_001),
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = root / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for name, value in (("small_spec.json", small_spec()), ("demo_spec.json", demo_spec())):
        (samples / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {samples / name}")


if __name__ == "__main__":
    main()
