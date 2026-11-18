"""Deterministic sample object generation powered by a fixed NumPy seed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .storage import sha256_bytes, staging_dir

SEED = 20261118

# Fixed, hand-authored metadata so access patterns are fully reproducible.
_SAMPLE_SPECS = [
    ("obj-01", 4096, 1_700_000_000_000, 90),
    ("obj-02", 8192, 1_700_000_060_000, 55),
    ("obj-03", 2048, 1_699_999_900_000, 12),
    ("obj-04", 16384, 1_700_000_120_000, 73),
    ("obj-05", 1024, 1_699_999_800_000, 4),
    ("obj-06", 3072, 1_700_000_180_000, 38),
    ("obj-07", 5120, 1_699_999_700_000, 21),
    ("obj-08", 2560, 1_700_000_240_000, 61),
]


def _payload(rng: np.random.Generator, size_bytes: int, key: str) -> bytes:
    # Deterministic pseudo-random bytes; the key salt keeps per-object streams
    # independent while the global seed fixes the whole sample set.
    salt = np.frombuffer(key.encode("utf-8"), dtype=np.uint8).astype(np.uint32)
    stream_seed = int(SEED) ^ int(salt.sum()) ^ size_bytes
    local_rng = np.random.default_rng(stream_seed)
    array = local_rng.integers(0, 256, size=size_bytes, dtype=np.uint8)
    # Mix in the global stream for an extra fixed transformation.
    mix = rng.integers(0, 256, size=size_bytes, dtype=np.uint8)
    return np.bitwise_xor(array, mix).tobytes()


def build_sample(
    out_dir: str | Path,
    *,
    policy_version: str = "policy-v1",
    tier_param_version: str = "tier-v1",
    hot_capacity: int = 16384,
    cold_capacity: int = 65536,
    hot_watermark: int = 12288,
    cold_reclaim: int = 32768,
    batch_size: int = 4,
    hot_cost: int = 5,
    cold_cost: int = 1,
    injection_sequence: List[str] | None = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    staging = staging_dir(out_dir)
    staging.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    objects = []
    for key, size_bytes, last_access_ms, access_count in _SAMPLE_SPECS:
        payload = _payload(rng, size_bytes, key)
        digest = sha256_bytes(payload)
        (staging / digest).write_bytes(payload)
        objects.append(
            {
                "key": key,
                "size_bytes": size_bytes,
                "last_access_ms": last_access_ms,
                "access_count": access_count,
                "digest": digest,
            }
        )

    manifest = {
        "policy_version": policy_version,
        "tier_param_version": tier_param_version,
        "tiers": [
            {
                "tier_id": "hot",
                "role": "hot",
                "capacity_bytes": hot_capacity,
                "unit_cost": hot_cost,
                "read_weight": 2,
                "write_weight": 3,
                "evict_priority": 0,
            },
            {
                "tier_id": "cold",
                "role": "cold",
                "capacity_bytes": cold_capacity,
                "unit_cost": cold_cost,
                "read_weight": 4,
                "write_weight": 2,
                "evict_priority": 100,
            },
        ],
        "objects": objects,
        "rule": {
            "hot_watermark_bytes": hot_watermark,
            "cold_reclaim_bytes": cold_reclaim,
            "batch_size": batch_size,
        },
        "injection_sequence": list(injection_sequence or []),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"manifest_path": str(manifest_path), "staging_dir": str(staging), "objects": objects}
