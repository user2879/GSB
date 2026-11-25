"""Deterministic sample generation and integer fixed-point aggregation.

Samples derive solely from a node's integer seed and the shard id, so the
same computation is bit-identical inside a worker subprocess, inside the
scheduler's replay verifier and on an independent reviewer's machine.

Each row holds ``(group_id, raw_value)`` as int64.  Raw values are converted
to the node's fixed-point scale by integer multiplication; every aggregate is
an integer, never a float, which is what makes revocation diffs exact.
"""

import hashlib
import json
import os
from typing import Dict

import numpy as np

GROUP_COUNT = 4
RAW_BOUND = 100_000
_CHECKSUM_MOD = 1 << 63


def derive_seed(node_seed: int, shard_id: str) -> int:
    """Stable 63-bit RNG seed from a node seed and a shard identifier."""
    digest = hashlib.blake2b(f"{node_seed}:{shard_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (1 << 63)


def generate_sample(node_seed: int, shard_id: str, rows: int) -> np.ndarray:
    """Return an int64 ``(rows, 2)`` array of ``(group_id, raw_value)``."""
    seed = derive_seed(node_seed, shard_id)
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, GROUP_COUNT, size=rows, dtype=np.int64)
    values = rng.integers(0, RAW_BOUND, size=rows, dtype=np.int64)
    return np.column_stack((groups, values))


def compute_summary(sample: np.ndarray, node_id: str, shard_id: str, fixed_point_scale: int) -> Dict:
    """Group-aggregate a sample using integer fixed-point arithmetic."""
    if sample.ndim != 2 or sample.shape[1] != 2:
        raise ValueError("sample must be a (rows, 2) int64 array")
    groups = sample[:, 0]
    raw = sample[:, 1]
    fp_values = raw * int(fixed_point_scale)

    group_stats = {}
    checksum = 0
    for gid in range(GROUP_COUNT):
        mask = groups == gid
        count = int(np.count_nonzero(mask))
        if count == 0:
            group_stats[str(gid)] = {"count": 0, "sum_fp": 0, "min_fp": 0, "max_fp": 0}
            continue
        bucket = fp_values[mask]
        entry = {
            "count": count,
            "sum_fp": int(bucket.sum(dtype=np.int64)),
            "min_fp": int(bucket.min()),
            "max_fp": int(bucket.max()),
        }
        group_stats[str(gid)] = entry
        checksum = (checksum + entry["sum_fp"] * 31 + count * 7) % _CHECKSUM_MOD

    summary = {
        "node_id": node_id,
        "shard_id": shard_id,
        "rows": int(sample.shape[0]),
        "fixed_point_scale": int(fixed_point_scale),
        "groups": group_stats,
        "total_sum_fp": int(fp_values.sum(dtype=np.int64)),
        "checksum": int(checksum),
    }
    return summary


def canonical_bytes(summary: Dict) -> bytes:
    """Canonical serialization; identical dicts always produce identical bytes."""
    return json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def summary_digest(summary: Dict) -> str:
    return hashlib.sha256(canonical_bytes(summary)).hexdigest()


def materialize_samples(declaration, samples_dir: str) -> str:
    """Persist every (node, shard) sample plus a JSON manifest.

    Returns the manifest path.  The manifest records the derived seed and row
    count so a reviewer can regenerate and compare files independently.
    """
    os.makedirs(samples_dir, exist_ok=True)
    manifest = {"generated_by": "sharding_diag.sampling", "entries": []}
    for node in sorted(declaration.nodes, key=lambda n: n.node_id):
        for shard in sorted(declaration.shards, key=lambda s: s.shard_id):
            sample = generate_sample(node.seed, shard.shard_id, shard.rows)
            name = f"{node.node_id}__{shard.shard_id}.npz"
            path = os.path.join(samples_dir, name)
            np.savez_compressed(path, sample=sample)
            manifest["entries"].append({
                "file": name,
                "node_id": node.node_id,
                "shard_id": shard.shard_id,
                "rows": shard.rows,
                "derived_seed": derive_seed(node.seed, shard.shard_id),
                "digest": hashlib.sha256(sample.tobytes()).hexdigest(),
            })
    manifest_path = os.path.join(samples_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
    return manifest_path


def load_sample(samples_dir: str, node_id: str, shard_id: str) -> np.ndarray:
    path = os.path.join(samples_dir, f"{node_id}__{shard_id}.npz")
    with np.load(path) as bundle:
        return bundle["sample"].copy()
