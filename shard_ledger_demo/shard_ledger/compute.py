"""Deterministic integer fixed-point aggregation with NumPy.

All numeric parameters are integers. Values are generated on a fixed PCG64
stream keyed by (node seed, shard identity, effective parameter set) and every
intermediate value is rounded to int64, so identical inputs always produce a
byte-identical aggregate summary regardless of platform or scheduling timing.
"""

import hashlib

import numpy as np

BASE_PARAMS = {
    "sample_count": 64,
    "group_count": 4,
    "scale": 1000,
    "shift": 50,
}


def stream_key(node_seed: int, shard_id: str, params: dict) -> int:
    digest = hashlib.sha256(
        f"{node_seed}|{shard_id}|{params['sample_count']}|{params['group_count']}|"
        f"{params['scale']}|{params['shift']}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def effective_params(manifest_revisions: dict, requested_version: int, node_id: str) -> dict:
    params = dict(BASE_PARAMS)
    for version in sorted(manifest_revisions):
        if version > requested_version:
            break
        revision = manifest_revisions[version]
        if node_id in revision.affects:
            params.update(revision.params)
    return params


def aggregate(
    node_id: str,
    shard_id: str,
    node_seed: int,
    requested_version: int,
    manifest_revisions: dict,
) -> dict:
    params = effective_params(manifest_revisions, requested_version, node_id)
    sample_count = params["sample_count"]
    group_count = params["group_count"]
    scale = params["scale"]
    shift = params["shift"]

    rng = np.random.Generator(np.random.PCG64(stream_key(node_seed, shard_id, params)))
    raw = rng.random(sample_count, dtype=np.float64)
    values = np.round(raw * scale + shift).astype(np.int64)
    groups = rng.integers(0, group_count, size=sample_count, dtype=np.int64)

    group_rows = []
    for group in range(group_count):
        mask = groups == group
        members = values[mask]
        count = int(members.size)
        total = int(members.sum(dtype=np.int64))
        minimum = int(members.min()) if count else 0
        maximum = int(members.max()) if count else 0
        checksum = int(np.bitwise_xor.reduce(members)) if count else 0
        group_rows.append(
            {
                "group": group,
                "count": count,
                "sum": total,
                "min": minimum,
                "max": maximum,
                "xor": checksum,
            }
        )

    return {
        "node_id": node_id,
        "shard_id": shard_id,
        "config_version": requested_version,
        "sample_count": sample_count,
        "group_count": group_count,
        "groups": group_rows,
        "total_sum": int(values.sum(dtype=np.int64)),
        "params": dict(sorted(params.items())),
    }
