"""Deterministic user bucketing and group mapping.

The bucket hash is BLAKE2b over ``seed|user_id``; the digest is read as a
big-endian integer and reduced modulo the bucket count. Groups receive
contiguous bucket ranges split by declared integer weights, keeping the
assignment fully reproducible.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .errors import E_HASH_CONFIG_INVALID, FailureRecord


@dataclass(frozen=True)
class BucketAssignment:
    user_id: str
    bucket: int
    group: str

    def to_dict(self):
        return {"user_id": self.user_id, "bucket": self.bucket, "group": self.group}


def bucket_for_user(user_id: str, seed, num_buckets: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}|{user_id}".encode("utf-8"), digest_size=16
    ).digest()
    return int.from_bytes(digest, "big") % num_buckets


def validate_bucket_config(config) -> List[FailureRecord]:
    failures: List[FailureRecord] = []
    num_buckets = getattr(config, "num_buckets", None)
    if not isinstance(num_buckets, int) or num_buckets <= 0:
        failures.append(
            FailureRecord(
                stage="bucketing",
                code=E_HASH_CONFIG_INVALID,
                reason="num_buckets must be a positive integer",
                detail=repr(num_buckets),
            )
        )
        return failures
    weight_total = sum(config.group_weights.values())
    if weight_total <= 0:
        failures.append(
            FailureRecord(
                stage="bucketing",
                code=E_HASH_CONFIG_INVALID,
                reason="group weights must sum to a positive integer",
            )
        )
    if any(w <= 0 for w in config.group_weights.values()):
        failures.append(
            FailureRecord(
                stage="bucketing",
                code=E_HASH_CONFIG_INVALID,
                reason="every group weight must be positive",
            )
        )
    if not failures and weight_total > num_buckets:
        failures.append(
            FailureRecord(
                stage="bucketing",
                code=E_HASH_CONFIG_INVALID,
                reason="sum of group weights exceeds num_buckets",
                detail=f"weights={weight_total}, buckets={num_buckets}",
            )
        )
    if not isinstance(config.hash_seed, int) or config.hash_seed < 0:
        failures.append(
            FailureRecord(
                stage="bucketing",
                code=E_HASH_CONFIG_INVALID,
                reason="hash_seed must be a non-negative integer",
            )
        )
    return failures


def group_boundaries(config) -> List[Tuple[str, int, int]]:
    """Return [(group, start_bucket, end_bucket_exclusive), ...] in config order."""
    total = sum(config.group_weights.values())
    boundaries: List[Tuple[str, int, int]] = []
    cursor = 0
    for group in config.group_order:
        share = config.group_weights[group] * config.num_buckets
        if share % total != 0:
            raise ValueError(
                "bucket count is not evenly divisible by group weight shares"
            )
        length = share // total
        boundaries.append((group, cursor, cursor + length))
        cursor += length
    if cursor != config.num_buckets:
        raise ValueError("group boundaries do not cover the full bucket range")
    return boundaries


def assign_users(
    user_ids, config
) -> Tuple[Dict[str, BucketAssignment], List[FailureRecord]]:
    failures = validate_bucket_config(config)
    assignments: Dict[str, BucketAssignment] = {}
    if failures:
        return assignments, failures
    boundaries = group_boundaries(config)

    def _group_of(bucket: int) -> str:
        for group, start, end in boundaries:
            if start <= bucket < end:
                return group
        raise RuntimeError("bucket outside all group ranges")

    for user_id in sorted(user_ids):
        bucket = bucket_for_user(user_id, config.hash_seed, config.num_buckets)
        assignments[user_id] = BucketAssignment(user_id, bucket, _group_of(bucket))
    return assignments, failures
