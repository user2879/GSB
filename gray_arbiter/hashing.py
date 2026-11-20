"""Deterministic, platform-independent hash bucketing."""

from __future__ import annotations

import hashlib


BUCKET_SPACE = 10001


def stable_bucket(*parts: str, bucket_space: int = BUCKET_SPACE) -> int:
    """Return a stable integer in ``[0, bucket_space)``.

    SHA-256 is used instead of Python's process-randomized ``hash`` so a
    Windows run and a replay on another Python build produce the same route.
    """

    if bucket_space <= 0:
        raise ValueError("bucket_space must be positive")
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        if index:
            digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    value = int.from_bytes(digest.digest()[:8], "big")
    return value % bucket_space


def gray_destination(client_key: str, rule_id: str, gray_weight_bp: int) -> tuple[str, int]:
    """Return ``("gray"|"stable", bucket)`` for one routing decision."""

    bucket = stable_bucket(client_key, rule_id)
    if gray_weight_bp <= 0:
        return ("stable", bucket)
    if gray_weight_bp >= 10000:
        return ("gray", bucket)
    return ("gray" if bucket < gray_weight_bp else "stable", bucket)


def mirror_sampled(request_id: str, rule_id: str, sample_rate_bp: int) -> tuple[bool, int]:
    bucket = stable_bucket(request_id, rule_id)
    if sample_rate_bp <= 0:
        return (False, bucket)
    if sample_rate_bp >= 10000:
        return (True, bucket)
    return (bucket < sample_rate_bp, bucket)
