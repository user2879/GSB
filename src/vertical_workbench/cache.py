from __future__ import annotations

import threading
from typing import Any

from .hashing import fingerprint


def stage_key(stage: str, **parts: Any) -> str:
    return fingerprint({"stage": stage, **parts})


class CacheStore:
    """Thread-safe in-memory content-addressed stage cache."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._values: dict[str, dict[str, Any]] = {}

    def get(self, stage: str, key: str) -> Any | None:
        with self._lock:
            bucket = self._values.get(stage, {})
            return bucket.get(key)

    def put(self, stage: str, key: str, value: Any) -> None:
        with self._lock:
            self._values.setdefault(stage, {})[key] = value

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {stage: len(bucket) for stage, bucket in sorted(self._values.items())}


def window_key(
    image_fingerprint: str,
    rect: dict[str, int],
    kernel_hash: str,
    quant_fingerprint: dict[str, Any],
    lease_generations: dict[str, int],
) -> str:
    return stage_key(
        "window",
        image=image_fingerprint,
        rect={k: rect[k] for k in ("x", "y", "w", "h")},
        kernel=kernel_hash,
        quant=quant_fingerprint,
        leases=lease_generations,
    )


def column_cc_key(
    image_fingerprint: str,
    column_rank: int,
    dependency_window_keys: list[str],
    kernel_hash: str,
    quant_fingerprint: dict[str, Any],
    lease_generation: int,
) -> str:
    return stage_key(
        "connected_component_column",
        image=image_fingerprint,
        column=column_rank,
        windows=sorted(dependency_window_keys),
        kernel=kernel_hash,
        quant=quant_fingerprint,
        lease_generation=lease_generation,
    )


def domain_key(
    image_fingerprint: str,
    domain_mask_fingerprint: str,
    kernel_hash: str,
    quant_fingerprint: dict[str, Any],
    lease_generation: int,
) -> str:
    return stage_key(
        "connected_domain",
        image=image_fingerprint,
        mask=domain_mask_fingerprint,
        kernel=kernel_hash,
        quant=quant_fingerprint,
        lease_generation=lease_generation,
    )


def keypoint_key(
    image_fingerprint: str,
    domain_cache_key: str,
    kernel_hash: str,
    quant_fingerprint: dict[str, Any],
    lease_generation: int,
    annotation_digest: str,
) -> str:
    return stage_key(
        "keypoint",
        image=image_fingerprint,
        domain=domain_cache_key,
        kernel=kernel_hash,
        quant=quant_fingerprint,
        lease_generation=lease_generation,
        annotation=annotation_digest,
    )


def reading_key(
    image_fingerprint: str,
    keypoint_keys: list[str],
    truth_digest: str,
    annotation_digest: str,
    quant_fingerprint: dict[str, Any],
    lease_generations: dict[str, int],
) -> str:
    return stage_key(
        "reading_order",
        image=image_fingerprint,
        keypoints=keypoint_keys,
        truth=truth_digest,
        annotation=annotation_digest,
        quant=quant_fingerprint,
        leases=lease_generations,
    )
