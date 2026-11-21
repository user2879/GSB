"""Content fingerprints used by the incremental execution engine."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .io_utils import sha256_json


FINGERPRINT_SCHEMA = 1


def candidate_fingerprint(
    *,
    channel: str,
    candidate_id: str,
    fixed: int,
    feature_dim: int,
) -> str:
    """Fingerprint a quantized channel candidate.

    The documented content tuple is ``channel``, ``candidate_id``, ``fixed``
    and ``feature_dim``.  Candidate identity is included to keep the
    fingerprint unique inside a channel even when two candidates quantize to
    the same integer.
    """

    return sha256_json(
        {
            "candidate_id": candidate_id,
            "channel": channel,
            "feature_dim": int(feature_dim),
            "fixed": int(fixed),
            "schema": FINGERPRINT_SCHEMA,
        }
    )


def fusion_fingerprint(
    *,
    candidate_id: str,
    candidate_fingerprints: Sequence[str],
    channel_weights: Iterable[tuple[str, float]],
    trunc_k: int,
) -> str:
    return sha256_json(
        {
            "candidate_fingerprints": list(candidate_fingerprints),
            "candidate_id": candidate_id,
            "channel_weights": [[name, float(weight)] for name, weight in channel_weights],
            "schema": FINGERPRINT_SCHEMA,
            "trunc_k": int(trunc_k),
        }
    )


def fragment_set_hash(fragments: Iterable[dict[str, Any]]) -> str:
    return sha256_json(
        [
            {"fragment": item["id"], "sha256": item["sha256"]}
            for item in sorted(fragments, key=lambda value: value["id"])
        ]
    )
