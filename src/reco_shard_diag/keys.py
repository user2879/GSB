from __future__ import annotations

from typing import Any

from .canonical import fingerprint


def width_profile(active_width: int, reference_width: int) -> dict[str, int]:
    return {"active_width": active_width, "reference_width": reference_width}


def aggregate_key(
    *,
    node_id: str,
    shard_id: str,
    samples_fingerprint: str,
    quota_fingerprint: str,
    width: int,
    lease_generation: int,
    upstream_summary_fingerprints: list[str],
    upstream_artifact_tokens: list[str] | None = None,
) -> str:
    return fingerprint(
        {
            "kind": "node_aggregate",
            "node_id": node_id,
            "shard_id": shard_id,
            "samples_fingerprint": samples_fingerprint,
            "quota_fingerprint": quota_fingerprint,
            "width": width,
            "lease_generation": lease_generation,
            "upstream_summary_fingerprints": sorted(upstream_summary_fingerprints),
            "upstream_artifact_tokens": sorted(upstream_artifact_tokens or []),
        }
    )


def receipt_key(
    *,
    shard_id: str,
    aggregate_summary_fingerprint: str,
    quota_fingerprint: str,
    width: int,
    lease_generation: int,
    upstream_summary_fingerprints: list[str],
    upstream_artifact_tokens: list[str] | None = None,
) -> str:
    return fingerprint(
        {
            "kind": "authoritative_receipt",
            "shard_id": shard_id,
            "aggregate_summary_fingerprint": aggregate_summary_fingerprint,
            "quota_fingerprint": quota_fingerprint,
            "width": width,
            "lease_generation": lease_generation,
            "upstream_summary_fingerprints": sorted(upstream_summary_fingerprints),
            "upstream_artifact_tokens": sorted(upstream_artifact_tokens or []),
        }
    )


def watermark_key(
    *,
    node_id: str,
    shard_id: str,
    quota_fingerprint: str,
    width: int,
    resource_units: int,
    capacity_slots: int,
    lease_generation: int,
) -> str:
    return fingerprint(
        {
            "kind": "watermark_conclusion",
            "node_id": node_id,
            "shard_id": shard_id,
            "quota_fingerprint": quota_fingerprint,
            "width": width,
            "resource_units": resource_units,
            "capacity_slots": capacity_slots,
            "lease_generation": lease_generation,
        }
    )


def readiness_key(
    *,
    shard_id: str,
    upstream: list[str],
    upstream_summary_fingerprints: list[str],
    quota_fingerprint: str,
    lease_generation: int,
    upstream_artifact_tokens: list[str] | None = None,
) -> str:
    return fingerprint(
        {
            "kind": "readiness_conclusion",
            "shard_id": shard_id,
            "upstream": sorted(upstream),
            "upstream_summary_fingerprints": sorted(upstream_summary_fingerprints),
            "upstream_artifact_tokens": sorted(upstream_artifact_tokens or []),
            "quota_fingerprint": quota_fingerprint,
            "lease_generation": lease_generation,
        }
    )


def artifact_record(
    cache_key: str,
    *,
    shard_id: str,
    upstream: list[str],
    quota_version: int,
    quota_fingerprint: str,
    width: int,
    lease_generation: int,
    product: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cache_key": cache_key,
        "shard_id": shard_id,
        "upstream": sorted(upstream),
        "quota_version": quota_version,
        "quota_fingerprint": quota_fingerprint,
        "width": width,
        "lease_generation": lease_generation,
        "product": product,
    }
