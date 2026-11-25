"""Persistent exact-content cache for bucket histories and downstream products."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import (
    canonical_json,
    fingerprint,
    read_json,
    sha256_bytes,
    sha256_text,
)


SCHEMA_VERSION = 1


def bucket_content_fingerprint(record_fingerprints: dict[str, str]) -> str:
    ordered = {key: record_fingerprints[key] for key in sorted(record_fingerprints)}
    return fingerprint({"schema": "lds-bucket-content-v1", "records": ordered})


def bucket_history_cache_key(
    *,
    bucket_id: str,
    content_fingerprint: str,
    params_fingerprint: str,
    random_source_digest: str,
    profile: str,
    budget_tier: int,
) -> str:
    return fingerprint(
        {
            "schema": "lds-bucket-history-key-v1",
            "bucket_id": bucket_id,
            "content_fingerprint": content_fingerprint,
            "params_fingerprint": params_fingerprint,
            "random_source_digest": random_source_digest,
            "profile": profile,
            "budget_tier": budget_tier,
        }
    )


def final_sample_cache_key(
    *,
    bucket_keys: dict[str, str],
    global_budget: int,
    priorities: dict[str, int],
    quantization: dict[str, Any],
) -> str:
    return fingerprint(
        {
            "schema": "lds-final-sample-key-v1",
            "bucket_keys": {
                bucket_id: bucket_keys[bucket_id]
                for bucket_id in sorted(bucket_keys)
            },
            "global_budget": global_budget,
            "priorities": {
                bucket_id: priorities[bucket_id]
                for bucket_id in sorted(priorities)
            },
            "quantization": {
                "decimals": int(quantization.get("decimals", 3)),
                "tier": str(quantization.get("tier", "fine")),
            },
        }
    )


def evidence_cache_key(final_key: str, version: str) -> str:
    return fingerprint(
        {
            "schema": "lds-evidence-key-v1",
            "final_key": final_key,
            "version": version,
        }
    )


class CacheStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.bucket_dir = self.root / "buckets"
        self.final_dir = self.root / "final"
        self.evidence_dir = self.root / "evidence"
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        if kind == "bucket":
            return self.bucket_dir / f"{key}.json"
        if kind == "final":
            return self.final_dir / f"{key}.json"
        raise ValueError(kind)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def load_bucket(self, key: str) -> dict[str, Any] | None:
        value = self._read_json(self._path("bucket", key))
        if not value or value.get("schema_version") != SCHEMA_VERSION:
            return None
        bucket = value.get("bucket")
        return bucket if isinstance(bucket, dict) else None

    def store_bucket(self, key: str, bucket: dict[str, Any]) -> None:
        from .canonical import atomic_write_json

        atomic_write_json(
            self._path("bucket", key),
            {
                "schema_version": SCHEMA_VERSION,
                "key": key,
                "bucket_id": bucket["bucket_id"],
                "bucket": bucket,
            },
        )

    def load_final(self, key: str) -> dict[str, Any] | None:
        value = self._read_json(self._path("final", key))
        if not value or value.get("schema_version") != SCHEMA_VERSION:
            return None
        final = value.get("truncation")
        return final if isinstance(final, dict) else None

    def store_final(self, key: str, truncation: dict[str, Any]) -> None:
        from .canonical import atomic_write_json

        atomic_write_json(
            self._path("final", key),
            {
                "schema_version": SCHEMA_VERSION,
                "key": key,
                "truncation": truncation,
            },
        )

    def _evidence_paths(self, version: str, key: str) -> tuple[Path, Path]:
        directory = self.evidence_dir / version
        return directory / f"{key}.bin", directory / f"{key}.metadata.json"

    def load_evidence(
        self, version: str, key: str
    ) -> tuple[bytes, dict[str, Any]] | None:
        payload_path, metadata_path = self._evidence_paths(version, key)
        if not payload_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = read_json(metadata_path)
            payload = payload_path.read_bytes()
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
            return None
        expected = metadata.get("sha256")
        if expected != sha256_bytes(payload):
            return None
        sidecar = metadata.get("sidecar")
        return payload, sidecar if isinstance(sidecar, dict) else {}

    def store_evidence(
        self, version: str, key: str, payload: bytes, sidecar: dict[str, Any]
    ) -> None:
        from .canonical import atomic_write_bytes, atomic_write_json

        payload_path, metadata_path = self._evidence_paths(version, key)
        atomic_write_bytes(payload_path, payload)
        atomic_write_json(
            metadata_path,
            {
                "schema_version": SCHEMA_VERSION,
                "key": key,
                "version": version,
                "sha256": sha256_bytes(payload),
                "byte_length": len(payload),
                "sidecar": sidecar,
            },
        )

    def load_index(self) -> dict[str, Any]:
        value = self._read_json(self.index_path)
        if not value or value.get("schema_version") != SCHEMA_VERSION:
            return {"buckets": {}}
        buckets = value.get("buckets")
        return {"buckets": buckets if isinstance(buckets, dict) else {}}

    def store_index(self, index: dict[str, Any]) -> None:
        from .canonical import atomic_write_json

        atomic_write_json(
            self.index_path,
            {"schema_version": SCHEMA_VERSION, "buckets": index.get("buckets", {})},
        )


def bucket_invalidation_reason(
    previous: dict[str, Any] | None,
    *,
    current_params_fingerprint: str,
    current_record_fingerprints: dict[str, str],
    current_profile: str,
    current_budget: int,
    cache_entry_found: bool,
) -> tuple[str, list[dict[str, Any]]]:
    """Return a bucket reason plus per-record reasons for a cache miss."""
    if cache_entry_found:
        return "cache_hit", [
            {
                "log_id": log_id,
                "status": "reuse_hit",
                "reason": "bucket_content_and_params_match",
            }
            for log_id in sorted(current_record_fingerprints)
        ]
    if previous is None:
        return "new_bucket", [
            {
                "log_id": log_id,
                "status": "invalidated",
                "reason": "new_bucket",
            }
            for log_id in sorted(current_record_fingerprints)
        ]

    previous_params = previous.get("params_fingerprint")
    previous_profile = previous.get("profile")
    previous_budget = previous.get("budget")
    if previous_profile != current_profile or previous_budget != current_budget:
        reason = "budget_tier_or_profile_changed"
    elif previous_params != current_params_fingerprint:
        reason = "bucket_parameters_changed"
    else:
        reason = "bucket_content_changed"

    previous_records = previous.get("record_fingerprints", {})
    if not isinstance(previous_records, dict):
        previous_records = {}
    record_ledger: list[dict[str, Any]] = []
    all_ids = sorted(set(current_record_fingerprints) | set(previous_records))
    for log_id in all_ids:
        current = current_record_fingerprints.get(log_id)
        prior = previous_records.get(log_id)
        if current is None:
            record_ledger.append(
                {
                    "log_id": log_id,
                    "status": "removed",
                    "reason": "record_deleted",
                }
            )
        elif prior is None:
            record_ledger.append(
                {
                    "log_id": log_id,
                    "status": "invalidated",
                    "reason": "record_added",
                }
            )
        elif prior != current:
            record_ledger.append(
                {
                    "log_id": log_id,
                    "status": "invalidated",
                    "reason": "record_content_changed",
                }
            )
        else:
            record_ledger.append(
                {
                    "log_id": log_id,
                    "status": "invalidated",
                    "reason": reason,
                }
            )
    return reason, record_ledger

