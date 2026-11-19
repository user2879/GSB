"""Content-addressed stage object cache with atomic replacement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .util import PRODUCT_VERSIONS, atomic_write_json, read_json, sha256_json


STAGE_DIRECTORIES = {
    "policy": "policy",
    "features": "features",
    "state_machine": "state",
    "endpoints": "endpoint",
}


def stage_cache_key(
    stage: str,
    *,
    sample_key: str,
    audio_fingerprint: str,
    config_fingerprint: str,
    profile: str | None,
    upstream_key: str | None = None,
    extras: Any = None,
) -> str:
    material = {
        "schema_version": "1.0",
        "stage": stage,
        "product_version": PRODUCT_VERSIONS[stage],
        "sample_key": sample_key,
        "audio_fingerprint": audio_fingerprint,
        "stage_config_fingerprint": config_fingerprint,
        "profile": profile,
        "upstream_key": upstream_key,
        "extras": extras,
    }
    return sha256_json(material)


class StageCache:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def object_path(self, stage: str, key: str) -> Path:
        try:
            directory = STAGE_DIRECTORIES[stage]
        except KeyError as exc:  # pragma: no cover - programming error
            raise ValueError(f"unknown cache stage: {stage}") from exc
        return self.root / directory / f"{key}.json"

    def read(self, stage: str, key: str) -> tuple[dict[str, Any] | None, str]:
        path = self.object_path(stage, key)
        if not path.exists():
            return None, "cache_object_missing"
        try:
            envelope = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None, "cache_object_invalid_json"
        if not isinstance(envelope, dict):
            return None, "cache_object_invalid_envelope"
        if envelope.get("schema_version") != "1.0":
            return None, "cache_object_schema_mismatch"
        if envelope.get("stage") != stage:
            return None, "cache_object_stage_mismatch"
        if envelope.get("cache_key") != key:
            return None, "cache_object_key_mismatch"
        if envelope.get("product_version") != PRODUCT_VERSIONS[stage]:
            return None, "cache_object_product_version_mismatch"
        result = envelope.get("result")
        if not isinstance(result, dict):
            return None, "cache_object_result_missing"
        return result, "cache_object_match"

    def write(self, stage: str, key: str, result: dict[str, Any]) -> None:
        envelope = {
            "schema_version": "1.0",
            "stage": stage,
            "product_version": PRODUCT_VERSIONS[stage],
            "cache_key": key,
            "result": result,
        }
        atomic_write_json(self.object_path(stage, key), envelope)
