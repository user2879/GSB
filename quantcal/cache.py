"""Content-addressed per-tensor stage cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import QuantCalError
from .util import atomic_write_json, canonical_json_bytes, fingerprint, load_json, sha256_bytes


def make_cache_key(
    stage: str,
    upstream_content_fingerprint: str,
    calibration_parameter_fingerprint: str,
    exchange_version: str,
) -> str:
    return fingerprint(
        {
            "schema": "quantcal.cache-key.v2",
            "stage": stage,
            "upstream_content_fingerprint": upstream_content_fingerprint,
            "calibration_parameter_fingerprint": calibration_parameter_fingerprint,
            "exchange_version": exchange_version,
        }
    )


def content_fingerprint(stage: str, payload: dict[str, Any]) -> str:
    return fingerprint({"schema": "quantcal.stage-result.v2", "stage": stage, "payload": payload})


@dataclass
class CacheLookup:
    payload: dict[str, Any] | None
    key: str
    status: str
    path: str | None = None


class StageCache:
    def __init__(self, work_dir: str | Path, enabled: bool = True) -> None:
        self.root = Path(work_dir) / "state" / "cache"
        self.enabled = enabled

    def lookup(self, stage: str, key: str) -> CacheLookup:
        path = self.root / stage / f"{key}.json"
        if not self.enabled:
            return CacheLookup(None, key, "computed")
        if not path.exists():
            return CacheLookup(None, key, "computed", str(path))
        try:
            record = load_json(path)
        except QuantCalError:
            return CacheLookup(None, key, "computed", str(path))
        if not isinstance(record, dict) or record.get("format") != "quantcal.cache-entry.v1":
            return CacheLookup(None, key, "computed", str(path))
        if record.get("key") != key or record.get("stage") != stage:
            return CacheLookup(None, key, "computed", str(path))
        if not self.verify_record_hash(record):
            return CacheLookup(None, key, "computed", str(path))
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return CacheLookup(None, key, "computed", str(path))
        expected = record.get("content_fingerprint")
        actual = content_fingerprint(stage, payload)
        if expected != actual:
            return CacheLookup(None, key, "computed", str(path))
        return CacheLookup(payload, key, "reused", str(path))

    def store(self, stage: str, key: str, payload: dict[str, Any]) -> str:
        content_hash = content_fingerprint(stage, payload)
        record = {
            "format": "quantcal.cache-entry.v1",
            "stage": stage,
            "key": key,
            "content_fingerprint": content_hash,
            "payload": payload,
        }
        path = self.root / stage / f"{key}.json"
        trailing_newline = canonical_json_bytes(record)
        record_hash = sha256_bytes(trailing_newline)
        # The hash guard catches accidental truncation or hand edits of cache JSON.
        record["record_hash"] = record_hash
        if self.enabled:
            atomic_write_json(path, record)
        return content_hash

    def verify_record_hash(self, record: dict[str, Any]) -> bool:
        expected = record.get("record_hash")
        body = {key: value for key, value in record.items() if key != "record_hash"}
        return expected == sha256_bytes(canonical_json_bytes(body))
