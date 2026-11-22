from __future__ import annotations

import base64
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from .util import atomic_write_json, read_json, sha256_bytes, sha256_json


class CacheStore:
    """Content-addressed cache for normalized documents, blocks, encodings and queries.

    All writes are staged in a sibling temporary directory and committed with an
    atomic rename.  Cache artifacts are immutable once visible.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.metrics: dict[str, int] = {
            "norm_lookups": 0,
            "norm_hits": 0,
            "block_lookups": 0,
            "block_hits": 0,
            "encoding_lookups": 0,
            "encoding_hits": 0,
            "encoding_calls": 0,
            "query_lookups": 0,
            "query_hits": 0,
        }

    def _bump(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self.metrics[key] = self.metrics.get(key, 0) + amount

    def snapshot_metrics(self) -> dict[str, int]:
        with self._lock:
            return dict(self.metrics)

    def increment_encoding_calls(self) -> None:
        self._bump("encoding_calls")

    @staticmethod
    def _check_key(raw_key: str) -> str:
        if len(raw_key) != 64 or any(char not in "0123456789abcdef" for char in raw_key):
            raise ValueError("cache key must be a lowercase sha256 hex digest")
        return raw_key

    def _read_json_artifact(self, path: Path, key: str, kind: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("cache_key") != key or payload.get("kind") != kind:
            return None
        return payload

    def lookup_normalized(self, key: str) -> dict[str, Any] | None:
        key = self._check_key(key)
        self._bump("norm_lookups")
        payload = self._read_json_artifact(self.root / "norm" / f"{key}.json", key, "normalized_document")
        if payload is not None:
            self._bump("norm_hits")
        return payload

    def store_normalized(self, key: str, payload: dict[str, Any]) -> None:
        key = self._check_key(key)
        path = self.root / "norm" / f"{key}.json"
        artifact = {"schema": 1, "kind": "normalized_document", "cache_key": key, **payload}
        atomic_write_json(path, artifact)

    def lookup_block(self, key: str) -> dict[str, Any] | None:
        key = self._check_key(key)
        self._bump("block_lookups")
        payload = self._read_json_artifact(self.root / "blocks" / f"{key}.json", key, "inverted_block")
        if payload is not None:
            self._bump("block_hits")
        return payload

    def store_block(self, key: str, payload: dict[str, Any]) -> None:
        key = self._check_key(key)
        path = self.root / "blocks" / f"{key}.json"
        artifact = {"schema": 1, "kind": "inverted_block", "cache_key": key, **payload}
        atomic_write_json(path, artifact)

    def lookup_encoding(self, key: str) -> tuple[dict[str, Any], bytes] | None:
        key = self._check_key(key)
        self._bump("encoding_lookups")
        metadata_path = self.root / "compressed" / f"{key}.json"
        binary_path = self.root / "compressed" / f"{key}.bin"
        payload = self._read_json_artifact(metadata_path, key, "compressed_block")
        if payload is None or not binary_path.exists():
            return None
        try:
            data = binary_path.read_bytes()
        except OSError:
            return None
        if payload.get("encoded_bytes") != len(data):
            return None
        if payload.get("binary_sha256") != sha256_bytes(data):
            return None
        self._bump("encoding_hits")
        return payload, data

    def store_encoding(self, key: str, metadata: dict[str, Any], data: bytes) -> None:
        key = self._check_key(key)
        directory = self.root / "compressed"
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / f"{key}.json"
        binary_path = directory / f"{key}.bin"
        artifact = {
            "schema": 1,
            "kind": "compressed_block",
            "cache_key": key,
            **metadata,
            "encoded_bytes": len(data),
            "binary_sha256": sha256_bytes(data),
        }
        if metadata_path.exists() and binary_path.exists():
            return
        temp_dir = directory / f".{key}.{uuid.uuid4().hex}.tmp"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            (temp_dir / f"{key}.bin").write_bytes(data)
            (temp_dir / f"{key}.json").write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for suffix in ("bin", "json"):
                destination = directory / f"{key}.{suffix}"
                source = temp_dir / f"{key}.{suffix}"
                if not destination.exists():
                    os.replace(source, destination)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def lookup_query(self, key: str) -> dict[str, Any] | None:
        key = self._check_key(key)
        self._bump("query_lookups")
        payload = self._read_json_artifact(self.root / "queries" / f"{key}.json", key, "query_result")
        if payload is not None:
            self._bump("query_hits")
        return payload

    def store_query(self, key: str, payload: dict[str, Any]) -> None:
        key = self._check_key(key)
        path = self.root / "queries" / f"{key}.json"
        artifact = {"schema": 1, "kind": "query_result", "cache_key": key, **payload}
        atomic_write_json(path, artifact)

    @staticmethod
    def normalized_key(raw_fingerprint: str, vocabulary_fingerprint: str, normalization_version: str) -> str:
        return sha256_json(
            {
                "kind": "normalization",
                "raw_fingerprint": raw_fingerprint,
                "vocabulary_fingerprint": vocabulary_fingerprint,
                "normalization_version": normalization_version,
            }
        )

    @staticmethod
    def block_key(shard: str, content_fingerprint: str, index_config_hash: str) -> str:
        return sha256_json(
            {
                "kind": "inverted_block",
                "schema": 1,
                "shard": shard,
                "content_fingerprint": content_fingerprint,
                "index_config_hash": index_config_hash,
            }
        )

    @staticmethod
    def encoding_key(
        block_key: str,
        block_fingerprint: str,
        tier: int,
        encoding_parameters_hash: str,
    ) -> str:
        return sha256_json(
            {
                "kind": "compressed_block",
                "schema": 1,
                "block_key": block_key,
                "block_fingerprint": block_fingerprint,
                "tier": tier,
                "encoding_parameters_hash": encoding_parameters_hash,
            }
        )

    @staticmethod
    def query_key(
        query_payload: dict[str, Any],
        referenced_blocks: list[dict[str, Any]],
        query_config_hash: str,
        document_lengths: dict[str, int] | None = None,
    ) -> str:
        return sha256_json(
            {
                "kind": "query_result",
                "schema": 1,
                "query": query_payload,
                "referenced_blocks": referenced_blocks,
                "document_lengths": document_lengths or {},
                "query_config_hash": query_config_hash,
            }
        )


def base64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def base64_decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)
