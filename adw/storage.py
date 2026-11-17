"""Atomic JSON I/O, content-addressed cache and stage snapshots."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import canonical_json, jsonable, sha256_text
from .errors import SnapshotError


SCHEMA_VERSION = 1


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(payload) + "\n"
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", suffix="", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    corrupt: int = 0
    writes: int = 0
    by_kind_hits: dict[str, int] = field(default_factory=dict)
    by_kind_misses: dict[str, int] = field(default_factory=dict)

    def hit(self, kind: str) -> None:
        self.hits += 1
        self.by_kind_hits[kind] = self.by_kind_hits.get(kind, 0) + 1

    def miss(self, kind: str) -> None:
        self.misses += 1
        self.by_kind_misses[kind] = self.by_kind_misses.get(kind, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "corrupt": self.corrupt,
            "writes": self.writes,
            "by_kind_hits": dict(sorted(self.by_kind_hits.items())),
            "by_kind_misses": dict(sorted(self.by_kind_misses.items())),
        }


class ContentCache:
    def __init__(self, root: str | Path | None, enabled: bool = True):
        self.root = Path(root).resolve() if root is not None else None
        self.enabled = bool(enabled and self.root is not None)
        self.memory: dict[tuple[str, str], dict[str, Any]] = {}
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        if self.root is None:
            raise RuntimeError("cache is disabled")
        return self.root / kind / key[:2] / f"{key}.json"

    def get(self, kind: str, key: str, stats: CacheStats) -> dict[str, Any] | None:
        if not self.enabled:
            stats.miss(kind)
            return None
        memory_key = (kind, key)
        if memory_key in self.memory:
            stats.hit(kind)
            return self.memory[memory_key]
        path = self._path(kind, key)
        if not path.exists():
            stats.miss(kind)
            return None
        try:
            envelope = read_json(path)
            if (
                not isinstance(envelope, dict)
                or envelope.get("schema_version") != SCHEMA_VERSION
                or envelope.get("kind") != kind
                or envelope.get("key") != key
                or not isinstance(envelope.get("artifact"), dict)
            ):
                raise ValueError("cache envelope mismatch")
            self.memory[memory_key] = envelope["artifact"]
            stats.hit(kind)
            return envelope["artifact"]
        except (OSError, ValueError, json.JSONDecodeError):
            stats.corrupt += 1
            stats.miss(kind)
            return None

    def put(
        self,
        kind: str,
        key: str,
        artifact: dict[str, Any],
        metadata: dict[str, Any],
        stats: CacheStats,
    ) -> None:
        if not self.enabled:
            return
        clean_artifact = jsonable(artifact)
        memory_key = (kind, key)
        self.memory[memory_key] = clean_artifact
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "key": key,
            "metadata": jsonable(metadata),
            "artifact": clean_artifact,
        }
        atomic_write_json(self._path(kind, key), envelope)
        stats.writes += 1

    def has(self, kind: str, key: str) -> bool:
        if not self.enabled:
            return False
        if (kind, key) in self.memory:
            return True
        return self._path(kind, key).exists()


class SnapshotManager:
    STAGES = ("preprocess", "score", "decision")

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.commit_counter = 0

    def _path(self, stage: str) -> Path:
        if stage not in self.STAGES:
            raise ValueError(f"unknown snapshot stage: {stage}")
        return self.directory / f"{stage}.json"

    def write_stage(
        self,
        *,
        stage: str,
        config_hash: str,
        data_fingerprint: str,
        completed_samples: int,
        artifact_version: str,
        payload: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_payload = jsonable(payload)
        payload_hash = sha256_text(canonical_json(clean_payload))
        self.commit_counter += 1
        record = {
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "completed_samples": int(completed_samples),
            "artifact_version": artifact_version,
            "payload_sha256": payload_hash,
            "commit_seq": self.commit_counter,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "payload": clean_payload,
            "extra": jsonable(extra or {}),
        }
        atomic_write_json(self._path(stage), record)
        return {
            "stage": stage,
            "completed_samples": int(completed_samples),
            "artifact_version": artifact_version,
            "payload_sha256": payload_hash,
        }

    def scan(self, data_fingerprint: str | None = None) -> dict[str, Any]:
        """Read only fully replaced, checksum-valid stage files; ignore temp files."""

        valid: dict[str, dict[str, Any]] = {}
        rejected: list[dict[str, Any]] = []
        for stage in self.STAGES:
            path = self._path(stage)
            if not path.exists():
                continue
            try:
                record = read_json(path)
                if not isinstance(record, dict):
                    raise ValueError("snapshot root is not an object")
                if record.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("schema version mismatch")
                if record.get("stage") != stage:
                    raise ValueError("stage mismatch")
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    raise ValueError("payload is missing")
                actual_hash = sha256_text(canonical_json(payload))
                if actual_hash != record.get("payload_sha256"):
                    raise ValueError("payload checksum mismatch")
                for field in ("config_hash", "data_fingerprint", "artifact_version"):
                    if not isinstance(record.get(field), str) or not record[field]:
                        raise ValueError(f"missing {field}")
                if not isinstance(record.get("completed_samples"), int):
                    raise ValueError("completed_samples is missing")
                if data_fingerprint is not None and record.get("data_fingerprint") != data_fingerprint:
                    rejected.append({"stage": stage, "reason": "data fingerprint mismatch"})
                    continue
                valid[stage] = record
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                rejected.append({"stage": stage, "reason": str(exc)})
        # Ignore explicitly incomplete temp files, but report their presence.
        temp_files = sorted(path.name for path in self.directory.glob(".*.tmp.*"))
        return {
            "valid_snapshots": [
                {
                    "stage": stage,
                    "completed_samples": int(record["completed_samples"]),
                    "artifact_version": record["artifact_version"],
                    "config_hash": record["config_hash"],
                    "data_fingerprint": record["data_fingerprint"],
                    "commit_seq": int(record.get("commit_seq", 0)),
                }
                for stage, record in sorted(valid.items(), key=lambda item: self.STAGES.index(item[0]))
            ],
            "rejected": rejected,
            "ignored_temp_files": temp_files,
            "records": valid,
        }

    def recovery_summary(self, data_fingerprint: str | None = None) -> dict[str, Any]:
        scan = self.scan(data_fingerprint)
        return {
            "snapshot_dir": str(self.directory),
            "valid_snapshots": scan["valid_snapshots"],
            "rejected": scan["rejected"],
            "ignored_temp_files": scan["ignored_temp_files"],
            "restored_from_complete_snapshots": bool(scan["valid_snapshots"]),
        }
