"""Content-addressed stage cache and deterministic reuse/recompute audit."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .config import SCHEME_RANK
from .util import atomic_write_json, hash_object


CACHE_SCHEMA_VERSION = 1
STAGE_RANK = {"encoding": 0, "scoring": 1, "tracing": 2}


def stage_cache_key(
    *,
    stage: str,
    scheme_parameters: dict[str, Any],
    sample_fingerprint: str,
    precision: str,
    dependency_fingerprint: str | None = None,
) -> tuple[str, dict[str, Any]]:
    material = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "scheme_parameters": scheme_parameters,
        "sample_fingerprint": sample_fingerprint,
        "precision": precision,
    }
    if dependency_fingerprint is not None:
        material["dependency_fingerprint"] = dependency_fingerprint
    return hash_object(material), material


def _identity(stage: str, scheme: str, sample_id: str) -> str:
    return f"{stage}\x1f{scheme}\x1f{sample_id}"


class StageCache:
    def __init__(self, state_dir: str | Path, *, enabled: bool = True) -> None:
        self.state_dir = Path(state_dir)
        self.root = self.state_dir / "cache"
        self.stage_dir = self.root / "stages"
        self.index_path = self.root / "index.json"
        self.enabled = enabled
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.prior_entries: dict[str, dict[str, Any]] = {}
        if self.index_path.exists():
            try:
                loaded = json.loads(self.index_path.read_text(encoding="utf-8"))
                if loaded.get("cache_schema_version") == CACHE_SCHEMA_VERSION:
                    self.prior_entries = dict(loaded.get("entries", {}))
            except (OSError, ValueError, TypeError):
                self.prior_entries = {}
        self.current_entries: dict[str, dict[str, Any]] = {}
        self.reused: dict[str, dict[str, Any]] = {}
        self.recomputed: dict[str, dict[str, Any]] = {}
        self.invalidated: dict[str, dict[str, Any]] = {}
        self.bypassed: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.stage_dir / f"{key}.json"

    def load(
        self,
        *,
        key: str,
        stage: str,
        scheme: str,
        sample_id: str,
        precision: str,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION
            or payload.get("cache_key") != key
            or payload.get("stage") != stage
            or payload.get("scheme") != scheme
            or payload.get("sample_id") != sample_id
            or payload.get("precision") != precision
        ):
            return None
        body = payload.get("payload")
        return body if isinstance(body, dict) else None

    def store(
        self,
        *,
        key: str,
        stage: str,
        scheme: str,
        sample_id: str,
        precision: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        record = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "stage": stage,
            "scheme": scheme,
            "sample_id": sample_id,
            "precision": precision,
            "payload": payload,
        }
        with self._lock:
            atomic_write_json(self._path(key), record)

    def _meta(
        self,
        *,
        key: str,
        stage: str,
        scheme: str,
        sample_id: str,
        precision: str,
        material: dict[str, Any],
    ) -> dict[str, Any]:
        scheme_fp = hash_object(material["scheme_parameters"])
        sample_fp = str(material["sample_fingerprint"])
        dependency_fp = material.get("dependency_fingerprint")
        return {
            "stage": stage,
            "scheme": scheme,
            "sample_id": sample_id,
            "precision": precision,
            "cache_key": key,
            "scheme_parameter_fingerprint": scheme_fp,
            "sample_fingerprint": sample_fp,
            "stage_dependency_fingerprint": dependency_fp,
        }

    def _reason(
        self,
        identity: str,
        meta: dict[str, Any],
        *,
        load_failed: bool = False,
    ) -> str:
        if not self.enabled:
            return "cache_disabled"
        prior = self.prior_entries.get(identity)
        if prior is None:
            return "cache_miss"
        if prior.get("precision") != meta["precision"]:
            return "precision_changed"
        if prior.get("scheme_parameter_fingerprint") != meta["scheme_parameter_fingerprint"]:
            return "scheme_parameter_changed"
        if prior.get("sample_fingerprint") != meta["sample_fingerprint"]:
            return "sample_content_changed"
        if prior.get("stage_dependency_fingerprint") != meta["stage_dependency_fingerprint"]:
            return "dependency_invalidated"
        if load_failed and prior.get("cache_key") == meta["cache_key"]:
            return "cache_corrupt"
        return "cache_miss"

    def record_reuse(
        self,
        *,
        key: str,
        stage: str,
        scheme: str,
        sample_id: str,
        precision: str,
        material: dict[str, Any],
    ) -> None:
        identity = _identity(stage, scheme, sample_id)
        meta = self._meta(
            key=key,
            stage=stage,
            scheme=scheme,
            sample_id=sample_id,
            precision=precision,
            material=material,
        )
        self.current_entries[identity] = meta
        self.reused[identity] = dict(meta, disposition="reused")

    def record_recompute(
        self,
        *,
        key: str,
        stage: str,
        scheme: str,
        sample_id: str,
        precision: str,
        material: dict[str, Any],
        load_failed: bool = False,
    ) -> str:
        identity = _identity(stage, scheme, sample_id)
        meta = self._meta(
            key=key,
            stage=stage,
            scheme=scheme,
            sample_id=sample_id,
            precision=precision,
            material=material,
        )
        reason = self._reason(identity, meta, load_failed=load_failed)
        self.current_entries[identity] = meta
        record = dict(meta, disposition="recomputed", reason=reason)
        self.recomputed[identity] = record
        prior = self.prior_entries.get(identity)
        if prior is not None and prior.get("cache_key") != key:
            self.invalidated[identity] = dict(
                meta,
                disposition="invalidated",
                reason=reason,
                previous_cache_key=prior.get("cache_key"),
            )
        return reason

    def record_bypass(
        self,
        *,
        stage: str,
        scheme: str,
        sample_id: str,
        reason: str = "injection_bypass",
    ) -> None:
        identity = _identity(stage, scheme, sample_id)
        self.bypassed[identity] = {
            "stage": stage,
            "scheme": scheme,
            "sample_id": sample_id,
            "disposition": "bypassed",
            "reason": reason,
        }

    def _sorted(self, values: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            values.values(),
            key=lambda item: (
                STAGE_RANK.get(str(item["stage"]), 99),
                SCHEME_RANK.get(str(item["scheme"]), 99),
                str(item["sample_id"]),
            ),
        )

    def write_index(self) -> None:
        if not self.enabled:
            return
        entries = dict(self.prior_entries)
        entries.update(self.current_entries)
        index = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "entries": {key: entries[key] for key in sorted(entries)},
        }
        atomic_write_json(self.index_path, index)

    def audit(self) -> dict[str, Any]:
        return {
            "reused": self._sorted(self.reused),
            "recomputed": self._sorted(self.recomputed),
            "invalidated": self._sorted(self.invalidated),
            "bypassed": self._sorted(self.bypassed),
        }
