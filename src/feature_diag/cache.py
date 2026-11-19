"""Content-addressed artifact cache with a deterministic reuse/invalidation ledger."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .serialization import atomic_write_text, dump_json, from_wire, load_json, to_wire
from .fingerprint import sha256_text

ARTIFACT_RANK = {
    "crossing": 0,
    "standardization": 1,
    "reduction": 2,
    "metrics": 3,
}


class ContentCache:
    def __init__(self, root: str | Path, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.payload_dir = self.root / "payloads"
        self.state_path = self.root / "state.json"
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self._global_lock = threading.RLock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._payload_memory: dict[str, dict[str, Any]] = {}
        self._entries: dict[str, dict[str, Any]] = {}
        self._consumers: dict[str, set[str]] = defaultdict(set)
        self._previous_lineages: dict[str, dict[str, Any]] = {}
        if self.enabled and self.state_path.exists():
            try:
                state = load_json(self.state_path, decode_wire=False)
                lineages = state.get("lineages", {}) if isinstance(state, dict) else {}
                if isinstance(lineages, dict):
                    self._previous_lineages = {
                        str(lineage): value for lineage, value in lineages.items() if isinstance(value, dict)
                    }
            except (OSError, ValueError, json.JSONDecodeError):
                self._previous_lineages = {}

    def _lock_for(self, key: str) -> threading.Lock:
        with self._global_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _payload_path(self, key: str) -> Path:
        return self.payload_dir / f"{key}.json"

    def _read_payload(self, key: str) -> tuple[dict[str, Any] | None, str | None]:
        if not self.enabled:
            return None, "cache_disabled"
        cached = self._payload_memory.get(key)
        if cached is not None:
            return cached, None
        path = self._payload_path(key)
        if not path.exists():
            return None, "payload_missing"
        try:
            wrapper = load_json(path, decode_wire=True)
            if not isinstance(wrapper, dict) or wrapper.get("key") != key:
                return None, "payload_key_mismatch"
            payload = wrapper.get("payload")
            expected = wrapper.get("payload_fingerprint")
            actual = sha256_text(dump_json(payload))
            if expected != actual:
                return None, "payload_fingerprint_mismatch"
            self._payload_memory[key] = payload
            return payload, None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None, "payload_corrupt"

    def _write_payload(
        self,
        *,
        key: str,
        kind: str,
        lineage: str,
        payload: dict[str, Any],
    ) -> str:
        payload_hash = sha256_text(dump_json(payload))
        wrapper = {
            "schema_version": 1,
            "key": key,
            "kind": kind,
            "lineage": lineage,
            "payload_fingerprint": payload_hash,
            "payload": to_wire(payload),
        }
        atomic_write_text(self._payload_path(key), json.dumps(wrapper, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        with self._global_lock:
            self._payload_memory[key] = payload
        return payload_hash

    def _entry_sort(self, entry: dict[str, Any]) -> tuple[Any, ...]:
        return (
            ARTIFACT_RANK.get(entry.get("artifact_kind", ""), 99),
            entry.get("scenario_id") or "",
            int(entry.get("sample_size") or 0),
            int(entry.get("order_limit") or 0),
            entry.get("path") or "",
            entry.get("precision") or "",
            entry.get("lineage") or "",
            entry.get("cache_key") or "",
        )

    def _classify(self, key: str, lineage: str, payload_error: str | None) -> tuple[str, str, str | None]:
        previous = self._previous_lineages.get(lineage)
        if payload_error is None:
            if previous is None:
                return "hit", "content_already_cached", None
            if previous.get("key") == key:
                return "hit", "same_upstream_and_parameters", None
            return "hit", "content_reused_from_earlier_parameterization", str(previous.get("key"))
        if previous is None:
            return "miss", "not_present_in_ledger", None
        if previous.get("key") == key:
            return "invalidated", payload_error, None
        return "invalidated", "upstream_or_parameter_changed", str(previous.get("key"))

    def get_or_compute(
        self,
        *,
        key: str,
        kind: str,
        lineage: str,
        sort_key: tuple[Any, ...],
        upstream_fingerprint: str,
        parameter_fingerprint: str,
        precision: str,
        scenario_id: str,
        sample_size: int,
        order_limit: int,
        path: str | None,
        consumer: str,
        compute: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.enabled:
            payload = compute()
            entry = {
                "cache_key": key,
                "artifact_kind": kind,
                "lineage": lineage,
                "status": "disabled",
                "reason": "cache_disabled",
                "previous_key": None,
                "upstream_fingerprint": upstream_fingerprint,
                "parameter_fingerprint": parameter_fingerprint,
                "precision": precision,
                "scenario_id": scenario_id,
                "sample_size": int(sample_size),
                "order_limit": int(order_limit),
                "path": path,
                "payload_fingerprint": sha256_text(dump_json(payload)),
                "sort_key": list(sort_key),
                "cache_backed": False,
            }
            with self._global_lock:
                self._entries[key] = entry
                self._consumers[key].add(consumer)
            return payload
        lock = self._lock_for(key)
        with lock:
            with self._global_lock:
                existing = self._entries.get(key)
            if existing is not None:
                payload, error = self._read_payload(key)
                if payload is None:
                    payload = compute()
                    self._write_payload(key=key, kind=kind, lineage=lineage, payload=payload)
                with self._global_lock:
                    self._consumers[key].add(consumer)
                return payload

            payload, payload_error = self._read_payload(key)
            status, reason, previous_key = self._classify(key, lineage, payload_error)
            if payload is None:
                payload = compute()
                payload_hash = self._write_payload(key=key, kind=kind, lineage=lineage, payload=payload)
                if previous_key is None and payload_error is None:
                    status, reason = "hit", "content_already_cached"
            else:
                payload_hash = sha256_text(dump_json(payload))
            entry = {
                "cache_key": key,
                "artifact_kind": kind,
                "lineage": lineage,
                "status": status,
                "reason": reason,
                "previous_key": previous_key,
                "upstream_fingerprint": upstream_fingerprint,
                "parameter_fingerprint": parameter_fingerprint,
                "precision": precision,
                "scenario_id": scenario_id,
                "sample_size": int(sample_size),
                "order_limit": int(order_limit),
                "path": path,
                "payload_fingerprint": payload_hash,
                "sort_key": list(sort_key),
                "cache_backed": True,
            }
            with self._global_lock:
                self._entries[key] = entry
                self._consumers[key].add(consumer)
            return payload

    def register_snapshot_reuse(
        self,
        *,
        key: str,
        kind: str,
        lineage: str,
        sort_key: tuple[Any, ...],
        upstream_fingerprint: str,
        parameter_fingerprint: str,
        precision: str,
        scenario_id: str,
        sample_size: int,
        order_limit: int,
        path: str | None,
        payload_fingerprint: str,
        consumer: str,
    ) -> None:
        with self._global_lock:
            if key not in self._entries:
                self._entries[key] = {
                    "cache_key": key,
                    "artifact_kind": kind,
                    "lineage": lineage,
                    "status": "snapshot_reuse",
                    "reason": "complete_snapshot_stage_reused",
                    "previous_key": None,
                    "upstream_fingerprint": upstream_fingerprint,
                    "parameter_fingerprint": parameter_fingerprint,
                    "precision": precision,
                    "scenario_id": scenario_id,
                    "sample_size": int(sample_size),
                    "order_limit": int(order_limit),
                    "path": path,
                    "payload_fingerprint": payload_fingerprint,
                    "sort_key": list(sort_key),
                    "cache_backed": False,
                }
            self._consumers[key].add(consumer)

    def finalize(self, *, persist_state: bool = True) -> dict[str, Any]:
        with self._global_lock:
            current = [dict(entry) for entry in self._entries.values()]
            current_lineages = {entry["lineage"] for entry in current}
            retired: list[dict[str, Any]] = []
            for lineage, previous in self._previous_lineages.items():
                if lineage in current_lineages:
                    continue
                retired.append(
                    {
                        "cache_key": previous.get("key"),
                        "artifact_kind": previous.get("kind"),
                        "lineage": lineage,
                        "status": "retired",
                        "reason": "not_requested_in_current_run",
                        "previous_key": previous.get("key"),
                        "upstream_fingerprint": previous.get("upstream_fingerprint"),
                        "parameter_fingerprint": previous.get("parameter_fingerprint"),
                        "precision": previous.get("precision"),
                        "scenario_id": previous.get("scenario_id"),
                        "sample_size": int(previous.get("sample_size") or 0),
                        "order_limit": int(previous.get("order_limit") or 0),
                        "path": previous.get("path"),
                        "payload_fingerprint": previous.get("payload_fingerprint"),
                        "sort_key": list(previous.get("sort_key") or []),
                        "cache_backed": True,
                    }
                )
            for entry in current:
                entry["consumers"] = sorted(self._consumers.get(entry["cache_key"], set()))
            for entry in retired:
                entry["consumers"] = []
            entries = current + retired
            entries.sort(key=self._entry_sort)
            summary: dict[str, int] = {
                "hit": 0,
                "miss": 0,
                "invalidated": 0,
                "snapshot_reuse": 0,
                "retired": 0,
                "disabled": 0,
            }
            for entry in entries:
                status = entry.get("status")
                if status in summary:
                    summary[status] += 1
            state_lineages: dict[str, Any] = {}
            for entry in current:
                if not entry.get("cache_backed"):
                    continue
                state_lineages[entry["lineage"]] = {
                    "key": entry["cache_key"],
                    "kind": entry["artifact_kind"],
                    "upstream_fingerprint": entry["upstream_fingerprint"],
                    "parameter_fingerprint": entry["parameter_fingerprint"],
                    "precision": entry["precision"],
                    "scenario_id": entry["scenario_id"],
                    "sample_size": entry["sample_size"],
                    "order_limit": entry["order_limit"],
                    "path": entry["path"],
                    "payload_fingerprint": entry["payload_fingerprint"],
                    "sort_key": entry["sort_key"],
                }
            if self.enabled and persist_state:
                state = {
                    "schema_version": 1,
                    "lineages": dict(sorted(state_lineages.items())),
                }
                atomic_write_text(self.state_path, json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        return {
            "schema_version": 1,
            "enabled": self.enabled,
            "entries": entries,
            "summary": {**summary, "total": len(entries)},
        }

    def current_payload_fingerprint(self, key: str) -> str | None:
        with self._global_lock:
            entry = self._entries.get(key)
            return None if entry is None else entry.get("payload_fingerprint")
