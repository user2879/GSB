from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import CacheError
from .util import atomic_write_json, fingerprint, read_json


KIND_ORDER = {
    "raw_window_feature": 10,
    "standardization_stats": 20,
    "standardized_observation": 30,
    "cluster_model": 40,
    "density_model": 50,
    "snapshot": 60,
}


def content_cache_key(kind: str, payload: dict[str, Any]) -> str:
    return f"{kind}:{fingerprint(payload)}"


class PersistentCache:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / "cache.json"
        self.entries: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = read_json(self.path)
        except (OSError, ValueError) as exc:
            raise CacheError(f"cannot read cache: {exc}", path="state/cache.json") from exc
        if not isinstance(data, dict):
            raise CacheError("cache root must be a mapping", path="state/cache.json")
        entries = data.get("entries", {})
        if not isinstance(entries, dict):
            raise CacheError("cache entries must be a mapping", path="state/cache.json.entries")
        self.entries = {
            str(key): value
            for key, value in entries.items()
            if isinstance(value, dict)
        }

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": 1,
                "entries": {
                    key: self.entries[key] for key in sorted(self.entries)
                },
            },
        )

    def _record_event(
        self,
        *,
        kind: str,
        key: str,
        outcome: str,
        reason: str,
        dependencies: list[str],
        sort_key: tuple[int, str],
        window_index: int | None,
        path: str,
    ) -> None:
        self.events.append(
            {
                "kind": kind,
                "key": key,
                "outcome": outcome,
                "reason": reason,
                "dependencies": list(dependencies),
                "sort_key": [sort_key[0], sort_key[1]],
                "window_index": window_index,
                "path": path,
            }
        )

    def lookup(
        self,
        *,
        kind: str,
        key: str,
        dependencies: list[str],
        miss_reason: str,
        sort_key: tuple[int, str],
        window_index: int | None,
        path: str,
    ) -> Any | None:
        entry = self.entries.get(key)
        if entry is not None and entry.get("kind") == kind:
            self._record_event(
                kind=kind,
                key=key,
                outcome="reused",
                reason="cache_key_hit",
                dependencies=dependencies,
                sort_key=sort_key,
                window_index=window_index,
                path=path,
            )
            return entry.get("payload")
        self._record_event(
            kind=kind,
            key=key,
            outcome="recomputed",
            reason=miss_reason,
            dependencies=dependencies,
            sort_key=sort_key,
            window_index=window_index,
            path=path,
        )
        return None

    def put(
        self,
        *,
        kind: str,
        key: str,
        payload: Any,
        dependencies: list[str],
        lease_generation: int | None = None,
        window_index: int | None = None,
    ) -> None:
        self.entries[key] = {
            "kind": kind,
            "payload": payload,
            "dependencies": list(dependencies),
            "lease_generation": lease_generation,
            "window_index": window_index,
        }

    def sorted_events(self) -> list[dict[str, Any]]:
        return sorted(
            self.events,
            key=lambda event: (
                int(event["sort_key"][0]),
                KIND_ORDER.get(str(event["kind"]), 99),
                str(event["key"]),
                str(event["outcome"]),
            ),
        )

    def ledger_keys(self) -> list[str]:
        return [str(event["key"]) for event in self.sorted_events()]

    def digest(self) -> str:
        return fingerprint(
            {
                key: {
                    "kind": entry.get("kind"),
                    "payload": entry.get("payload"),
                }
                for key, entry in sorted(self.entries.items())
            }
        )

    def summary(self) -> dict[str, Any]:
        events = self.sorted_events()
        reused = sum(1 for event in events if event["outcome"] == "reused")
        recomputed = sum(1 for event in events if event["outcome"] == "recomputed")
        return {
            "path": str(self.path),
            "entry_count": len(self.entries),
            "reused_count": reused,
            "recomputed_count": recomputed,
            "events": events,
            "artifact_key_order": [event["key"] for event in events],
            "digest": self.digest(),
        }

