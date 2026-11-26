"""Dependency-aware in-memory cache used by the materializer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from .canonical import sha256_value


T = TypeVar("T")


@dataclass(frozen=True)
class CacheEntry:
    stage: str
    key: str
    dependencies: tuple[str, ...]
    block_refs: tuple[str, ...]
    window_refs: tuple[str, ...]
    cluster_refs: tuple[str, ...]
    config_tags: tuple[str, ...]
    payload_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "key": self.key,
            "dependencies": list(self.dependencies),
            "block_refs": list(self.block_refs),
            "window_refs": list(self.window_refs),
            "cluster_refs": list(self.cluster_refs),
            "config_tags": list(self.config_tags),
            "payload_hash": self.payload_hash,
        }


@dataclass
class CacheRegistry:
    entries: dict[str, CacheEntry] = field(default_factory=dict)
    payloads: dict[str, Any] = field(default_factory=dict)
    last_reuse: list[str] = field(default_factory=list)

    def clear(self) -> None:
        self.entries.clear()
        self.payloads.clear()
        self.last_reuse.clear()

    def has(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str) -> Any:
        try:
            return self.payloads[key]
        except KeyError as exc:
            raise KeyError(f"cache payload missing for {key}") from exc

    def put(
        self,
        *,
        stage: str,
        key: str,
        payload: Any,
        dependencies: Iterable[str] = (),
        block_refs: Iterable[str] = (),
        window_refs: Iterable[str] = (),
        cluster_refs: Iterable[str] = (),
        config_tags: Iterable[str] = (),
    ) -> CacheEntry:
        entry = CacheEntry(
            stage=stage,
            key=key,
            dependencies=tuple(sorted(set(dependencies))),
            block_refs=tuple(sorted(set(block_refs))),
            window_refs=tuple(sorted(set(window_refs))),
            cluster_refs=tuple(sorted(set(cluster_refs))),
            config_tags=tuple(sorted(set(config_tags))),
            payload_hash=sha256_value(payload),
        )
        self.entries[key] = entry
        self.payloads[key] = payload
        return entry

    def get_or_compute(
        self,
        *,
        stage: str,
        key: str,
        factory: Callable[[], T],
        dependencies: Iterable[str] = (),
        block_refs: Iterable[str] = (),
        window_refs: Iterable[str] = (),
        cluster_refs: Iterable[str] = (),
        config_tags: Iterable[str] = (),
    ) -> tuple[T, bool]:
        if key in self.entries:
            self.last_reuse.append(key)
            return self.get(key), True
        payload = factory()
        self.put(
            stage=stage,
            key=key,
            payload=payload,
            dependencies=dependencies,
            block_refs=block_refs,
            window_refs=window_refs,
            cluster_refs=cluster_refs,
            config_tags=config_tags,
        )
        return payload, False

    def invalidate_blocks(self, block_ids: Iterable[str]) -> list[str]:
        ids = set(block_ids)
        if not ids:
            return []
        invalidated = {
            key
            for key, entry in self.entries.items()
            if ids.intersection(entry.block_refs)
        }
        removed = self._remove_with_dependents(invalidated)
        return sorted(removed)

    def invalidate_config_tags(self, tags: Iterable[str]) -> list[str]:
        selected = set(tags)
        if not selected:
            return []
        invalidated = {
            key
            for key, entry in self.entries.items()
            if selected.intersection(entry.config_tags)
        }
        removed = self._remove_with_dependents(invalidated)
        return sorted(removed)

    def invalidate_keys(self, keys: Iterable[str]) -> list[str]:
        selected = set(keys)
        removed = self._remove_with_dependents(selected)
        return sorted(removed)

    def _remove_with_dependents(self, keys: set[str]) -> set[str]:
        removed = set(keys)
        changed = True
        while changed:
            changed = False
            for key, entry in self.entries.items():
                if key in removed:
                    continue
                if removed.intersection(entry.dependencies):
                    removed.add(key)
                    changed = True
        for key in removed:
            self.entries.pop(key, None)
            self.payloads.pop(key, None)
        return removed

    def current_manifest(self) -> list[dict[str, Any]]:
        return [self.entries[key].to_dict() for key in sorted(self.entries)]

    def dependencies_manifest(self) -> dict[str, list[str]]:
        return {key: list(entry.dependencies) for key, entry in sorted(self.entries.items())}

    def prune_to(self, roots: Iterable[str]) -> None:
        """Remove entries not reachable from the supplied final-stage keys."""

        keep: set[str] = set()
        stack = list(roots)
        while stack:
            key = stack.pop()
            if key in keep or key not in self.entries:
                continue
            keep.add(key)
            stack.extend(self.entries[key].dependencies)
        self.invalidate_keys(set(self.entries) - keep)
