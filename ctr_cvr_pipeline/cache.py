from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .canonical import fingerprint
from .io_utils import atomic_write_json


CACHE_SCHEMA = "ctr-cvr-content-cache-v1"


@dataclass
class CacheTask:
    stage: str
    key: str
    samples: tuple[str, ...]
    features: tuple[str, ...]
    compute: Callable[[], Any]
    upstream_key: str | None = None
    extra_refs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheResult:
    payloads: list[Any]
    ledger: list[dict[str, Any]]
    reused_count: int
    recomputed_count: int


class ArtifactCache:
    """Persistent content-addressed cache with deterministic, ordered ledgering."""

    def __init__(self, root: str | Path | None, *, collect_ledger: bool = True):
        self.root = Path(root) if root is not None else None
        self.collect_ledger = collect_ledger

    def _path(self, stage: str, key: str) -> Path:
        digest = key.split(":", 1)[-1]
        assert self.root is not None
        return self.root / stage / digest[:2] / f"{digest}.json"

    def _load(self, stage: str, key: str) -> Any | None:
        if self.root is None:
            return None
        path = self._path(stage, key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        if envelope.get("schema") != CACHE_SCHEMA or envelope.get("stage") != stage:
            return None
        if envelope.get("key") != key:
            return None
        payload = envelope.get("payload")
        if envelope.get("payload_fingerprint") != fingerprint(payload):
            return None
        return payload

    def _store(self, task: CacheTask, payload: Any) -> None:
        if self.root is None:
            return
        envelope = {
            "schema": CACHE_SCHEMA,
            "stage": task.stage,
            "key": task.key,
            "upstream_key": task.upstream_key,
            "samples": list(task.samples),
            "features": list(task.features),
            "extra_refs": task.extra_refs,
            "payload_fingerprint": fingerprint(payload),
            "payload": payload,
        }
        atomic_write_json(self._path(task.stage, task.key), envelope, pretty=False)

    def get_many(self, tasks: list[CacheTask], workers: int = 1) -> CacheResult:
        payloads: list[Any] = [None] * len(tasks)
        statuses: list[str] = ["COMPUTED"] * len(tasks)
        misses: list[int] = []
        for index, task in enumerate(tasks):
            payload = self._load(task.stage, task.key)
            if payload is not None:
                payloads[index] = payload
                statuses[index] = "REUSED"
            else:
                misses.append(index)

        if misses:
            if workers > 1 and len(misses) > 1:
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ctr-cvr-cache") as pool:
                    computed = list(pool.map(lambda idx: tasks[idx].compute(), misses))
            else:
                computed = [tasks[idx].compute() for idx in misses]
            for idx, payload in zip(misses, computed):
                payloads[idx] = payload
                self._store(tasks[idx], payload)

        ledger: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            if not self.collect_ledger:
                continue
            entry: dict[str, Any] = {
                "order": len(ledger),
                "stage": task.stage,
                "key": task.key,
                "status": statuses[index],
                "samples": list(task.samples),
                "features": list(task.features),
            }
            if task.upstream_key is not None:
                entry["upstream_key"] = task.upstream_key
            if task.extra_refs:
                entry["extra_refs"] = task.extra_refs
            ledger.append(entry)

        reused = sum(1 for status in statuses if status == "REUSED")
        return CacheResult(
            payloads=payloads,
            ledger=ledger,
            reused_count=reused,
            recomputed_count=len(tasks) - reused,
        )

    def get_one(self, task: CacheTask, workers: int = 1) -> tuple[Any, dict[str, Any], str]:
        result = self.get_many([task], workers=workers)
        status = result.ledger[0]["status"] if result.ledger else "COMPUTED"
        return result.payloads[0], result.ledger[0] if result.ledger else {}, status
