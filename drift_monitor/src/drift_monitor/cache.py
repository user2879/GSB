"""Content-fingerprint derived cache keys and stage artifact storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STAGES = ("load", "slice", "metric", "decision", "report", "visual")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_default)


def _json_default(value):
    if isinstance(value, (tuple, set)):
        return sorted(value, key=lambda x: str(x))
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def fingerprint(payload: Any, prefix: str = "") -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}{digest[:16]}" if prefix else digest[:16]


def derived_key(upstream: str, params: Any, namespace: str) -> str:
    return fingerprint(
        {"upstream": upstream, "params": params, "namespace": namespace},
        prefix=f"{namespace}:",
    )


class StageCache:
    """File-backed per-stage artifact cache keyed by derived cache keys."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        for stage in STAGES:
            (self.root / stage).mkdir(exist_ok=True)

    def _path(self, stage: str, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / stage / f"{safe}.json"

    def get(self, stage: str, key: str) -> Any | None:
        path = self._path(stage, key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, stage: str, key: str, payload: Any) -> None:
        path = self._path(stage, key)
        path.write_text(canonical_json(payload), encoding="utf-8")

    def existing_keys(self, stage: str) -> set[str]:
        keys = set()
        for path in (self.root / stage).glob("*.json"):
            keys.add(path.stem)
        return keys
