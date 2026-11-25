from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class JsonlLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            seq = 1
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as handle:
                    seq = sum(1 for _ in handle) + 1
            entry = {"ledger_id": f"L{seq:06d}", "seq": seq, "kind": kind, **payload}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return entry

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


class SidecarStore:
    """Persistent bypass metadata indexed by deterministic sidecar/business ids."""

    def __init__(self, path: str | Path, ledger: JsonlLedger | None = None):
        self.path = Path(path)
        self.ledger = ledger
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._items = data.get("items", {})

    def put(self, sidecar_id: str, business_key: str, item: dict[str, Any]) -> dict[str, Any]:
        record = {"sidecar_id": sidecar_id, "business_key": business_key, **item}
        with self._lock:
            prior = self._items.get(sidecar_id)
            versions = prior.get("versions", []) if prior else []
            versions.append(record)
            self._items[sidecar_id] = {"latest": record, "versions": versions}
            atomic_write_json(self.path, {"items": self._items})
        if self.ledger:
            self.ledger.append(
                "sidecar_put",
                {"sidecar_id": sidecar_id, "business_key": business_key, "fields": sorted(item.keys())},
            )
        return record

    def get(self, sidecar_id: str) -> dict[str, Any] | None:
        item = self._items.get(sidecar_id)
        return item["latest"] if item else None

    def find_latest_for_business_key(self, business_key: str) -> dict[str, Any] | None:
        matches = [
            item["latest"]
            for item in self._items.values()
            if item["latest"].get("business_key") == business_key
        ]
        return matches[-1] if matches else None

    def all_items(self) -> dict[str, dict[str, Any]]:
        return self._items
