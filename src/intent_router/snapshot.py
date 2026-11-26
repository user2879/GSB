"""Atomic staged snapshots with version/config-hash gated commit."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fingerprints import digest
from .ledger import Ledger

SNAPSHOT_FILE = "snapshot.json"
STAGE_DIRNAME = ".stage"


def config_hash(config_dict: dict[str, Any]) -> str:
    return digest({"kind": "config", "config": config_dict})


@dataclass
class SnapshotBundle:
    version: int
    config_hash: str
    gear_key: str
    head_revision: str
    products: dict[str, Any]
    pending: list[dict[str, Any]]
    consumed: dict[str, Any]
    ledger: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config_hash": self.config_hash,
            "gear_key": self.gear_key,
            "head_revision": self.head_revision,
            "products": self.products,
            "pending": self.pending,
            "consumed": self.consumed,
            "ledger": self.ledger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SnapshotBundle":
        return cls(
            version=int(data["version"]),
            config_hash=str(data["config_hash"]),
            gear_key=str(data["gear_key"]),
            head_revision=str(data["head_revision"]),
            products=dict(data.get("products", {})),
            pending=list(data.get("pending", [])),
            consumed=dict(data.get("consumed", {})),
            ledger=dict(data.get("ledger", {})),
        )


class SnapshotStore:
    """Writes go to a temp staging dir then atomically replace the snapshot."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / SNAPSHOT_FILE
        self.stage_dir = self.root / STAGE_DIRNAME

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> SnapshotBundle | None:
        if not self.exists():
            return None
        with open(self.path, "r", encoding="utf-8") as handle:
            return SnapshotBundle.from_dict(json.load(handle))

    def load_ledger(self) -> Ledger:
        bundle = self.load()
        if bundle is None or not bundle.ledger:
            return Ledger()
        return Ledger.from_snapshot(bundle.ledger)

    def commit(self, bundle: SnapshotBundle, expected_version: int | None = None) -> SnapshotBundle:
        """Stage then atomically replace; gate on the global version number."""
        current = self.load()
        if expected_version is not None and current is not None:
            if current.version != expected_version:
                raise RuntimeError(
                    f"version conflict: expected {expected_version}, "
                    f"found {current.version}"
                )
        payload = bundle.to_dict()
        payload["snapshot_fingerprint"] = digest(
            {k: v for k, v in payload.items() if k != "snapshot_fingerprint"}
        )
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix="snap-", suffix=".json", dir=str(self.stage_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return bundle
