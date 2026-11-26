from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from . import SNAPSHOT_FORMAT_VERSION
from .canonical import canonical_json, content_digest
from .ledger import Ledger
from .spec import Spec


class SnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def build_snapshot(self, spec: Spec, ledger: Ledger) -> dict[str, Any]:
        view = ledger.export_view()
        return {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "ledger_format_version": 1,
            "spec_digest": spec.digest,
            "topo_order": list(spec.topo_order),
            "view": view,
            "content_digest": content_digest(view),
        }

    def write_atomic(self, snapshot: dict[str, Any]) -> Path:
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        return self.path

    def sync(self, spec: Spec, ledger: Ledger) -> dict[str, Any]:
        snapshot = self.build_snapshot(spec, ledger)
        self.write_atomic(snapshot)
        return snapshot

    def open_or_replace(
        self, spec: Spec, ledger: Ledger
    ) -> tuple[dict[str, Any], bool, str | None]:
        reason: str | None = None
        existing: dict[str, Any] | None = None
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                reason = "unreadable_snapshot"
        if reason is None and existing is not None:
            if existing.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
                reason = "format_version_mismatch"
            elif existing.get("spec_digest") != spec.digest:
                reason = "spec_digest_mismatch"
            elif existing.get("content_digest") != content_digest(existing.get("view")):
                reason = "content_digest_mismatch"
        if reason is not None or existing is None:
            snapshot = self.sync(spec, ledger)
            return snapshot, True, reason or "missing_snapshot"
        return existing, False, None


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
