"""Read-only diagnostic service over an immutable snapshot."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import E_DEPENDENCY_STALE, fail
from .hashing import canonical_json_bytes, sha256_bytes


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


class DiagnosticSnapshot:
    def __init__(self, data: dict[str, Any], *, source_path: str, source_hash: str):
        self._data = _freeze(copy.deepcopy(data))
        self._json_bytes = canonical_json_bytes(data)
        self.source_path = source_path
        self.source_hash = source_hash
        if self._json_bytes != canonical_json_bytes(json.loads(self._json_bytes)):
            fail(E_DEPENDENCY_STALE, "diagnostic snapshot is not canonically stable")

    @property
    def revision(self) -> int:
        return int(self._data["revision"])

    @property
    def data(self) -> MappingProxyType:
        return self._data

    @property
    def json_bytes(self) -> bytes:
        return self._json_bytes

    def projection_png(self) -> bytes:
        encoded = self._data["projection"]["preview_png_base64"]
        return base64.b64decode(str(encoded), validate=True)


class DiagnosticService:
    """Exposes one frozen snapshot and deliberately has no recompute entry point."""

    def __init__(self, snapshot: DiagnosticSnapshot):
        self._snapshot = snapshot

    @classmethod
    def from_workspace(cls, workspace: Path | str) -> "DiagnosticService":
        from .store import WorkspaceStore

        store = WorkspaceStore(workspace)
        data, snapshot_hash = store.load_latest_snapshot()
        latest_name = (store.snapshot_dir / "LATEST").read_text(encoding="ascii").strip()
        path = store.snapshot_dir / latest_name
        if sha256_bytes(path.read_bytes()) != snapshot_hash:
            fail(E_DEPENDENCY_STALE, "snapshot changed while loading", path=str(path))
        return cls(DiagnosticSnapshot(data, source_path=str(path), source_hash=snapshot_hash))

    @property
    def snapshot(self) -> DiagnosticSnapshot:
        return self._snapshot

    def snapshot_json(self) -> bytes:
        return self._snapshot.json_bytes

    def projection_png(self) -> bytes:
        return self._snapshot.projection_png()
