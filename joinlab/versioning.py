from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import fail
from .logging_utils import atomic_write_text
from .params import AnalysisParams


class VersionStore:
    def __init__(self, path: str | Path | None, workspace_id: str) -> None:
        self.path = Path(path) if path else None
        self.workspace_id = workspace_id

    def _read(self) -> dict[str, Any]:
        if not self.path or not self.path.exists():
            return {"format": "joinlab-version-state-v1", "workspaces": {}}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            fail("E_SNAP", "参数版本状态文件无法读取", "version_state", reason=str(exc))
        if not isinstance(document, dict) or not isinstance(document.get("workspaces", {}), dict):
            fail("E_SNAP", "参数版本状态文件结构无效", "version_state")
        document.setdefault("workspaces", {})
        return document

    def _write(self, document: dict[str, Any]) -> None:
        if not self.path:
            return
        atomic_write_text(self.path, canonical_json(document) + "\n")

    def ensure(self, params: AnalysisParams) -> int:
        document = self._read()
        workspaces = document["workspaces"]
        entry = workspaces.get(self.workspace_id)
        params_hash = params.digest()
        if isinstance(entry, dict) and entry.get("params_digest") == params_hash:
            version = entry.get("version")
            if isinstance(version, int) and not isinstance(version, bool) and version > 0:
                return version
        version = 1
        if isinstance(entry, dict) and isinstance(entry.get("version"), int):
            version = max(version, int(entry["version"]) + 1)
        workspaces[self.workspace_id] = {
            "version": version,
            "params_digest": params_hash,
            "params": params.as_dict(),
        }
        self._write(document)
        return version

    def bump(self, params: AnalysisParams) -> int:
        document = self._read()
        workspaces = document["workspaces"]
        entry = workspaces.get(self.workspace_id)
        previous = entry.get("version", 0) if isinstance(entry, dict) else 0
        if not isinstance(previous, int) or isinstance(previous, bool):
            previous = 0
        version = previous + 1
        workspaces[self.workspace_id] = {
            "version": version,
            "params_digest": params.digest(),
            "params": params.as_dict(),
        }
        self._write(document)
        return version

