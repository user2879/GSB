"""Versioned snapshot store with atomic temp-file replacement.

Layout::

    <root>/
      manifest.json
      stages/
        retrieval/<cache-key>.json
        extraction/<cache-key>.json
        citation/<cache-key>.json

Every write goes to ``<target>.tmp`` followed by ``os.replace`` so a
partially written file is never observable. Stage products are content
addressed by cache key; the manifest tracks global versioning and refuses
to resume when the configuration hash / rule fingerprint no longer matches.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_MISMATCH

MANIFEST_NAME = "manifest.json"
STAGE_NAMES = ("retrieval", "extraction", "citation")


class ConfigMismatchError(Exception):
    def __init__(self, message: str, detail: dict[str, Any]):
        super().__init__(message)
        self.code = E_CONFIG_MISMATCH
        self.detail = detail


class SnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.stage_dirs = {
            name: self.root / "stages" / name for name in STAGE_NAMES
        }
        self.manifest_path = self.root / MANIFEST_NAME

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.stage_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def write_stage(
        self, stage: str, cache_key: str, payload: dict[str, Any]
    ) -> Path:
        if stage not in STAGE_NAMES:
            raise KeyError(f"unknown stage: {stage}")
        path = self.stage_dirs[stage] / f"{cache_key}.json"
        self._atomic_write(path, payload)
        return path

    def read_stage(self, stage: str, cache_key: str) -> dict[str, Any] | None:
        path = self.stage_dirs[stage] / f"{cache_key}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def stage_versions(self) -> dict[str, list[str]]:
        versions: dict[str, list[str]] = {}
        for stage, directory in self.stage_dirs.items():
            if not directory.exists():
                versions[stage] = []
                continue
            versions[stage] = sorted(
                path.stem for path in directory.glob("*.json")
            )
        return versions

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self._atomic_write(self.manifest_path, manifest)

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def check_resume(
        self,
        config_hash: str,
        rules_fingerprint: str,
    ) -> dict[str, Any] | None:
        """Return the prior manifest when resuming is safe.

        Raises :class:`ConfigMismatchError` (E_CONFIG_MISMATCH) when an
        existing manifest references different configuration / rules.
        """
        manifest = self.read_manifest()
        if manifest is None:
            return None
        if (
            manifest.get("config_hash") != config_hash
            or manifest.get("rules_fingerprint") != rules_fingerprint
        ):
            raise ConfigMismatchError(
                "snapshot manifest does not match current config/rules",
                {
                    "stored_config_hash": manifest.get("config_hash"),
                    "current_config_hash": config_hash,
                    "stored_rules_fingerprint": manifest.get(
                        "rules_fingerprint"
                    ),
                    "current_rules_fingerprint": rules_fingerprint,
                    "stored_global_version": manifest.get("global_version"),
                },
            )
        return manifest

    def reset(self) -> None:
        if self.root.exists():
            for path in sorted(self.root.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        self.ensure_dirs()

