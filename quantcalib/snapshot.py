"""Crash-safe generation snapshots with an atomic CURRENT pointer."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from .utils import read_json, tree_digest, write_json


class StateStore:
    def __init__(self, root: Path, *, full_rebuild: bool = False) -> None:
        self.root = Path(root)
        self.generations = self.root / "generations"
        self.pointer_path = self.root / "CURRENT"
        self.root.mkdir(parents=True, exist_ok=True)
        self.generations.mkdir(parents=True, exist_ok=True)
        self.previous_generation = self._read_pointer()
        self.previous_manifest = self._read_manifest(self.previous_generation)
        self.transaction = self.root / f".txn-{uuid.uuid4().hex}"
        if not full_rebuild and self.previous_generation is not None:
            shutil.copytree(self.previous_generation, self.transaction)
        else:
            self.transaction.mkdir(parents=True, exist_ok=True)
        self._snapshot_counter = 0
        self.committed_snapshots: list[dict[str, Any]] = []

    def _read_pointer(self) -> Path | None:
        if not self.pointer_path.exists():
            return None
        try:
            data = json.loads(self.pointer_path.read_text(encoding="utf-8"))
            name = data.get("generation")
            if not isinstance(name, str):
                return None
            candidate = self.generations / name
            return candidate if candidate.is_dir() else None
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _read_manifest(generation: Path | None) -> dict[str, Any] | None:
        if generation is None:
            return None
        try:
            manifest = read_json(generation / "manifest.json")
            return manifest if isinstance(manifest, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def artifact_path(
        self,
        stage: str,
        layer: str | None = None,
        group_index: int | None = None,
    ) -> Path:
        base = self.transaction / "stages" / stage
        if layer is not None:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", layer)[:64]
            suffix = hashlib.sha256(layer.encode("utf-8")).hexdigest()[:12]
            base = base / f"{safe}-{suffix}"
        if group_index is not None:
            base = base / f"{group_index:08d}.json"
        else:
            base = base / "artifact.json"
        return base

    def read_artifact(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            value = read_json(path)
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def write_artifact(self, path: Path, value: dict[str, Any]) -> None:
        write_json(path, value)

    def stage_digest(self, stage: str) -> str:
        return tree_digest(self.transaction / "stages" / stage)

    def commit(self, manifest: dict[str, Any], snapshot_label: str) -> dict[str, Any]:
        self._snapshot_counter += 1
        snapshot_manifest = dict(manifest)
        snapshot_manifest["snapshot"] = {
            "sequence": self._snapshot_counter,
            "label": snapshot_label,
        }
        snapshot_manifest["stage_artifact_versions"] = {
            stage: self.stage_digest(stage)
            for stage in (
                "loading",
                "calibration",
                "quantization",
                "error_analysis",
                "distillation",
                "receipt_ledger",
            )
        }
        temporary = self.root / f".snap-{uuid.uuid4().hex}"
        shutil.copytree(self.transaction, temporary)
        write_json(temporary / "manifest.json", snapshot_manifest)
        generation_name = self._next_generation_name()
        destination = self.generations / generation_name
        temporary.replace(destination)
        pointer_tmp = self.root / f".CURRENT-{uuid.uuid4().hex}.tmp"
        pointer_tmp.write_text(
            json.dumps({"generation": generation_name}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(pointer_tmp, self.pointer_path)
        record = {
            "sequence": self._snapshot_counter,
            "label": snapshot_label,
            "generation": generation_name,
        }
        self.committed_snapshots.append(record)
        self._cleanup()
        return record

    def _next_generation_name(self) -> str:
        maximum = 0
        for item in self.generations.iterdir():
            if not item.is_dir():
                continue
            match = re.fullmatch(r"gen-(\d{8})", item.name)
            if match:
                maximum = max(maximum, int(match.group(1)))
        return f"gen-{maximum + 1:08d}"

    def _cleanup(self, keep: int = 8) -> None:
        current = self._read_pointer()
        generations = sorted(
            (item for item in self.generations.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        )
        removable = [item for item in generations if item != current]
        while len(generations) > keep and removable:
            victim = removable.pop(0)
            shutil.rmtree(victim, ignore_errors=True)
            generations.remove(victim)

    def close(self) -> None:
        shutil.rmtree(self.transaction, ignore_errors=True)

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
