"""Versioned, atomically published snapshots and live training checkpoints."""

from __future__ import annotations

import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cache import ArtifactCache
from .errors import E_SNAPSHOT_STALE, fail
from .util import atomic_write_json, atomic_write_text, content_hash, copy_tree_link_or_copy, read_json


MANIFEST_SCHEMA = "tabular-lab-snapshot/1"
LIVE_SCHEMA = "tabular-lab-live-checkpoint/1"
CURRENT_POINTER = "CURRENT"


@dataclass
class SnapshotHandle:
    path: Path
    manifest: dict[str, Any]


class SnapshotStore:
    def __init__(self, state_dir: str | Path, *, keep_generations: int = 3):
        self.state_dir = Path(state_dir)
        self.snapshots_dir = self.state_dir / "snapshots"
        self.pointer_path = self.state_dir / CURRENT_POINTER
        self.live_dir = self.state_dir / "live"
        self.keep_generations = max(1, int(keep_generations))
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)

    def load_current(self) -> SnapshotHandle | None:
        if not self.pointer_path.exists():
            return None
        try:
            generation_name = self.pointer_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            fail(E_SNAPSHOT_STALE, f"cannot read snapshot pointer: {exc}")
        if not generation_name:
            fail(E_SNAPSHOT_STALE, "snapshot pointer is empty")
        generation_path = self.snapshots_dir / generation_name
        manifest_path = generation_path / "manifest.json"
        if not generation_path.is_dir() or not manifest_path.is_file():
            fail(E_SNAPSHOT_STALE, "snapshot pointer references a missing generation", generation=generation_name)
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            fail(E_SNAPSHOT_STALE, f"snapshot manifest is unreadable: {exc}", generation=generation_name)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
            fail(E_SNAPSHOT_STALE, "snapshot manifest has an unsupported schema", generation=generation_name)
        return SnapshotHandle(generation_path, manifest)

    def check_resume(
        self,
        *,
        config_hash: str,
        data_fingerprint: str | None,
        strict: bool,
        expected_version: int | None,
    ) -> SnapshotHandle | None:
        current = self.load_current()
        if expected_version is not None and expected_version > 0:
            if current is None or int(current.manifest.get("global_version", -1)) != int(expected_version):
                fail(
                    E_SNAPSHOT_STALE,
                    "snapshot global version does not match the expected version",
                    expected_version=expected_version,
                    actual_version=None if current is None else current.manifest.get("global_version"),
                )
        if current is None or not strict:
            return current
        mismatches: dict[str, Any] = {}
        if current.manifest.get("config_hash") != config_hash:
            mismatches["config_hash"] = {
                "manifest": current.manifest.get("config_hash"),
                "current": config_hash,
            }
        if data_fingerprint is not None and current.manifest.get("data_fingerprint") != data_fingerprint:
            mismatches["data_fingerprint"] = {
                "manifest": current.manifest.get("data_fingerprint"),
                "current": data_fingerprint,
            }
        if mismatches:
            fail(E_SNAPSHOT_STALE, "snapshot is incompatible with the requested resume", mismatches=mismatches)
        return current

    def begin(self, *, previous: SnapshotHandle | None, rebuild: bool = False) -> "SnapshotSession":
        self._remove_abandoned_staging()
        previous_version = int(previous.manifest.get("global_version", 0)) if previous else 0
        generation = previous_version + 1
        staging = self.snapshots_dir / f"g{generation:08d}-{uuid.uuid4().hex[:8]}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        if previous is not None and not rebuild:
            # Hard-link immutable cache files and copy any small changed files.
            for child in previous.path.iterdir():
                destination = staging / child.name
                if child.is_dir():
                    copy_tree_link_or_copy(child, destination)
                else:
                    try:
                        import os

                        os.link(child, destination)
                    except OSError:
                        shutil.copy2(child, destination)
        (staging / "stages").mkdir(parents=True, exist_ok=True)
        (staging / "cache").mkdir(parents=True, exist_ok=True)
        (staging / "models").mkdir(parents=True, exist_ok=True)
        (staging / "exports").mkdir(parents=True, exist_ok=True)
        (staging / "sidecars").mkdir(parents=True, exist_ok=True)
        return SnapshotSession(
            store=self,
            staging=staging,
            generation=generation,
            previous=previous,
        )

    def _remove_abandoned_staging(self) -> None:
        for candidate in sorted(self.snapshots_dir.glob("*.staging")):
            if candidate.is_dir():
                try:
                    shutil.rmtree(candidate)
                except OSError:
                    pass

    def _prune_generations(self, keep_generation: str) -> None:
        current = self.load_current()
        keep_name = current.path.name if current else keep_generation
        generations = sorted(
            [path for path in self.snapshots_dir.iterdir() if path.is_dir() and not path.name.endswith(".staging")],
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        kept = 0
        for path in generations:
            if path.name == keep_name:
                kept = max(kept, 1)
                continue
            if kept < self.keep_generations:
                kept += 1
                continue
            try:
                shutil.rmtree(path)
            except OSError:
                pass


class SnapshotSession:
    def __init__(self, *, store: SnapshotStore, staging: Path, generation: int, previous: SnapshotHandle | None):
        self.store = store
        self.staging = staging
        self.generation = generation
        self.previous = previous
        self.cache = ArtifactCache(staging / "cache")
        self._committed = False
        self.stage_versions: dict[str, str] = {}

    def write_stage(self, name: str, version: str, payload: dict[str, Any]) -> None:
        if name not in {"load", "standardize", "train", "convert"}:
            raise ValueError(f"unknown snapshot stage: {name}")
        stage_dir = self.staging / "stages" / name
        stage_dir.mkdir(parents=True, exist_ok=True)
        body = {
            "stage": name,
            "version": version,
            "global_version": self.generation,
            "payload": payload,
        }
        atomic_write_json(stage_dir / "manifest.json", body)
        self.stage_versions[name] = version

    def write_json(self, relative_path: str, payload: Any) -> Path:
        destination = self.staging / relative_path
        atomic_write_json(destination, payload)
        return destination

    def write_text(self, relative_path: str, text: str) -> Path:
        destination = self.staging / relative_path
        atomic_write_text(destination, text)
        return destination

    def commit(
        self,
        *,
        config_hash: str,
        data_fingerprint: str,
        exchange_version: str,
        completed_models: list[str],
        failed_models: list[str],
        pending_models: list[str],
        consumed_epochs: int,
        model_refs: dict[str, Any],
        extra: dict[str, Any] | None = None,
        final_status: str = "complete",
    ) -> SnapshotHandle:
        if self._committed:
            raise RuntimeError("snapshot session already committed")
        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA,
            "global_version": self.generation,
            "final_status": final_status,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "exchange_version": exchange_version,
            "completed_models": list(completed_models),
            "failed_models": list(failed_models),
            "pending_models": list(pending_models),
            "consumed_epochs": int(consumed_epochs),
            "stage_versions": dict(sorted(self.stage_versions.items())),
            "model_refs": model_refs,
        }
        if extra:
            manifest["extra"] = extra
        atomic_write_json(self.staging / "manifest.json", manifest)
        final_name = self.staging.name[: -len(".staging")] if self.staging.name.endswith(".staging") else self.staging.name
        final_path = self.staging.with_name(final_name)
        import os

        if final_path.exists():
            shutil.rmtree(final_path)
        os.replace(self.staging, final_path)
        self.staging = final_path
        atomic_write_text(self.store.pointer_path, final_name + "\n")
        self._committed = True
        self.store._prune_generations(final_name)
        return SnapshotHandle(self.staging, manifest)

    def abort(self) -> None:
        if not self._committed and self.staging.exists():
            shutil.rmtree(self.staging, ignore_errors=True)


class LiveCheckpointStore:
    """Atomic crash-recovery checkpoints independent of a published generation."""

    def __init__(self, state_dir: str | Path):
        self.root = Path(state_dir) / "live"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.ledger_path = self.root / "ledger.json"

    def _path(self, train_key: str) -> Path:
        return self.root / f"{train_key}.json"

    def save(
        self,
        *,
        train_key: str,
        model_name: str,
        config_hash: str,
        data_fingerprint: str,
        state: dict[str, Any],
    ) -> None:
        payload = {
            "schema_version": LIVE_SCHEMA,
            "train_key": train_key,
            "model": model_name,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
            "state": state,
            "state_fingerprint": content_hash("live", state),
        }
        with self._lock:
            atomic_write_json(self._path(train_key), payload)

    def load(
        self,
        *,
        train_key: str,
        config_hash: str,
        data_fingerprint: str,
        strict_context: bool = False,
    ) -> dict[str, Any] | None:
        path = self._path(train_key)
        if not path.exists():
            return None
        try:
            payload = read_json(path)
        except Exception:
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != LIVE_SCHEMA:
            return None
        if payload.get("train_key") != train_key:
            return None
        state = payload.get("state")
        if not isinstance(state, dict):
            return None
        if payload.get("state_fingerprint") != content_hash("live", state):
            return None
        if strict_context and (
            payload.get("config_hash") != config_hash or payload.get("data_fingerprint") != data_fingerprint
        ):
            return None
        return state

    def remove(self, train_key: str) -> None:
        with self._lock:
            try:
                self._path(train_key).unlink()
            except FileNotFoundError:
                pass

    def update_ledger(
        self,
        *,
        train_key: str,
        model_name: str,
        consumed_epochs: int,
        config_hash: str,
        data_fingerprint: str,
    ) -> None:
        with self._lock:
            ledger: dict[str, Any] = {"schema_version": "budget-ledger/1", "models": {}}
            if self.ledger_path.exists():
                try:
                    loaded = read_json(self.ledger_path)
                    if isinstance(loaded, dict) and isinstance(loaded.get("models"), dict):
                        ledger = loaded
                except Exception:
                    pass
            ledger["config_hash"] = config_hash
            ledger["data_fingerprint"] = data_fingerprint
            ledger["models"][train_key] = {
                "model": model_name,
                "train_key": train_key,
                "consumed_epochs": int(consumed_epochs),
            }
            atomic_write_json(self.ledger_path, ledger)

    def ledger_summary(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            return {"models": {}, "total_recorded_epochs": 0}
        try:
            payload = read_json(self.ledger_path)
        except Exception:
            return {"models": {}, "total_recorded_epochs": 0}
        models = payload.get("models", {}) if isinstance(payload, dict) else {}
        if not isinstance(models, dict):
            models = {}
        total = sum(int(item.get("consumed_epochs", 0)) for item in models.values() if isinstance(item, dict))
        return {
            "models": {key: models[key] for key in sorted(models)},
            "total_recorded_epochs": total,
        }
