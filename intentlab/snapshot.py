"""Whole-generation snapshot staging, commit, and crash recovery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .fingerprints import digest


class SnapshotError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CurrentReference:
    generation: str
    state_token: str
    config_hash: str
    global_version: int
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "state_token": self.state_token,
            "config_hash": self.config_hash,
            "global_version": self.global_version,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class SnapshotHandle:
    generation: str
    path: Path
    manifest: Mapping[str, Any]
    reference: CurrentReference


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
    with temp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.generations = self.root / "generations"
        self.staging = self.root / "staging"
        self.current_path = self.root / "current.json"
        self.generations.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)

    def _read_reference(self) -> CurrentReference | None:
        if not self.current_path.exists():
            return None
        try:
            data = json.loads(self.current_path.read_text(encoding="utf-8"))
            return CurrentReference(
                generation=str(data["generation"]),
                state_token=str(data["state_token"]),
                config_hash=str(data["config_hash"]),
                global_version=int(data["global_version"]),
                manifest_sha256=str(data["manifest_sha256"]),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SnapshotError("E_SNAPSHOT_INVALID", f"invalid current pointer: {exc}") from exc

    def current_reference(self) -> CurrentReference | None:
        return self._read_reference()

    def load_current(self) -> SnapshotHandle | None:
        reference = self._read_reference()
        if reference is None:
            return None
        generation_path = self.generations / reference.generation
        manifest_path = generation_path / "manifest.json"
        if not manifest_path.exists():
            raise SnapshotError(
                "E_SNAPSHOT_INTEGRITY",
                f"committed generation manifest is missing: {manifest_path}",
            )
        manifest_hash = _sha256_file(manifest_path)
        if manifest_hash != reference.manifest_sha256:
            raise SnapshotError(
                "E_SNAPSHOT_INTEGRITY",
                "manifest checksum does not match current pointer",
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SnapshotError("E_SNAPSHOT_INTEGRITY", f"cannot read manifest: {exc}") from exc
        if not manifest.get("complete"):
            raise SnapshotError("E_SNAPSHOT_INTEGRITY", "current snapshot is not marked complete")
        if manifest.get("state_token") != reference.state_token:
            raise SnapshotError("E_SNAPSHOT_INTEGRITY", "state token mismatch for current snapshot")
        files = manifest.get("files", {})
        for relative, expected_hash in files.items():
            path = generation_path / relative
            if not path.exists() or _sha256_file(path) != expected_hash:
                raise SnapshotError(
                    "E_SNAPSHOT_INTEGRITY",
                    f"snapshot artifact failed integrity check: {relative}",
                )
        return SnapshotHandle(
            generation=reference.generation,
            path=generation_path,
            manifest=manifest,
            reference=reference,
        )

    def read_artifact(self, handle: SnapshotHandle, relative: str) -> Any:
        path = handle.path / relative
        return json.loads(path.read_text(encoding="utf-8"))

    def new_stager(
        self,
        *,
        state_token: str,
        config_hash: str,
        global_version: int,
        expected_reference: CurrentReference | None,
    ) -> "SnapshotStager":
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        generation = f"gen-{stamp}-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        return SnapshotStager(
            store=self,
            generation=generation,
            state_token=state_token,
            config_hash=config_hash,
            global_version=global_version,
            expected_reference=expected_reference,
        )

    def recovery_info(self, current: SnapshotHandle | None) -> dict[str, Any]:
        staging_entries: list[dict[str, Any]] = []
        if self.staging.exists():
            for path in sorted(self.staging.iterdir()):
                if not path.is_dir():
                    continue
                stages_path = path / "stages.json"
                stages: list[str] = []
                if stages_path.exists():
                    try:
                        stages = [str(item) for item in json.loads(stages_path.read_text(encoding="utf-8"))]
                    except (OSError, ValueError):
                        stages = ["<corrupt-stages-index>"]
                staging_entries.append(
                    {
                        "generation": path.name,
                        "completed_stages": stages,
                        "committed": False,
                    }
                )
        committed_generation = current.generation if current else None
        unreferenced_generations: list[dict[str, Any]] = []
        if self.generations.exists():
            for path in sorted(self.generations.iterdir()):
                if not path.is_dir() or path.name == committed_generation:
                    continue
                manifest_path = path / "manifest.json"
                complete = False
                stages: list[str] = []
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        complete = bool(manifest.get("complete"))
                        stages = [str(item) for item in manifest.get("stages", [])]
                    except (OSError, ValueError):
                        stages = ["<corrupt-manifest>"]
                unreferenced_generations.append(
                    {
                        "generation": path.name,
                        "complete_manifest": complete,
                        "completed_stages": stages,
                        "loadable": False,
                    }
                )
        return {
            "committed_generation": committed_generation,
            "resumed_from_committed_snapshot": current is not None,
            "ignored_staging": staging_entries,
            "unreferenced_generations": unreferenced_generations,
            "policy": "only the generation referenced by current.json is loadable",
        }


@dataclass
class SnapshotStager:
    store: SnapshotStore
    generation: str
    state_token: str
    config_hash: str
    global_version: int
    expected_reference: CurrentReference | None
    path: Path = field(init=False)
    files: dict[str, str] = field(default_factory=dict)
    stages: list[str] = field(default_factory=list)
    stage_metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = self.store.staging / self.generation
        self.path.mkdir(parents=True, exist_ok=False)
        _fsync_dir(self.path.parent)

    def write_json(self, relative: str, data: Any) -> None:
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"invalid snapshot path: {relative}")
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        with target.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.files[relative] = hashlib.sha256(payload).hexdigest()

    def write_text(self, relative: str, text: str) -> None:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = text.encode("utf-8")
        with target.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self.files[relative] = hashlib.sha256(payload).hexdigest()

    def finish_stage(self, stage: str, metrics: Mapping[str, Any] | None = None) -> None:
        self.stages.append(stage)
        if metrics is not None:
            self.stage_metrics[stage] = dict(metrics)
        _atomic_write_json(self.path / "stages.json", self.stages)
        _fsync_dir(self.path)
        self._maybe_crash(stage)

    @staticmethod
    def _maybe_crash(stage: str) -> None:
        target = os.environ.get("INTENTLAB_CRASH_AFTER_STAGE", "")
        if target != stage:
            return
        mode = os.environ.get("INTENTLAB_CRASH_MODE", "sigkill")
        if mode == "exit":
            os._exit(137)
        os.kill(os.getpid(), signal.SIGKILL)

    def commit(self) -> CurrentReference:
        current = self.store.current_reference()
        expected = self.expected_reference
        if expected is None:
            if current is not None:
                raise SnapshotError(
                    "E_SNAPSHOT_CONCURRENT_COMMIT",
                    "another run committed a generation while this run was staging",
                )
        else:
            if current is None or current.to_dict() != expected.to_dict():
                raise SnapshotError(
                    "E_SNAPSHOT_CONCURRENT_COMMIT",
                    "global version or config hash changed before snapshot commit",
                )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "generation": self.generation,
            "complete": True,
            "state_token": self.state_token,
            "config_hash": self.config_hash,
            "global_version": self.global_version,
            "files": dict(sorted(self.files.items())),
            "stages": list(self.stages),
            "stage_metrics": self.stage_metrics,
        }
        manifest_path = self.path / "manifest.json"
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        with manifest_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.path)
        manifest_hash = hashlib.sha256(payload).hexdigest()
        reference = CurrentReference(
            generation=self.generation,
            state_token=self.state_token,
            config_hash=self.config_hash,
            global_version=self.global_version,
            manifest_sha256=manifest_hash,
        )
        final_generation_path = self.store.generations / self.generation
        if final_generation_path.exists():
            shutil.rmtree(final_generation_path)
        os.replace(self.path, final_generation_path)
        _fsync_dir(self.store.generations)
        _atomic_write_json(self.store.current_path, reference.to_dict())
        return reference


def config_hash(config: Mapping[str, Any]) -> str:
    return digest(config)
