"""Whole-snapshot replacement, manifest validation and stale-snapshot detection."""

from __future__ import annotations

import copy
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .codes import E_SNAPSHOT_STALE
from .models import Failure
from .util import read_json, remove_tree, write_json_atomic, write_text_atomic

SNAPSHOT_SCHEMA_VERSION = 1
ARTIFACT_NAMES = ("validation", "samples", "minhash", "dedup", "result")


class SnapshotStaleError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.details = details or {}
        super().__init__(message)

    def failure(self) -> Failure:
        return Failure(
            source="",
            doc_id="",
            stage="snapshot",
            code=E_SNAPSHOT_STALE,
            message=str(self),
        )


@dataclass
class SnapshotBundle:
    root: Path
    version: int
    manifest: dict[str, Any]
    artifacts: dict[str, Any]

    def state(self) -> dict[str, Any]:
        state = {
            "global_version": self.version,
            "config_hashes": copy.deepcopy(self.manifest.get("config_hashes", {})),
            "corpus_digest": self.manifest.get("corpus_digest"),
            "stage_versions": copy.deepcopy(self.manifest.get("stage_versions", {})),
            "stage_statuses": copy.deepcopy(self.manifest.get("stage_statuses", {})),
        }
        for name in ARTIFACT_NAMES:
            state[name] = copy.deepcopy(self.artifacts.get(name))
        return state


class SnapshotStore:
    def __init__(self, output_root: str | Path, run_id: str):
        if not run_id or any(part in run_id for part in ("/", "\\", "..")):
            raise ValueError("run_id must be a non-empty path-safe string")
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.root = self.output_root / "snapshots" / run_id
        self.versions_root = self.root / "versions"
        self.temp_root = self.output_root / "tmp" / "snapshots" / run_id
        self.current_path = self.root / "CURRENT"
        self.versions_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_temps()

    def _cleanup_temps(self) -> None:
        for path in sorted(self.temp_root.glob("commit-*")):
            remove_tree(path)

    def exists(self) -> bool:
        return self.current_path.exists()

    def reset(self) -> None:
        remove_tree(self.root)
        self.versions_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self._cleanup_temps()

    def current_version(self) -> int | None:
        if not self.current_path.exists():
            return None
        try:
            text = self.current_path.read_text(encoding="utf-8").strip()
            return int(text.removeprefix("v"))
        except (OSError, ValueError):
            raise SnapshotStaleError("CURRENT pointer is unreadable")

    def load_current(self) -> SnapshotBundle:
        version = self.current_version()
        if version is None:
            raise SnapshotStaleError("no snapshot CURRENT pointer exists")
        version_dir = self.versions_root / f"v{version:06d}"
        manifest_path = version_dir / "manifest.json"
        if not version_dir.is_dir() or not manifest_path.exists():
            raise SnapshotStaleError(f"snapshot version v{version:06d} is missing")
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotStaleError(f"snapshot manifest is invalid: {exc}")
        if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotStaleError(
                f"snapshot schema version {manifest.get('schema_version')} is unsupported"
            )
        if manifest.get("run_id") != self.run_id:
            raise SnapshotStaleError("snapshot run_id does not match requested run")
        if int(manifest.get("global_version", -1)) != version:
            raise SnapshotStaleError("snapshot manifest version does not match CURRENT")

        artifacts: dict[str, Any] = {}
        hashes = manifest.get("artifact_hashes", {})
        for name in ARTIFACT_NAMES:
            path = version_dir / f"{name}.json"
            if not path.exists():
                artifacts[name] = None
                continue
            try:
                value = read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                raise SnapshotStaleError(f"snapshot artifact {name} is unreadable: {exc}")
            expected_hash = hashes.get(name)
            if expected_hash is not None:
                from .util import sha256_json

                if sha256_json(value) != expected_hash:
                    raise SnapshotStaleError(f"snapshot artifact {name} hash mismatch")
            artifacts[name] = value
        return SnapshotBundle(version_dir, version, manifest, artifacts)

    def verify_resume(
        self,
        bundle: SnapshotBundle,
        *,
        expected_version: int | None,
        balance_hash: str,
        dedup_hash: str,
        corpus_digest: str | None = None,
    ) -> None:
        reasons: list[str] = []
        manifest = bundle.manifest
        if expected_version is not None and int(expected_version) != bundle.version:
            reasons.append(
                f"expected snapshot version {expected_version}, found {bundle.version}"
            )
        hashes = manifest.get("config_hashes", {})
        if hashes.get("balance") != balance_hash:
            reasons.append("balance configuration hash changed; old sampled artifacts are stale")
        if hashes.get("dedup") != dedup_hash:
            reasons.append("dedup configuration hash changed; old signatures/groups are stale")
        if reasons:
            raise SnapshotStaleError("; ".join(reasons), {"reasons": reasons})

    def commit(
        self,
        state: dict[str, Any],
        *,
        stage_name: str,
        stage_status: str,
        config_hashes: dict[str, str],
        corpus_digest: str,
        stage_elapsed_ms: dict[str, float] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        current_version = self.current_version() or 0
        next_version = max(int(state.get("global_version", 0) or 0), current_version) + 1
        while (self.versions_root / f"v{next_version:06d}").exists():
            next_version += 1
        new_state = copy.deepcopy(state)
        new_state["global_version"] = next_version
        new_state["config_hashes"] = dict(config_hashes)
        new_state["corpus_digest"] = corpus_digest
        stage_versions = dict(new_state.get("stage_versions", {}))
        stage_statuses = dict(new_state.get("stage_statuses", {}))
        stage_versions[stage_name] = next_version
        stage_statuses[stage_name] = stage_status
        new_state["stage_versions"] = stage_versions
        new_state["stage_statuses"] = stage_statuses

        version_dir = self.versions_root / f"v{next_version:06d}"
        temp_dir = self.temp_root / f"commit-{next_version:06d}-{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)

        artifact_hashes: dict[str, str] = {}
        try:
            from .util import sha256_json

            for name in ARTIFACT_NAMES:
                value = new_state.get(name)
                if value is None:
                    continue
                write_json_atomic(temp_dir / f"{name}.json", value)
                artifact_hashes[name] = sha256_json(value)

            minhash = new_state.get("minhash") or {}
            samples_artifact = new_state.get("samples") or {}
            manifest = {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "run_id": self.run_id,
                "global_version": next_version,
                "config_hashes": dict(config_hashes),
                "corpus_digest": corpus_digest,
                "samples": samples_artifact.get("samples", []),
                "sample_count": len(samples_artifact.get("samples", [])),
                "consumed_points": int(minhash.get("consumed_points", 0) or 0),
                "pending_queue": minhash.get("pending", []),
                "cursor": minhash.get("cursor", {}),
                "stage_versions": stage_versions,
                "stage_statuses": stage_statuses,
                "stage_elapsed_ms": stage_elapsed_ms or {},
                "artifact_hashes": artifact_hashes,
            }
            write_json_atomic(temp_dir / "manifest.json", manifest)
            self._fsync_directory(temp_dir)
            os.replace(temp_dir, version_dir)
            write_text_atomic(self.current_path, f"v{next_version:06d}\n")
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        return next_version, new_state

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass
