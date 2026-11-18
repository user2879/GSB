from __future__ import annotations

import json
import gzip
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .util import atomic_write_json, fingerprint


class SnapshotError(RuntimeError):
    pass


class SnapshotManager:
    """Publishes immutable snapshot versions through an atomic CURRENT pointer.

    A snapshot is first written and fsynced in a private temporary directory.
    The directory is then renamed into ``versions/`` and only after that does
    ``CURRENT`` atomically move to the new version. A killed writer can leave a
    temporary directory behind, but readers never observe a partial snapshot.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.current_version = 0
        current = self.load_current()
        if current is not None:
            self.current_version = int(current["manifest"]["snapshot_version"])

    def load_current(self) -> dict[str, Any] | None:
        pointer = self.root / "CURRENT"
        if not pointer.exists():
            return None
        try:
            version_text = pointer.read_text(encoding="utf-8").strip()
            version = int(version_text)
        except (OSError, ValueError) as exc:
            raise SnapshotError(f"invalid CURRENT pointer: {exc}") from exc
        directory = self.versions_dir / f"v{version:08d}"
        manifest_path = directory / "manifest.json"
        state_path = directory / "state.json.gz"
        plain_state_path = directory / "state.json"
        if not manifest_path.exists() or not (
            state_path.exists() or plain_state_path.exists()
        ):
            raise SnapshotError(f"snapshot v{version} is incomplete")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if state_path.exists():
                with gzip.open(state_path, "rt", encoding="utf-8") as handle:
                    state = json.load(handle)
            else:
                state = json.loads(plain_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError(f"cannot read snapshot v{version}: {exc}") from exc
        if int(manifest.get("snapshot_version", -1)) != version:
            raise SnapshotError("manifest version does not match CURRENT pointer")
        return {
            "version": version,
            "directory": str(directory),
            "manifest": manifest,
            "state": state,
        }

    def commit(
        self,
        *,
        state: dict[str, Any],
        phase: str,
        cursor: int,
        status: str,
        config_hash: str,
        vocab_fingerprint: str,
        summary_fingerprint: str,
        artifact_versions: dict[str, int],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version = self.current_version + 1
        token = uuid.uuid4().hex
        temp_dir = Path(
            tempfile.mkdtemp(prefix=f".tmp-{version:08d}-{token}-", dir=self.root)
        )
        try:
            state_path = temp_dir / "state.json.gz"
            _write_json_gz_sync(state_path, state)
            artifact_paths = {
                stage: temp_dir / f"{stage}.json" for stage in artifact_versions
            }
            for stage, path in artifact_paths.items():
                _write_json_sync(
                    path,
                    {
                        "stage": stage,
                        "artifact_version": artifact_versions[stage],
                        "content_fingerprint": fingerprint(
                            {
                                "stage": stage,
                                "artifact_version": artifact_versions[stage],
                                "summary_fingerprint": summary_fingerprint,
                            }
                        ),
                    },
                )
            manifest = {
                "schema_version": 1,
                "snapshot_version": version,
                "config_hash": config_hash,
                "vocab_fingerprint": vocab_fingerprint,
                "cursor": cursor,
                "phase": phase,
                "status": status,
                "summary_fingerprint": summary_fingerprint,
                "artifact_versions": dict(sorted(artifact_versions.items())),
                "state_fingerprint": fingerprint(state),
                **(extra or {}),
            }
            _write_json_sync(temp_dir / "manifest.json", manifest)
            final_dir = self.versions_dir / f"v{version:08d}"
            if final_dir.exists():
                final_dir = self.versions_dir / f"v{version:08d}-{token[:8]}"
            os.replace(temp_dir, final_dir)
            _fsync_directory(self.versions_dir)
            self._publish_pointer(version)
            self.current_version = version
            return manifest
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _publish_pointer(self, version: int) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".CURRENT.", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{version}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.root / "CURRENT")
            _fsync_directory(self.root)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def write_export(self, destination: str | Path, value: Any) -> None:
        atomic_write_json(destination, value)


def _write_json_sync(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_gz_sync(path: Path, value: Any) -> None:
    with open(path, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            compressed.write(payload)
        raw.flush()
        os.fsync(raw.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
