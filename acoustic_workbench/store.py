"""Content-addressed workspace persistence with startup hash verification."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_DEPENDENCY_STALE, E_TILE_HASH, fail
from .hashing import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    content_hash,
    read_json,
    sha256_bytes,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BlobRef:
    sha256: str
    suffix: str


class WorkspaceStore:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.scene_path = self.root / "scene.json"
        self.state_path = self.root / "state.json"
        self.blob_dir = self.root / "blobs"
        self.snapshot_dir = self.root / "snapshots"
        self.output_dir = self.root / "outputs"
        self.lock_path = self.root / "locks" / "commit.lock"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def scene_exists(self) -> bool:
        return self.scene_path.exists()

    def state_exists(self) -> bool:
        return self.state_path.exists()

    def load_scene_document(self) -> dict[str, Any] | None:
        if not self.scene_path.exists():
            return None
        try:
            document = read_json(self.scene_path)
        except (OSError, json.JSONDecodeError) as exc:
            fail(E_DEPENDENCY_STALE, "scene document is unreadable", path=str(self.scene_path), error=str(exc))
        if not isinstance(document, dict) or "scene" not in document or "versions" not in document:
            fail(E_DEPENDENCY_STALE, "scene document has an invalid shape", path=str(self.scene_path))
        return document

    def save_scene_document(self, scene: dict[str, Any], versions: dict[str, int]) -> None:
        self.ensure()
        atomic_write_json(self.scene_path, {"scene": scene, "versions": versions})

    def load_state(self) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        try:
            state = read_json(self.state_path)
        except (OSError, json.JSONDecodeError) as exc:
            fail(E_DEPENDENCY_STALE, "graph state is unreadable", path=str(self.state_path), error=str(exc))
        if not isinstance(state, dict) or state.get("schema") != SCHEMA_VERSION:
            fail(E_DEPENDENCY_STALE, "graph state schema is unsupported", path=str(self.state_path))
        return state

    def save_state(self, state: dict[str, Any]) -> None:
        self.ensure()
        atomic_write_json(self.state_path, state)

    def blob_path(self, artifact_hash: str) -> Path:
        return self.blob_dir / f"{artifact_hash}.bin"

    def write_blob(self, data: bytes) -> str:
        self.ensure()
        artifact_hash = sha256_bytes(data)
        path = self.blob_path(artifact_hash)
        if path.exists():
            if sha256_bytes(path.read_bytes()) != artifact_hash:
                fail(E_DEPENDENCY_STALE, "existing content-addressed blob is corrupt", artifact_hash=artifact_hash)
            return artifact_hash
        atomic_write_bytes(path, data)
        if sha256_bytes(path.read_bytes()) != artifact_hash:
            fail(E_DEPENDENCY_STALE, "blob verification failed after write", artifact_hash=artifact_hash)
        return artifact_hash

    def read_blob(self, artifact_hash: str, *, node_type: str | None = None) -> bytes:
        path = self.blob_path(artifact_hash)
        if not path.exists():
            self._artifact_error(node_type, "artifact blob is missing", artifact_hash=artifact_hash)
        data = path.read_bytes()
        if sha256_bytes(data) != artifact_hash:
            self._artifact_error(node_type, "artifact hash does not match its content", artifact_hash=artifact_hash)
        return data

    def node_content_hash(self, payload: dict[str, Any], artifact: bytes) -> str:
        return content_hash(payload, artifact)

    def verify_node(self, node: dict[str, Any]) -> bytes:
        node_type = str(node.get("type"))
        artifact_hash = str(node.get("artifact_hash", ""))
        artifact = self.read_blob(artifact_hash, node_type=node_type)
        computed = content_hash(node.get("payload", {}), artifact)
        if computed != node.get("content_hash"):
            self._artifact_error(
                node_type,
                "node content hash mismatch",
                node_id=node.get("id"),
                expected=node.get("content_hash"),
                actual=computed,
            )
        return artifact

    def verify_state_artifacts(self, state: dict[str, Any]) -> dict[str, bytes]:
        blobs: dict[str, bytes] = {}
        nodes = state.get("nodes", {})
        if not isinstance(nodes, dict):
            fail(E_DEPENDENCY_STALE, "graph state nodes are invalid")
        for node_id in sorted(nodes):
            node = nodes[node_id]
            blobs[node_id] = self.verify_node(node)
        for node_id in sorted(nodes):
            node = nodes[node_id]
            for dependency_id, expected_hash in node.get("input_hashes", {}).items():
                actual = nodes.get(dependency_id, {}).get("content_hash")
                if actual != expected_hash:
                    fail(
                        E_DEPENDENCY_STALE,
                        "cached node input hash does not match its upstream node",
                        node_id=node_id,
                        dependency_id=dependency_id,
                        expected=expected_hash,
                        actual=actual,
                    )
        return blobs

    def save_snapshot(self, snapshot: dict[str, Any]) -> tuple[str, str]:
        self.ensure()
        data = canonical_json_bytes(snapshot)
        snapshot_hash = sha256_bytes(data)
        revision = int(snapshot.get("revision", 0))
        path = self.snapshot_dir / f"snapshot-r{revision:06d}-{snapshot_hash[:16]}.json"
        if path.exists():
            existing = path.read_bytes()
            if existing != data:
                fail(
                    E_DEPENDENCY_STALE,
                    "immutable snapshot path already contains different bytes",
                    path=str(path),
                )
        else:
            atomic_write_bytes(path, data, readonly=True)
        latest = self.snapshot_dir / "LATEST"
        atomic_write_bytes(latest, (path.name + "\n").encode("ascii"))
        return path.name, snapshot_hash

    def load_latest_snapshot(self) -> tuple[dict[str, Any], str]:
        latest_path = self.snapshot_dir / "LATEST"
        if not latest_path.exists():
            fail(E_DEPENDENCY_STALE, "no diagnostic snapshot is available")
        name = latest_path.read_text(encoding="ascii").strip()
        if not name:
            fail(E_DEPENDENCY_STALE, "latest snapshot pointer is empty")
        path = self.snapshot_dir / name
        if not path.exists():
            fail(E_DEPENDENCY_STALE, "latest snapshot is missing", path=str(path))
        data = path.read_bytes()
        actual_hash = sha256_bytes(data)
        expected_prefix = path.stem.rsplit("-", 1)[-1]
        if not actual_hash.startswith(expected_prefix):
            fail(E_DEPENDENCY_STALE, "immutable snapshot hash mismatch", path=str(path), expected_prefix=expected_prefix, actual=actual_hash)
        try:
            snapshot = json.loads(data)
        except json.JSONDecodeError as exc:
            fail(E_DEPENDENCY_STALE, "snapshot JSON is invalid", path=str(path), error=str(exc))
        return snapshot, actual_hash

    def export_output(self, name: str, data: bytes) -> Path:
        self.ensure()
        safe_name = Path(name).name
        path = self.output_dir / safe_name
        atomic_write_bytes(path, data)
        return path

    def _artifact_error(self, node_type: str | None, message: str, **details: Any) -> None:
        if node_type == "tile":
            fail(E_TILE_HASH, message, **details)
        fail(E_DEPENDENCY_STALE, message, **details)


def make_node_record(
    *,
    node_id: str,
    node_type: str,
    payload: dict[str, Any],
    artifact: bytes,
    artifact_hash: str,
    dependencies: list[str],
    input_hashes: dict[str, str],
    param_stamps: dict[str, int],
    trigger_keys: list[str],
    revision: int,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "artifact_hash": artifact_hash,
        "content_hash": content_hash(payload, artifact),
        "payload": copy.deepcopy(payload),
        "dependencies": sorted(set(dependencies)),
        "input_hashes": dict(sorted(input_hashes.items())),
        "param_stamps": dict(sorted(param_stamps.items())),
        "trigger_keys": sorted(set(trigger_keys)),
        "computed_revision": int(revision),
        "last_reuse_source_revision": None,
        "last_build_status": "recomputed",
    }
