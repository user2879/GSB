"""Content-addressed node artifact cache."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import Artifact
from .util import atomic_write_json


def _node_directory(root: Path, node_id: str) -> Path:
    import hashlib

    digest = hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:24]
    return root / digest


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as handle:
            np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


class ArtifactCache:
    def __init__(self, workspace: str | os.PathLike[str]):
        self.root = Path(workspace) / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def has(self, node_id: str, cache_key: str) -> bool:
        directory = _node_directory(self.root, node_id)
        return (directory / cache_key / "metadata.json").is_file() and (directory / cache_key / "values.npy").is_file()

    def save(self, artifact: Artifact) -> None:
        directory = _node_directory(self.root, artifact.node_id) / artifact.cache_key
        directory.mkdir(parents=True, exist_ok=True)
        _write_npy_atomic(directory / "values.npy", artifact.values)
        if artifact.raw_values is not None:
            _write_npy_atomic(directory / "raw.npy", artifact.raw_values)
        metadata = {
            "node_id": artifact.node_id,
            "op": artifact.op,
            "cache_key": artifact.cache_key,
            "values_hash": artifact.values_hash,
            "raw_values_hash": artifact.raw_values_hash,
            "has_raw_values": artifact.raw_values is not None,
            "metadata": artifact.metadata,
        }
        atomic_write_json(directory / "metadata.json", metadata)

    def load(self, node_id: str, cache_key: str) -> Artifact | None:
        directory = _node_directory(self.root, node_id) / cache_key
        metadata_path = directory / "metadata.json"
        values_path = directory / "values.npy"
        if not metadata_path.is_file() or not values_path.is_file():
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            values = np.load(values_path, allow_pickle=False)
            artifact = Artifact(
                node_id=str(metadata["node_id"]),
                op=str(metadata["op"]),
                values=np.ascontiguousarray(values),
                cache_key=str(metadata["cache_key"]),
                metadata=dict(metadata["metadata"]),
            )
            if artifact.values_hash != metadata.get("values_hash"):
                return None
            if metadata.get("has_raw_values"):
                raw_path = directory / "raw.npy"
                if not raw_path.is_file():
                    return None
                artifact.raw_values = np.ascontiguousarray(np.load(raw_path, allow_pickle=False))
                if artifact.raw_values_hash != metadata.get("raw_values_hash"):
                    return None
            return artifact
        except (OSError, ValueError, KeyError, TypeError):
            return None
