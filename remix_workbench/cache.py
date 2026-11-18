from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Artifact:
    key: str
    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]


class ArtifactStore:
    """Content-addressed artifacts whose validity is gated by snapshot manifests."""

    def __init__(self, root: str | Path, trusted_keys: set[str] | None = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.trusted_keys = set(trusted_keys or set())
        self._memory: dict[str, Artifact] = {}
        self.commits = 0

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    def is_trusted(self, key: str) -> bool:
        return key in self.trusted_keys and self.path_for(key).is_file()

    def trust(self, keys: set[str]) -> None:
        self.trusted_keys.update(keys)

    def get(self, key: str) -> Artifact | None:
        if key in self._memory:
            return self._memory[key]
        if not self.is_trusted(key):
            return None
        path = self.path_for(key)
        try:
            with np.load(path, allow_pickle=False) as archive:
                raw_meta = archive["__meta__"]
                meta = json.loads(str(raw_meta.item()))
                arrays = {
                    name: archive[name]
                    for name in archive.files
                    if name != "__meta__"
                }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        artifact = Artifact(key=key, meta=meta, arrays=arrays)
        self._memory[key] = artifact
        return artifact

    def put(
        self,
        key: str,
        *,
        meta: dict[str, Any],
        arrays: dict[str, np.ndarray] | None = None,
    ) -> Artifact:
        arrays = arrays or {}
        meta_array = np.array(
            json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        )
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        with temp.open("wb") as handle:
            np.savez(
                handle,
                __meta__=meta_array,
                **{
                    name: np.ascontiguousarray(values)
                    for name, values in sorted(arrays.items())
                },
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        artifact = Artifact(key=key, meta=meta, arrays=arrays)
        self._memory[key] = artifact
        self.commits += 1
        return artifact

