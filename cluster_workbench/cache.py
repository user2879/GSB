from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .errors import E_CACHE_CORRUPT, Issue
from .util import atomic_write_text, canonical_json, hash_array, to_jsonable


CACHE_SCHEMA_VERSION = 1


class CacheStore:
    """Small content-addressed cache backed by atomic JSON and NPY files.

    Cache keys are supplied by the pipeline.  Each payload stores its own
    content fingerprint; loaded arrays are re-hashed so a corrupted or
    partially-written artifact becomes a cache miss rather than silent input.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, kind: str) -> Path:
        safe = "".join(character for character in kind if character.isalnum() or character in "_-")
        return self.root / safe

    def _json_path(self, kind: str, key: str) -> Path:
        return self._directory(kind) / f"{key}.json"

    def _array_path(self, kind: str, key: str) -> Path:
        return self._directory(kind) / f"{key}.npy"

    def get(self, kind: str, key: str) -> tuple[dict[str, Any] | None, list[Issue]]:
        issues: list[Issue] = []
        json_path = self._json_path(kind, key)
        if not json_path.exists():
            return None, issues
        try:
            import json

            with json_path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except Exception as exc:
            return None, [
                Issue(
                    E_CACHE_CORRUPT,
                    f"cannot read cache metadata: {exc}",
                    severity="warning",
                    context={"scope": "cache", "kind": kind, "cache_key": key},
                )
            ]
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != CACHE_SCHEMA_VERSION
            or record.get("kind") != kind
            or record.get("cache_key") != key
            or not isinstance(record.get("payload"), dict)
        ):
            return None, [
                Issue(
                    E_CACHE_CORRUPT,
                    "cache metadata schema/key mismatch",
                    severity="warning",
                    context={"scope": "cache", "kind": kind, "cache_key": key},
                )
            ]
        array_spec = record.get("array")
        if array_spec is not None:
            try:
                array_path = self.root / str(array_spec["file"])
                array = np.load(array_path, allow_pickle=False)
                expected_hash = str(array_spec["sha256"])
                actual_hash = hash_array(array)
                if actual_hash != expected_hash:
                    raise ValueError("cache array fingerprint mismatch")
                record["array_value"] = array
            except Exception as exc:
                return None, [
                    Issue(
                        E_CACHE_CORRUPT,
                        f"cannot read cache array: {exc}",
                        severity="warning",
                        context={"scope": "cache", "kind": kind, "cache_key": key},
                    )
                ]
        return record, issues

    def put(
        self,
        kind: str,
        key: str,
        payload: dict[str, Any],
        *,
        array: np.ndarray | None = None,
    ) -> None:
        directory = self._directory(kind)
        directory.mkdir(parents=True, exist_ok=True)
        array_spec: dict[str, Any] | None = None
        if array is not None:
            array_path = self._array_path(kind, key)
            self._atomic_save_npy(array_path, array)
            array_spec = {
                "file": array_path.relative_to(self.root).as_posix(),
                "sha256": hash_array(array),
                "dtype": array.dtype.str,
                "shape": list(array.shape),
            }
        record = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "kind": kind,
            "cache_key": key,
            "payload": to_jsonable(payload),
            "array": array_spec,
        }
        atomic_write_text(self._json_path(kind, key), canonical_json(record))

    def _atomic_save_npy(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for kind_dir in self.root.iterdir():
            if not kind_dir.is_dir():
                continue
            kind = kind_dir.name
            counts[kind] = sum(1 for item in kind_dir.glob("*.json") if item.is_file())
        return dict(sorted(counts.items()))
