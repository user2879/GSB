"""Version-scoped, checksummed block persistence."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._numpy import np
from .errors import EngineError
from .spec import Spec
from .util import atomic_write_json, ensure_dir, read_json
from .versioning import block_digest


FORMAT_VERSION = 1


@dataclass(frozen=True)
class StoredBlock:
    index: int
    frames: "np.ndarray"
    metadata: dict[str, Any]
    digest: str


@dataclass(frozen=True)
class BlockRecord:
    index: int
    metadata: dict[str, Any]
    digest: str

    @classmethod
    def from_stored(cls, block: StoredBlock) -> "BlockRecord":
        return cls(block.index, block.metadata, block.digest)

    @property
    def frame_count(self) -> int:
        return int(self.metadata["frame_count"])


class StateStore:
    def __init__(self, spec: Spec, version: str) -> None:
        self.spec = spec
        self.version = version
        self.root = spec.runtime.state_dir / version
        self.blocks_dir = self.root / "blocks"
        self.manifest_path = self.root / "manifest.json"
        self.interrupted_path = self.root / "interrupted.json"
        ensure_dir(self.blocks_dir)
        self._cleanup_stale_temp_files()

    def _cleanup_stale_temp_files(self) -> None:
        for path in self.blocks_dir.glob(".*.tmp.npz"):
            try:
                path.unlink()
            except OSError:
                pass

    def block_path(self, index: int) -> Path:
        return self.blocks_dir / f"{index:06d}.npz"

    def expected_block_frames(self, index: int) -> int:
        total_frames = self.spec.total_frames
        start = index * self.spec.frames_per_block
        end = min(total_frames, start + self.spec.frames_per_block)
        return max(0, end - start)

    def expected_block_range(self, index: int) -> tuple[int, int]:
        total_frames = self.spec.total_frames
        start = index * self.spec.frames_per_block
        return start, min(total_frames, start + self.spec.frames_per_block)

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            value = read_json(self.manifest_path)
        except (OSError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        return value

    def base_manifest(self, complete: bool = False) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "version": self.version,
            "frame_ms": self.spec.frame.frame_ms,
            "block_ms": self.spec.runtime.block_ms,
            "frames_per_block": self.spec.frames_per_block,
            "total_frames": self.spec.total_frames,
            "total_blocks": self.spec.total_blocks,
            "complete": complete,
            "blocks": {},
        }

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["complete"] = bool(manifest.get("complete", False))
        manifest["complete_blocks"] = len(manifest.get("blocks", {}))
        atomic_write_json(self.manifest_path, manifest)

    def record_block(self, manifest: dict[str, Any], block: StoredBlock) -> dict[str, Any]:
        blocks = dict(manifest.get("blocks", {}))
        blocks[str(block.index)] = {
            "digest": block.digest,
            "frame_count": int(block.frames.shape[0]),
        }
        manifest = {**manifest, "blocks": blocks, "complete": False}
        self.write_manifest(manifest)
        return manifest

    def save_block(
        self,
        index: int,
        frames: "np.ndarray",
        metadata: dict[str, Any],
    ) -> StoredBlock:
        base_meta = {**metadata, "version": self.version}
        digest = block_digest(self.version, index, frames, base_meta)
        stored_meta = {**base_meta, "digest": digest}
        destination = self.block_path(index)
        temp = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
        )
        try:
            with temp.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    frames=np.asarray(frames, dtype=np.uint8),
                    metadata=np.asarray(json.dumps(stored_meta, ensure_ascii=False, sort_keys=True)),
                    digest=np.asarray(digest),
                    version=np.asarray(self.version),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, destination)
            _fsync_directory(destination.parent)
        finally:
            if temp.exists():
                temp.unlink()
        return StoredBlock(index, np.asarray(frames, dtype=np.uint8), stored_meta, digest)

    def read_block(self, index: int) -> tuple[StoredBlock | None, str | None]:
        path = self.block_path(index)
        if not path.exists():
            return None, "missing"
        try:
            with np.load(path, allow_pickle=False) as archive:
                frames = np.asarray(archive["frames"])
                metadata = json.loads(str(archive["metadata"].item()))
                digest = str(archive["digest"].item())
                version = str(archive["version"].item())
        except Exception:
            return None, "unreadable"

        if version != self.version or metadata.get("version") != self.version:
            return None, "version_mismatch"
        if not isinstance(metadata, dict):
            return None, "metadata_invalid"
        expected_frames = self.expected_block_frames(index)
        start_frame, end_frame = self.expected_block_range(index)
        if frames.ndim != 3:
            return None, "shape_invalid"
        if int(frames.shape[0]) != expected_frames:
            return None, "frame_count_mismatch"
        if int(frames.shape[1]) != self.spec.frame.height or int(frames.shape[2]) != self.spec.frame.width:
            return None, "frame_size_mismatch"
        if metadata.get("index") != index or metadata.get("frame_count") != expected_frames:
            return None, "metadata_frame_count_mismatch"
        if metadata.get("start_frame") != start_frame or metadata.get("end_frame") != end_frame:
            return None, "metadata_range_mismatch"
        base_meta = {key: value for key, value in metadata.items() if key != "digest"}
        computed = block_digest(self.version, index, frames, base_meta)
        if digest != computed or metadata.get("digest") != computed:
            return None, "digest_mismatch"
        return StoredBlock(index, frames, metadata, computed), None

    def quarantine_invalid(self, index: int) -> None:
        path = self.block_path(index)
        if not path.exists():
            return
        target = path.with_name(f"{path.name}.invalid")
        try:
            os.replace(path, target)
        except OSError:
            try:
                path.unlink()
            except OSError:
                pass

    def write_interrupted(self, signal_number: int, completed_blocks: list[int]) -> None:
        atomic_write_json(
            self.interrupted_path,
            {
                "version": self.version,
                "signal": signal_number,
                "completed_blocks": completed_blocks,
                "message": "Termination accepted at a millisecond block boundary",
            },
        )

    def clear_interrupted(self) -> None:
        try:
            self.interrupted_path.unlink()
        except FileNotFoundError:
            pass

    def finalize_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        finalized = {**manifest, "complete": True}
        self.write_manifest(finalized)
        return finalized


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
