from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_json_bytes, sha256_bytes
from .errors import PipelineError
from .geometry import BlockPlan

SCHEMA_VERSION = 1


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    atomic_write_bytes(path, data)


def blocks_dir(state_dir: Path) -> Path:
    return state_dir / "blocks"


def archive_dir(state_dir: Path) -> Path:
    return state_dir / "archive"


def manifest_path(state_dir: Path) -> Path:
    return state_dir / "manifest.json"


def interrupt_path(state_dir: Path) -> Path:
    return state_dir / "INTERRUPTED.json"


def block_path(state_dir: Path, plan: BlockPlan) -> Path:
    return blocks_dir(state_dir) / f"block_r{plan.row:04d}_c{plan.col:04d}.npz"


@dataclass(frozen=True)
class BlockRecord:
    path: Path
    gray: np.ndarray
    depth: np.ndarray
    meta: dict[str, Any]

    @property
    def version_id(self) -> str:
        return str(self.meta["version_id"])

    @property
    def edit_digest(self) -> str:
        return str(self.meta["edit_digest"])

    @property
    def has_nonzero_samples(self) -> bool:
        return bool(self.meta["has_nonzero_samples"])


@dataclass(frozen=True)
class CacheIssue:
    code: str
    path: Path
    message: str


def _record_digest(meta_without_digest: dict[str, Any], gray: np.ndarray, depth: np.ndarray) -> str:
    gray64 = np.ascontiguousarray(gray, dtype="<f8")
    depth64 = np.ascontiguousarray(depth, dtype="<f8")
    return sha256_bytes(
        canonical_json_bytes(meta_without_digest),
        gray64.tobytes(order="C"),
        depth64.tobytes(order="C"),
    )


def _archive_existing(path: Path, state_dir: Path, reason: str) -> None:
    if not path.exists():
        return
    destination_dir = archive_dir(state_dir) / path.stem
    destination_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = destination_dir / f"{reason}-{stamp}-{uuid.uuid4().hex[:12]}.npz"
    try:
        os.replace(path, destination)
    except OSError as exc:
        raise PipelineError(
            "E_STATE", "state_dir", f"cannot archive old block: {exc}", {"block": str(path)}
        )
    _fsync_directory(path.parent)
    _fsync_directory(destination_dir)


def write_block_record(
    state_dir: Path,
    plan: BlockPlan,
    *,
    version_id: str,
    edit_digest: str,
    has_nonzero_samples: bool,
    gray: np.ndarray,
    depth: np.ndarray,
    run_id: str,
) -> BlockRecord:
    gray64 = np.ascontiguousarray(gray, dtype="<f8")
    depth64 = np.ascontiguousarray(depth, dtype="<f8")
    expected_shape = (plan.y1 - plan.y0, plan.x1 - plan.x0)
    if gray64.shape != expected_shape or depth64.shape != expected_shape:
        raise PipelineError(
            "E_DIGEST",
            "state_dir",
            "block array shape differs from the declared block pixel count",
            {
                "expected_shape": list(expected_shape),
                "gray_shape": list(gray64.shape),
                "depth_shape": list(depth64.shape),
            },
        )
    meta_without_digest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "index": plan.index,
        "row": plan.row,
        "col": plan.col,
        "pixel_bounds": plan.pixel_bounds,
        "pixel_count": int(plan.pixel_count),
        "shape": list(expected_shape),
        "version_id": version_id,
        "edit_digest": edit_digest,
        "has_nonzero_samples": bool(has_nonzero_samples),
        "voxel_bounds": plan.voxel_bounds,
        "ray_hit_count": plan.ray_hit_count,
        "dtype": "<f8",
        "run_id": run_id,
    }
    digest = _record_digest(meta_without_digest, gray64, depth64)
    meta = dict(meta_without_digest)
    meta["payload_digest"] = digest
    meta_bytes = canonical_json_bytes(meta)

    target = block_path(state_dir, plan)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(
                handle,
                gray=gray64,
                depth=depth64,
                meta=np.frombuffer(meta_bytes, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        _archive_existing(target, state_dir, reason=version_id[:16])
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise

    return BlockRecord(path=target, gray=gray64, depth=depth64, meta=meta)


def read_block_record(path: Path, plan: BlockPlan) -> tuple[BlockRecord | None, CacheIssue | None]:
    if not path.exists():
        return None, None
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"gray", "depth", "meta"}:
                return None, CacheIssue("E_DIGEST", path, "block record has unexpected arrays")
            gray = np.asarray(data["gray"], dtype=np.float64)
            depth = np.asarray(data["depth"], dtype=np.float64)
            meta_raw = np.asarray(data["meta"], dtype=np.uint8).tobytes()
        meta = json.loads(meta_raw.decode("utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, CacheIssue("E_DIGEST", path, f"cannot read block record: {exc}")

    if not isinstance(meta, dict):
        return None, CacheIssue("E_DIGEST", path, "block metadata is not an object")
    expected_shape = (plan.y1 - plan.y0, plan.x1 - plan.x0)
    declared_count = meta.get("pixel_count")
    declared_shape = meta.get("shape")
    if (
        not isinstance(declared_count, int)
        or declared_count != plan.pixel_count
        or gray.size != declared_count
        or depth.size != declared_count
        or gray.shape != expected_shape
        or depth.shape != expected_shape
        or declared_shape != list(expected_shape)
    ):
        return None, CacheIssue("E_DIGEST", path, "declared pixel count or shape is inconsistent")
    if not np.all(np.isfinite(gray)) or not np.all(np.isfinite(depth)):
        return None, CacheIssue("E_DIGEST", path, "block arrays contain non-finite values")
    if float(np.min(gray)) < -1e-12 or float(np.max(gray)) > 1.0 + 1e-12:
        return None, CacheIssue("E_DIGEST", path, "gray values are outside [0, 1]")
    digest = meta.get("payload_digest")
    if not isinstance(digest, str):
        return None, CacheIssue("E_DIGEST", path, "block payload digest is missing")
    meta_without_digest = dict(meta)
    meta_without_digest.pop("payload_digest", None)
    actual_digest = _record_digest(meta_without_digest, gray, depth)
    if actual_digest != digest:
        return None, CacheIssue("E_DIGEST", path, "block payload digest mismatch")
    if meta.get("index") != plan.index or meta.get("row") != plan.row or meta.get("col") != plan.col:
        return None, CacheIssue("E_DIGEST", path, "block identity does not match its file location")
    return BlockRecord(path=path, gray=gray, depth=depth, meta=meta), None


def read_manifest(state_dir: Path) -> dict[str, Any] | None:
    path = manifest_path(state_dir)
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        return None
    return manifest


def write_manifest(state_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(manifest_path(state_dir), payload)


def write_interrupt_marker(
    state_dir: Path,
    *,
    run_id: str,
    version_id: str,
    completed_indices: list[int],
    pending_indices: list[int],
) -> None:
    atomic_write_json(
        interrupt_path(state_dir),
        {
            "schema_version": SCHEMA_VERSION,
            "interrupted": True,
            "run_id": run_id,
            "version_id": version_id,
            "completed_block_indices": completed_indices,
            "pending_block_indices": pending_indices,
            "exit_code": 130,
        },
    )


def clear_interrupt_marker(state_dir: Path) -> None:
    try:
        interrupt_path(state_dir).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def atomic_write_npy(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array, dtype="<f8"), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return temporary


def write_json_temp(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return temporary


def atomic_replace_bundle(temporary_by_target: dict[Path, Path]) -> None:
    """Replace a set of output files as one best-effort transaction.

    All payloads are already fsynced in the destination directories. Existing
    targets are moved aside before any new target becomes visible, and are
    restored if a replacement fails. Any exception therefore leaves either the
    complete old bundle or the complete new bundle at the canonical paths.
    """

    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for target in temporary_by_target:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
                os.replace(target, backup)
                backups[target] = backup
        for target, temporary in temporary_by_target.items():
            os.replace(temporary, target)
            installed.append(target)
        for directory in {target.parent for target in temporary_by_target}:
            _fsync_directory(directory)
        for backup in backups.values():
            try:
                backup.unlink()
            except OSError:
                pass
    except BaseException:
        for target in installed:
            try:
                target.unlink()
            except OSError:
                pass
        for target, backup in backups.items():
            try:
                if backup.exists():
                    os.replace(backup, target)
            except OSError:
                pass
        for temporary in temporary_by_target.values():
            try:
                temporary.unlink()
            except OSError:
                pass
        raise
