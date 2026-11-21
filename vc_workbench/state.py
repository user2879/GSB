"""Stage artifacts and atomically switched, self-contained snapshots."""

from __future__ import annotations

import os
import shutil
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_SNAPSHOT_STALE
from .util import (
    atomic_write_json,
    canonical_bytes,
    canonical_json,
    ensure_dir,
    read_json,
    sha256_bytes,
    sha256_value,
)


STAGE_NAMES = ("analysis", "stats", "mappings", "assembly", "ledger")


class SnapshotError(RuntimeError):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.code = E_SNAPSHOT_STALE
        self.detail = detail


@dataclass(slots=True)
class SnapshotState:
    snapshot_id: str
    manifest: dict[str, Any]
    analysis: dict[str, Any]
    stats: dict[str, Any]
    mappings: dict[str, Any]
    assembly: dict[str, Any]
    ledger: dict[str, Any]
    pcm_by_segment: dict[str, bytes] = field(default_factory=dict)
    assembly_pcm: bytes = b""

    @property
    def stage_versions(self) -> dict[str, str]:
        return dict(self.manifest.get("stage_versions", {}))


def _piece_name(segment_id: str) -> str:
    return segment_id.replace("/", "_").replace("\\", "_") + ".pcm.z"


def _write_synced(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir_best_effort(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _stage_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def commit_snapshot(
    state_root: Path,
    *,
    manifest_core: dict[str, Any],
    analysis: dict[str, Any],
    stats: dict[str, Any],
    mappings: dict[str, Any],
    assembly: dict[str, Any],
    ledger: dict[str, Any],
    pcm_by_segment: dict[str, bytes],
    assembly_pcm: bytes,
) -> SnapshotState:
    """Commit a complete snapshot and only then switch CURRENT atomically."""

    state_root = state_root.resolve()
    snapshots_dir = ensure_dir(state_root / "snapshots")
    temp_root = ensure_dir(state_root / "tmp")
    pieces: dict[str, bytes] = {
        segment_id: zlib.compress(bytes(pcm), level=9)
        for segment_id, pcm in sorted(pcm_by_segment.items())
    }
    assembly_piece = zlib.compress(bytes(assembly_pcm), level=9)
    artifacts = {
        "analysis": analysis,
        "stats": stats,
        "mappings": mappings,
        "assembly": assembly,
        "ledger": ledger,
    }
    serialized = {name: _stage_bytes(value) for name, value in artifacts.items()}
    stage_versions = {name: sha256_bytes(serialized[name]) for name in STAGE_NAMES}
    id_payload = {
        "manifest_core": manifest_core,
        "stage_versions": stage_versions,
        "pieces": {name: sha256_bytes(data) for name, data in pieces.items()},
        "assembly_piece": sha256_bytes(assembly_piece),
    }
    base_snapshot_id = sha256_value(id_payload)[:32]
    for revision in range(0, 100):
        snapshot_id = base_snapshot_id if revision == 0 else f"{base_snapshot_id}-r{revision}"
        manifest = dict(manifest_core)
        manifest["snapshot_id"] = snapshot_id
        manifest["stage_versions"] = stage_versions
        manifest["schema"] = "vc-workbench-snapshot"
        manifest["schema_version"] = 1
        manifest_bytes = _stage_bytes(manifest)
        manifest_sha = sha256_bytes(manifest_bytes)
        target = snapshots_dir / snapshot_id

        if target.exists():
            # A fully committed identical snapshot is already the strongest state.
            existing_manifest = target / "manifest.json"
            if existing_manifest.exists():
                try:
                    if sha256_bytes(existing_manifest.read_bytes()) == manifest_sha:
                        state = load_snapshot_directory(target)
                        atomic_write_json(
                            state_root / "CURRENT",
                            {"snapshot_id": snapshot_id, "manifest_sha256": manifest_sha},
                            pretty=True,
                        )
                        return state
                except Exception:
                    pass
            # Keep the corrupt/old directory visible to the old CURRENT until the
            # revised snapshot below is complete, then switch the pointer.
            continue

        temp_dir = Path(tempfile.mkdtemp(prefix=f"{snapshot_id}.", suffix=".tmp", dir=temp_root))
        try:
            piece_dir = ensure_dir(temp_dir / "pieces")
            for name, data in pieces.items():
                _write_synced(piece_dir / _piece_name(name), data)
            _write_synced(temp_dir / "assembly.pcm.z", assembly_piece)
            for name in STAGE_NAMES:
                _write_synced(temp_dir / f"{name}.json", serialized[name])
            _write_synced(temp_dir / "manifest.json", manifest_bytes)
            _fsync_dir_best_effort(piece_dir)
            _fsync_dir_best_effort(temp_dir)
            temp_dir.rename(target)
            _fsync_dir_best_effort(snapshots_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        # This single os.replace is the visibility boundary for the whole snapshot.
        atomic_write_json(
            state_root / "CURRENT",
            {"snapshot_id": snapshot_id, "manifest_sha256": manifest_sha},
            pretty=True,
        )
        return load_snapshot_directory(target)
    raise SnapshotError(f"cannot allocate a complete snapshot id for {base_snapshot_id}")


def _verify_stage(path: Path, expected: str, name: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise SnapshotError(f"missing {name} stage: {exc}") from exc
    actual = sha256_bytes(data)
    if actual != expected:
        raise SnapshotError(f"{name} stage checksum mismatch: expected {expected}, actual {actual}")
    return data


def load_snapshot_directory(snapshot_dir: Path) -> SnapshotState:
    try:
        manifest = read_json(snapshot_dir / "manifest.json")
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "vc-workbench-snapshot":
        raise SnapshotError("manifest schema is missing or unsupported")
    versions = manifest.get("stage_versions")
    if not isinstance(versions, dict) or any(name not in versions for name in STAGE_NAMES):
        raise SnapshotError("manifest has incomplete stage versions")
    decoded: dict[str, Any] = {}
    for name in STAGE_NAMES:
        data = _verify_stage(snapshot_dir / f"{name}.json", str(versions[name]), name)
        try:
            import json

            decoded[name] = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SnapshotError(f"{name} stage is not valid JSON: {exc}") from exc

    pcm_by_segment: dict[str, bytes] = {}
    for segment_id, mapping in sorted(decoded["mappings"].get("segments", {}).items()):
        if mapping.get("mapping_status") != "mapped":
            continue
        piece_name = mapping.get("piece_file")
        if not piece_name:
            raise SnapshotError(f"mapping for {segment_id} has no piece file")
        piece_path = snapshot_dir / piece_name
        try:
            compressed = piece_path.read_bytes()
            pcm = zlib.decompress(compressed)
        except (OSError, zlib.error) as exc:
            raise SnapshotError(f"cannot load mapping piece {segment_id}: {exc}") from exc
        expected_pcm = str(mapping.get("pcm_sha256", ""))
        actual_pcm = sha256_bytes(pcm)
        if actual_pcm != expected_pcm:
            raise SnapshotError(
                f"mapping piece checksum mismatch for {segment_id}: expected {expected_pcm}, actual {actual_pcm}"
            )
        pcm_by_segment[segment_id] = pcm
    try:
        assembly_pcm = zlib.decompress((snapshot_dir / "assembly.pcm.z").read_bytes())
    except (OSError, zlib.error) as exc:
        raise SnapshotError(f"cannot load assembly PCM: {exc}") from exc
    expected_assembly = decoded["assembly"].get("pcm_sha256")
    if expected_assembly is not None and sha256_bytes(assembly_pcm) != expected_assembly:
        raise SnapshotError("assembly PCM checksum mismatch")
    return SnapshotState(
        snapshot_id=str(manifest.get("snapshot_id")),
        manifest=manifest,
        analysis=decoded["analysis"],
        stats=decoded["stats"],
        mappings=decoded["mappings"],
        assembly=decoded["assembly"],
        ledger=decoded["ledger"],
        pcm_by_segment=pcm_by_segment,
        assembly_pcm=assembly_pcm,
    )


def load_current(state_root: Path) -> SnapshotState | None:
    current_path = state_root / "CURRENT"
    if not current_path.exists():
        return None
    try:
        current = read_json(current_path)
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"cannot read CURRENT pointer: {exc}") from exc
    if not isinstance(current, dict):
        raise SnapshotError("CURRENT pointer must be an object")
    snapshot_id = current.get("snapshot_id")
    expected_manifest_sha = current.get("manifest_sha256")
    if not isinstance(snapshot_id, str) or not isinstance(expected_manifest_sha, str):
        raise SnapshotError("CURRENT pointer is incomplete")
    snapshot_dir = state_root / "snapshots" / snapshot_id
    if not snapshot_dir.is_dir():
        raise SnapshotError(f"current snapshot directory is missing: {snapshot_id}")
    try:
        manifest_bytes = (snapshot_dir / "manifest.json").read_bytes()
    except OSError as exc:
        raise SnapshotError(f"current manifest is unreadable: {exc}") from exc
    actual_manifest_sha = sha256_bytes(manifest_bytes)
    if actual_manifest_sha != expected_manifest_sha:
        raise SnapshotError("current manifest checksum does not match CURRENT pointer")
    state = load_snapshot_directory(snapshot_dir)
    if state.snapshot_id != snapshot_id:
        raise SnapshotError("snapshot manifest id does not match CURRENT pointer")
    return state


def recovery_point(state: SnapshotState | None) -> dict[str, Any]:
    if state is None:
        return {"present": False, "recoverable": False}
    manifest = state.manifest
    return {
        "present": True,
        "recoverable": True,
        "snapshot_id": state.snapshot_id,
        "global_version": int(manifest.get("global_version", 0)),
        "command_cursor": int(manifest.get("command_cursor", 0)),
        "command_count": int(manifest.get("command_count", 0)),
        "command_complete": bool(manifest.get("command_complete", False)),
        "active_command_seq": manifest.get("active_command_seq"),
        "consumed_frame_budget": int(manifest.get("consumed_frame_budget", 0)),
        "stage_versions": dict(manifest.get("stage_versions", {})),
        "config_hash": manifest.get("config_hash"),
        "audio_fingerprint": manifest.get("audio_fingerprint"),
    }


def inspect_current(state_root: Path) -> dict[str, Any]:
    try:
        state = load_current(state_root)
    except SnapshotError as exc:
        return {
            "present": True,
            "recoverable": False,
            "code": E_SNAPSHOT_STALE,
            "detail": exc.detail,
            "recovery_point": {"present": True, "recoverable": False},
        }
    point = recovery_point(state)
    return {"present": state is not None, "recoverable": state is not None, "recovery_point": point}
