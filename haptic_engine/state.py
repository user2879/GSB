"""Frame-block persistence, version identification and digest verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from .errors import E_STATE, HapticError
from .models import Manifest
from .packing import Block, block_from_dict

VERSION_FILE = "version.json"
INTERRUPT_FILE = "interrupt.json"
ARCHIVE_DIR = "archive"


def canonical_json(payload: object) -> str:
    """Stable JSON text used for every digest so results are reproducible."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_version_id(manifest: Manifest) -> str:
    """Version digest over normalized device/event/anchor declarations."""
    normalized = {
        "devices": [
            {
                "device_id": device.device_id,
                "sample_rate_hz": device.sample_rate_hz,
                "frame_duration_ms": device.frame_duration_ms,
                "min_intensity": device.min_intensity,
                "max_intensity": device.max_intensity,
                "levels": device.levels,
                "max_commands_per_frame": device.max_commands_per_frame,
            }
            for device in sorted(manifest.devices, key=lambda item: item.device_id)
        ],
        "events": [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "start_ms": event.start_ms,
                "duration_ms": event.duration_ms,
                "target_device": event.target_device,
                "curve": [
                    {"time_ms": point.time_ms, "intensity": point.intensity}
                    for point in event.curve
                ],
            }
            for event in sorted(manifest.events, key=lambda item: item.event_id)
        ],
        "sync_anchors": [
            {
                "anchor_id": anchor.anchor_id,
                "time_ms": anchor.time_ms,
                "device_ids": list(anchor.device_ids),
                "reference_device": anchor.reference_device,
                "allowed_samples": anchor.allowed_samples,
            }
            for anchor in sorted(manifest.sync_anchors, key=lambda item: item.anchor_id)
        ],
    }
    digest = hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def block_digest(block: Block) -> str:
    payload = {
        "block_index": block.block_index,
        "nominal_frame": block.nominal_frame,
        "is_overflow": block.is_overflow,
        "frame_start_sample": block.frame_start_sample,
        "frame_end_sample": block.frame_end_sample,
        "commands": [command.to_dict() for command in block.commands],
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def atomic_write(path: Path, payload: dict) -> None:
    """Write via a same-directory temporary file then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    with open(temp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


class StateStore:
    """Manages the state directory for one synthesis run."""

    def __init__(self, manifest: Manifest) -> None:
        self.root = Path(manifest.state_dir)
        self.version_id = compute_version_id(manifest)
        self.version_file = self.root / VERSION_FILE
        self.archive_dir = self.root / ARCHIVE_DIR

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.version_file.exists():
            try:
                stored = json.loads(self.version_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HapticError(E_STATE, "unreadable version file", path=str(self.version_file)) from exc
            if stored.get("version_id") != self.version_id:
                self._archive_old_version(stored.get("version_id", "unknown"))
        atomic_write(
            self.version_file,
            {"version_id": self.version_id, "format": "haptic-state-v1"},
        )

    def _archive_old_version(self, old_version: str) -> None:
        """Move pre-existing snapshots aside instead of overwriting them."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        tag = old_version.replace(":", "_")
        destination = self.archive_dir / tag
        suffix = 1
        while destination.exists():
            suffix += 1
            destination = self.archive_dir / f"{tag}__{suffix}"
        destination.mkdir(parents=True, exist_ok=False)
        for child in list(self.root.iterdir()):
            if child.name == ARCHIVE_DIR:
                continue
            shutil.move(str(child), str(destination / child.name))

    def device_dir(self, device_id: str) -> Path:
        path = self.root / "blocks" / device_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def block_path(self, device_id: str, block_index: int) -> Path:
        return self.device_dir(device_id) / f"block_{block_index:06d}.json"

    def write_block(self, device_id: str, block: Block, total_samples: int) -> None:
        payload = {
            "version_id": self.version_id,
            "device_id": device_id,
            "total_samples": total_samples,
            "digest": block_digest(block),
            "block": block.to_dict(),
        }
        atomic_write(self.block_path(device_id, block.block_index), payload)

    def load_block(self, device_id: str, block_index: int) -> dict | None:
        path = self.block_path(device_id, block_index)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version_id") != self.version_id:
            return None
        try:
            block = block_from_dict(payload["block"])
        except (KeyError, TypeError):
            return None
        if block.block_index != block_index:
            return None
        declared_count = payload["block"].get("command_count")
        if declared_count != block.command_count:
            return None
        if payload.get("digest") != block_digest(block):
            return None
        return {"block": block, "total_samples": payload.get("total_samples")}

    def verify_or_discard(self, device_id: str, block_index: int) -> Block | None:
        """Return a verified block or remove its corrupt snapshot."""
        path = self.block_path(device_id, block_index)
        loaded = self.load_block(device_id, block_index)
        if loaded is None:
            if path.exists():
                path.unlink()
            return None
        return loaded["block"]

    def list_block_indices(self, device_id: str) -> list[int]:
        directory = self.root / "blocks" / device_id
        if not directory.exists():
            return []
        indices: list[int] = []
        for child in directory.glob("block_*.json"):
            try:
                indices.append(int(child.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(indices)

    def record_interrupt(self, signal_name: str, signum: int) -> None:
        existing = self.read_interrupt_count()
        atomic_write(
            self.root / INTERRUPT_FILE,
            {
                "interrupt_count": existing + 1,
                "last_signal": signal_name,
                "last_signum": signum,
                "pending_resume": True,
            },
        )

    def read_interrupt_count(self) -> int:
        path = self.root / INTERRUPT_FILE
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        count = payload.get("interrupt_count")
        return count if isinstance(count, int) and count >= 0 else 0

    def has_pending_resume(self) -> bool:
        path = self.root / INTERRUPT_FILE
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(payload.get("pending_resume", False))

    def clear_pending_resume(self) -> None:
        path = self.root / INTERRUPT_FILE
        if path.exists():
            atomic_write(
                path,
                {
                    "interrupt_count": self.read_interrupt_count(),
                    "last_signal": None,
                    "last_signum": None,
                    "pending_resume": False,
                },
            )
