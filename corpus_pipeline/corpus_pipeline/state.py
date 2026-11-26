from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .errors import Code, fail
from .models import content_hash


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    readable = "".join(character if character.isalnum() or character in "-_." else "_" for character in value)
    return f"{readable[:48]}-{digest}"


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str).encode("utf-8") + b"\n"
    atomic_write_bytes(path, payload)


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        fail(Code.STATE_CORRUPT, "state file does not exist", path=str(path))
    except (OSError, json.JSONDecodeError) as exc:
        fail(Code.STATE_CORRUPT, "state file is unreadable or corrupt", path=str(path), error=str(exc))


class StateStore:
    """Crash-safe checkpoint, ledger, and whole-snapshot publication store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.state_dir = self.root / "state"
        self.snapshots_dir = self.state_dir / "snapshots"
        self.work_dir = self.state_dir / "work"
        self.current_pointer = self.state_dir / "CURRENT"
        self.ledger_path = self.state_dir / "ledger.jsonl"
        self.lock_path = self.state_dir / ".pipeline.lock"
        self._ledger_lock = threading.Lock()

    def initialize(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def exclusive_run_lock(self) -> Iterator[None]:
        self.initialize()
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def current_snapshot_name(self) -> str | None:
        try:
            text = self.current_pointer.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return text or None

    def load_current_snapshot(self) -> dict[str, Any] | None:
        name = self.current_snapshot_name()
        if not name:
            return None
        directory = self.snapshots_dir / name
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            fail(Code.STATE_CORRUPT, "CURRENT points at a snapshot without a manifest", snapshot=name)
        snapshot: dict[str, Any] = {
            "directory": str(directory),
            "name": name,
            "manifest": read_json(manifest_path),
        }
        for artifact in (
            "sampling",
            "dedup",
            "composition",
            "attribution",
            "failures",
            "receipts",
            "report",
        ):
            path = directory / f"{artifact}.json"
            if path.exists():
                snapshot[artifact] = read_json(path)
        return snapshot

    def work_path(self, run_key: str) -> Path:
        return self.work_dir / _safe_name(run_key)

    def load_work_manifest(self, run_key: str) -> dict[str, Any] | None:
        path = self.work_path(run_key) / "manifest.json"
        if not path.exists():
            return None
        return read_json(path)

    def save_work_manifest(self, run_key: str, manifest: Mapping[str, Any]) -> None:
        directory = self.work_path(run_key)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "manifest.json", manifest)

    def save_work_stage(self, run_key: str, stage: str, value: Any) -> None:
        directory = self.work_path(run_key) / "stages"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / f"{stage}.json", value)

    def load_work_stage(self, run_key: str, stage: str) -> Any | None:
        path = self.work_path(run_key) / "stages" / f"{stage}.json"
        if not path.exists():
            return None
        return read_json(path)

    def shard_checkpoint_path(self, run_key: str, shard_id: str) -> Path:
        return self.work_path(run_key) / "dedup-shards" / f"{_safe_name(shard_id)}.json"

    def save_shard_checkpoint(self, run_key: str, shard_id: str, value: Any) -> None:
        atomic_write_json(self.shard_checkpoint_path(run_key, shard_id), value)

    def load_shard_checkpoint(self, run_key: str, shard_id: str) -> Any | None:
        path = self.shard_checkpoint_path(run_key, shard_id)
        if not path.exists():
            return None
        return read_json(path)

    def append_ledger(self, events: Sequence[Mapping[str, Any]]) -> None:
        if not events:
            return
        self.initialize()
        data = b"".join(
            json.dumps(dict(event), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8") + b"\n"
            for event in events
        )
        with self._ledger_lock:
            with self.ledger_path.open("ab") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

    def ledger_events(self, *, run_key: str | None = None) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with self.ledger_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if run_key is None or event.get("run_key") == run_key:
                        events.append(event)
        except (OSError, json.JSONDecodeError) as exc:
            fail(Code.STATE_CORRUPT, "ledger is corrupt", error=str(exc), path=str(self.ledger_path))
        return events

    def publish_snapshot(
        self,
        run_key: str,
        artifacts: Mapping[str, Any],
        manifest: Mapping[str, Any],
    ) -> str:
        self.initialize()
        sequence = 1
        existing = [path.name for path in self.snapshots_dir.glob(f"{_safe_name(run_key)[:56]}-*")]
        if existing:
            sequence = len(existing) + 1
        name = f"{_safe_name(run_key)[:56]}-{sequence:04d}"
        staging = self.snapshots_dir / f".staging-{_safe_name(run_key)[:48]}-{uuid.uuid4().hex[:8]}"
        final = self.snapshots_dir / name
        if final.exists():
            fail(Code.STATE_CORRUPT, "snapshot destination already exists", snapshot=name)
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for artifact, value in sorted(artifacts.items()):
                atomic_write_json(staging / f"{artifact}.json", value)
            atomic_write_json(staging / "manifest.json", manifest)
            _fsync_directory(staging)
            os.replace(staging, final)
            _fsync_directory(self.snapshots_dir)
            atomic_write_bytes(self.current_pointer, (name + "\n").encode("utf-8"))
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return name


def event_fingerprint(event: Mapping[str, Any]) -> str:
    return content_hash(dict(event))
