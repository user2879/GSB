"""Deterministic hashing, atomic file writes, and cross-platform file locks."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_hash(payload: Any, artifact: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(payload))
    digest.update(b"\x00artifact\x00")
    digest.update(artifact)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes, *, readonly: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if readonly and path.exists():
            with contextlib.suppress(OSError):
                os.chmod(path, 0o666)
        os.replace(temp_path, path)
        if readonly:
            with contextlib.suppress(OSError):
                os.chmod(path, 0o444)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def atomic_write_json(path: Path, value: Any, *, readonly: bool = False) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value), readonly=readonly)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class FileLock:
    """Advisory process lock; serializes parameter commits and state writes."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None
        self._thread_lock: threading.Lock | None = None

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.path.resolve())
        with _THREAD_LOCKS_GUARD:
            self._thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())
        self._thread_lock.acquire()
        try:
            self._handle = self.path.open("a+b")
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                self._handle.write(b"\0")
                self._handle.flush()
                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        except BaseException:
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            self._thread_lock.release()
            self._thread_lock = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._handle is None:
            if self._thread_lock is not None:
                self._thread_lock.release()
                self._thread_lock = None
            return
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            with contextlib.suppress(OSError):
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            with contextlib.suppress(OSError):
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
        if self._thread_lock is not None:
            self._thread_lock.release()
            self._thread_lock = None


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    with FileLock(path):
        yield
