"""Filesystem primitives: exclusive locks and validated atomic replacement."""

from __future__ import annotations

import json
import hashlib
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Callable

from .canonical import CanonicalError, canonical_json_bytes, sha256_hex


class LockConflict(RuntimeError):
    """Raised when another live writer owns the workspace lock."""


class AtomicWriteError(OSError):
    """Raised when a validated atomic write cannot be completed."""


def _crash_at(name: str) -> None:
    if os.environ.get("BACKUP_ENGINE_CRASH_AT") == name:
        # Deliberate process death for crash-boundary tests. No cleanup or
        # report is promised after this point.
        os._exit(94)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                process_query_limited_information, False, pid
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return ctypes.windll.kernel32.GetLastError() == 5
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


class WorkspaceLock:
    """Single-writer lock using exclusive file creation.

    A dead owner's lock is reclaimed. A malformed or live lock is treated as a
    conflict, never silently removed.
    """

    def __init__(self, path: Path):
        self.path = path
        self.token: str | None = None

    def _read_owner(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            token = secrets.token_hex(16)
            payload = canonical_json_bytes(
                {"pid": os.getpid(), "thread": threading.get_ident(), "token": token}
            )
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                owner = self._read_owner()
                if owner is None:
                    raise LockConflict(f"workspace lock exists and is unreadable: {self.path}")
                owner_pid = owner.get("pid")
                if isinstance(owner_pid, int) and not _pid_alive(owner_pid):
                    try:
                        self.path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        raise LockConflict(str(exc)) from exc
                    continue
                raise LockConflict(f"workspace is locked by pid {owner_pid!r}")
            try:
                with os.fdopen(fd, "wb", closefd=True) as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                try:
                    self.path.unlink()
                except OSError:
                    pass
                raise
            self.token = token
            return
        raise LockConflict("unable to acquire workspace lock")

    def release(self) -> None:
        if self.token is None:
            return
        owner = self._read_owner()
        if owner is not None and owner.get("token") == self.token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self.token = None

    def __enter__(self) -> "WorkspaceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.release()


def cleanup_orphan_temps(root: Path) -> None:
    """Remove transient files left by an interrupted writer."""

    if not root.exists():
        return
    for path in root.rglob("*.tmp-*"):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except OSError:
            pass


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    before_replace: Callable[[bytes], None] | None = None,
    crash_after_temp_write: str | None = None,
    crash_before_replace: str | None = None,
) -> str:
    """Atomically replace ``path`` after validating the complete JSON bytes.

    Returns the SHA-256 of the committed bytes.
    """

    try:
        data = canonical_json_bytes(value)
    except CanonicalError as exc:
        raise AtomicWriteError(f"cannot serialize JSON for {path}: {exc}") from exc

    try:
        round_trip = json.loads(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AtomicWriteError(f"refusing to write invalid JSON for {path}: {exc}") from exc
    if canonical_json_bytes(round_trip) != data:
        raise AtomicWriteError(f"canonical round-trip mismatch for {path}")
    if before_replace is not None:
        before_replace(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / (
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{secrets.token_hex(6)}"
    )
    try:
        with open(temp_path, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if crash_after_temp_write:
            _crash_at(crash_after_temp_write)
        committed = temp_path.read_bytes()
        if committed != data or sha256_hex(committed) != sha256_hex(data):
            raise AtomicWriteError(f"temporary file verification failed for {path}")
        if crash_before_replace:
            _crash_at(crash_before_replace)
        os.replace(temp_path, path)
        if os.name != "nt":
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return sha256_hex(data)
    except Exception:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
