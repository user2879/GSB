"""Cross-process workspace lock with a non-blocking E_LOCK contract."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import IO

from .errors import LockConflictError


_PROCESS_GUARD = threading.Lock()
_PROCESS_LOCKS: set[str] = set()


class WorkspaceLock:
    """Serialize writers to one workspace using an advisory file lock."""

    def __init__(self, path_like: str | Path) -> None:
        self.path = Path(path_like).resolve()
        self._key = os.path.normcase(str(self.path))
        self._fd: int | None = None
        self._process_locked = False

    def acquire(self) -> "WorkspaceLock":
        with _PROCESS_GUARD:
            if self._key in _PROCESS_LOCKS:
                raise LockConflictError(
                    f"workspace lock is already held in this process: {self.path}"
                )
            _PROCESS_LOCKS.add(self._key)
            self._process_locked = True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(
                str(self.path),
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            self._lock_os_file()
        except Exception:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._release_process_lock()
            raise

        hold_seconds = os.environ.get("AUDIT_FORENSICS_TEST_HOLD_LOCK_SECONDS")
        if hold_seconds:
            ready = os.environ.get("AUDIT_FORENSICS_TEST_LOCK_READY_FILE")
            if ready:
                Path(ready).write_text("locked\n", encoding="utf-8")
            time.sleep(max(0.0, float(hold_seconds)))
        return self

    def _lock_os_file(self) -> None:
        assert self._fd is not None
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(self._fd).st_size == 0:
                    os.write(self._fd, b"0")
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise LockConflictError(
                f"workspace lock conflict for {self.path}: {exc}"
            ) from exc

    def release(self) -> None:
        try:
            if self._fd is not None:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(self._fd, 0, os.SEEK_SET)
                    try:
                        msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                    except OSError:
                        pass
                else:
                    import fcntl

                    try:
                        fcntl.flock(self._fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(self._fd)
                self._fd = None
        finally:
            self._release_process_lock()

    def _release_process_lock(self) -> None:
        if self._process_locked:
            with _PROCESS_GUARD:
                _PROCESS_LOCKS.discard(self._key)
            self._process_locked = False

    def __enter__(self) -> "WorkspaceLock":
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
