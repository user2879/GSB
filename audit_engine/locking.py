"""检查点并发写的文件锁与原子重命名工具。

POSIX 使用 fcntl.flock，Windows 使用 msvc.locking；
锁冲突立即返回 E_LOCK 而非无限阻塞。
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import Iterator

from .errors import AuditError, E_LOCK

try:
    import fcntl  # POSIX
except ImportError:  # pragma: no cover - Windows 分支
    fcntl = None
    import msvcrt  # type: ignore


class LockHandle:
    def __init__(self, path: str):
        self.path = path
        self.handle = open(path, "a+b")
        self.acquired = False

    def acquire(self) -> "LockHandle":
        if fcntl is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.handle.close()
                raise AuditError(f"检查点锁被占用: {self.path}", E_LOCK) from exc
        else:  # pragma: no cover - Windows 分支
            try:
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                self.handle.close()
                raise AuditError(f"检查点锁被占用: {self.path}", E_LOCK) from exc
        self.acquired = True
        return self

    def release(self) -> None:
        if not self.acquired:
            return
        if fcntl is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        else:  # pragma: no cover - Windows 分支
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.acquired = False
        self.handle.close()


@contextmanager
def checkpoint_lock(lock_path: str) -> Iterator[None]:
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    handle = LockHandle(lock_path).acquire()
    try:
        yield
    finally:
        handle.release()


def atomic_write(path: str, data: str) -> None:
    """先写同目录临时文件，再 os.replace 整体替换（NTFS/POSIX 均原子）。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
