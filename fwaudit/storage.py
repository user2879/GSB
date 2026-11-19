"""原子快照整体替换、文件锁串行化与检查点演进。"""

import contextlib
import json
import os
import tempfile

from .errors import AuditError, E_LOCK
from .patch import BudgetConfig

CHECKPOINT_NAME = "checkpoint.json"
CHECKPOINT_STRUCT_VERSION = 1
LOCK_NAME = ".fwaudit.lock"


def atomic_write(path, payload):
    """先写同目录临时文件再整体重命名替换。"""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-%s-" % os.path.basename(path), suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            if isinstance(payload, str):
                handle.write(payload.encode("utf-8"))
            else:
                handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def atomic_write_json(path, obj):
    atomic_write(
        path,
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2),
    )


@contextlib.contextmanager
def file_lock(workdir, lock_name=LOCK_NAME, blocking=False):
    """同一工作目录并发写入串行化；非阻塞冲突返回 E_LOCK。"""
    os.makedirs(workdir, exist_ok=True)
    lock_path = os.path.join(workdir, lock_name)
    handle = open(lock_path, "a+b")
    try:
        try:
            import msvcrt  # Windows
        except ImportError:
            msvcrt = None
        if msvcrt is not None:
            try:
                msvcrt.locking(
                    handle.fileno(),
                    msvcrt.LK_NBLCK if not blocking else msvcrt.LK_LOCK,
                    1,
                )
            except OSError as exc:
                raise AuditError(
                    E_LOCK, "工作目录 %s 已被其他审计进程锁定" % workdir,
                ) from exc
        else:
            import fcntl
            flags = fcntl.LOCK_EX
            if not blocking:
                flags |= fcntl.LOCK_NB
            try:
                fcntl.flock(handle.fileno(), flags)
            except BlockingIOError as exc:
                raise AuditError(
                    E_LOCK, "工作目录 %s 已被其他审计进程锁定" % workdir,
                ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(b"%d\n" % os.getpid())
        handle.flush()
        yield lock_path
    finally:
        with contextlib.suppress(OSError):
            if msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def checkpoint_path(workdir):
    return os.path.join(workdir, CHECKPOINT_NAME)


def save_checkpoint(workdir, state, budget):
    payload = {
        "struct_version": CHECKPOINT_STRUCT_VERSION,
        "policy_version": budget.policy_version,
        "generation_watermark": state["generation_watermark"],
        "images": state["images"],
        "patch_records": state["patch_records"],
        "budget": budget.as_dict(),
    }
    atomic_write_json(checkpoint_path(workdir), payload)


def load_checkpoint(workdir, budget):
    """恢复检查点；结构/策略版本变化或损坏时整体重建（返回 None）。"""
    path = checkpoint_path(workdir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            payload = json.loads(handle.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("struct_version") != CHECKPOINT_STRUCT_VERSION:
        return None
    if payload.get("policy_version") != budget.policy_version:
        return None
    required = ("generation_watermark", "images", "patch_records")
    if not all(key in payload for key in required):
        return None
    return payload


def reset_checkpoint(workdir):
    with contextlib.suppress(OSError):
        os.unlink(checkpoint_path(workdir))


def temp_cleanup(workdir):
    """清理崩溃残留的临时文件。"""
    if not os.path.isdir(workdir):
        return
    for name in os.listdir(workdir):
        if name.startswith(".tmp-") and name.endswith(".tmp"):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(workdir, name))
