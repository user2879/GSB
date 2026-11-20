from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .errors import fail


def canonical_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        fail("E_INTERNAL", f"value is not canonically serializable: {exc}")


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (canonical_json_text(value) + suffix).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        fail("E_INTERNAL", f"value is not JSON serializable: {exc}")
    return (text + "\n").encode("utf-8")


def make_writable(path: Path) -> None:
    if not path.exists():
        return
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode | stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
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


def atomic_write_bytes(path: Path, data: bytes, *, readonly: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        make_writable(path)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
        if readonly:
            try:
                os.chmod(path, stat.S_IREAD)
            except OSError:
                pass
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def copy_atomic_readonly(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    try:
        data = source.read_bytes()
    except OSError as exc:
        fail("E_IO", f"cannot read {source}: {exc}", manifest_path=str(source))
    atomic_write_bytes(destination, data, readonly=True)


def read_bytes(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        fail("E_IO", f"cannot read {path}: {exc}", manifest_path=str(path))


def sha256_hex(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value, newline=False))


def load_json(path: Path, *, code: str = "E_FORMAT") -> Any:
    data = read_bytes(path)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(code, f"JSON is not UTF-8: {exc}", manifest_path=str(path), path="$")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        fail(
            code,
            f"invalid JSON: {exc.msg}",
            path="$",
            manifest_path=str(path),
            details={"line": exc.lineno, "column": exc.colno},
        )
