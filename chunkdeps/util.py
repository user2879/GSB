"""Small deterministic primitives used throughout the workbench."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import E_CHUNK, E_INPUT, BusinessError


SUPPORTED_DIGESTS = frozenset({"sha256", "sha384", "sha512"})


def canonical_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_text(value).encode("utf-8")


def digest_bytes(data: bytes, algorithm: str = "sha256") -> str:
    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise BusinessError(
            E_INPUT,
            "unsupported digest algorithm",
            {"algorithm": algorithm, "supported": sorted(SUPPORTED_DIGESTS)},
        ) from exc
    hasher.update(data)
    return hasher.hexdigest()


def digest_text(text: str, algorithm: str = "sha256") -> str:
    return digest_bytes(text.encode("utf-8"), algorithm)


def digest_json(value: Any, algorithm: str = "sha256") -> str:
    return digest_bytes(canonical_bytes(value), algorithm)


def digest_file(path: str | os.PathLike[str], algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def wall_time_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(value: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001 - input conversion boundary
        raise BusinessError(E_CHUNK, "chunk payload is not valid base64") from exc


def validate_chunk_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BusinessError(
            E_CHUNK,
            "chunk size must be a positive integer",
            {"chunk_size": value},
        )
    return value


def validate_total_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BusinessError(
            E_CHUNK,
            "empty or invalid source file",
            {"total_size": value},
        )
    return value


def chunk_count(total_size: int, chunk_size: int) -> int:
    return (total_size + chunk_size - 1) // chunk_size


def expected_chunk_size(total_size: int, chunk_size: int, index: int, count: int) -> int:
    if index < 0 or index >= count:
        raise BusinessError(
            E_CHUNK,
            "chunk index is outside the session range",
            {"index": index, "chunk_count": count},
        )
    if index == count - 1:
        return total_size - (count - 1) * chunk_size
    return chunk_size


def iter_file_chunks(
    path: str | os.PathLike[str],
    chunk_size: int,
    algorithm: str = "sha256",
) -> Iterator[tuple[int, bytes, str]]:
    """Yield ``(index, bytes, digest)`` without changing source bytes."""

    validate_chunk_size(chunk_size)
    file_path = Path(path)
    if not file_path.is_file():
        raise BusinessError(E_CHUNK, "source file does not exist", {"path": str(path)})
    total = file_path.stat().st_size
    validate_total_size(total)
    with file_path.open("rb") as stream:
        index = 0
        while True:
            payload = stream.read(chunk_size)
            if not payload:
                break
            yield index, payload, digest_bytes(payload, algorithm)
            index += 1


def read_file_chunk(
    path: str | os.PathLike[str],
    index: int,
    chunk_size: int,
    total_size: int,
    algorithm: str = "sha256",
) -> tuple[bytes, str]:
    validate_chunk_size(chunk_size)
    validate_total_size(total_size)
    count = chunk_count(total_size, chunk_size)
    expected = expected_chunk_size(total_size, chunk_size, index, count)
    file_path = Path(path)
    with file_path.open("rb") as stream:
        stream.seek(index * chunk_size)
        payload = stream.read(expected)
    if len(payload) != expected:
        raise BusinessError(
            E_CHUNK,
            "source file was truncated while reading a chunk",
            {"index": index, "expected_size": expected, "actual_size": len(payload)},
        )
    return payload, digest_bytes(payload, algorithm)


def bitmap_from_indices(indices: Iterable[int], count: int) -> str:
    chars = ["0"] * count
    for index in sorted(set(indices)):
        if 0 <= index < count:
            chars[index] = "1"
    return "".join(chars)


def indices_from_bitmap(bitmap: str) -> list[int]:
    return [index for index, value in enumerate(bitmap) if value == "1"]


def missing_indices(bitmap: str) -> list[int]:
    return [index for index, value in enumerate(bitmap) if value == "0"]


def validate_object_key(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise BusinessError(E_INPUT, "object key must be a non-empty string")
    if len(value) > 512 or "\x00" in value:
        raise BusinessError(E_INPUT, "object key is too long or contains NUL")
    if value.startswith("/") or value.endswith("/"):
        raise BusinessError(E_INPUT, "object key must not start or end with '/'")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BusinessError(E_INPUT, "object key contains an invalid path segment")
    return value


def prefixes_for_key(key: str) -> list[str]:
    parts = key.split("/")
    prefixes = ["*"]
    for index in range(1, len(parts)):
        prefixes.append("/".join(parts[:index]))
    return prefixes


def view_prefix_matches(key: str, prefix: str) -> bool:
    return prefix == "*" or key == prefix or key.startswith(prefix + "/")


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        _fsync_directory(target.parent)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    atomic_write_text(path, canonical_text(value) + "\n")


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


def file_set_digest(paths: Iterable[str], algorithm: str = "sha256") -> str:
    entries = sorted(str(path) for path in paths)
    return digest_json(entries, algorithm)


def ensure_relative_posix(path: str | os.PathLike[str], root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise BusinessError(E_INPUT, "path escapes the workbench root") from exc


def resolve_relative(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise BusinessError(E_INPUT, "stored path escapes the workbench root") from exc
    return candidate


def tags_canonical(tags: Any) -> tuple[dict[str, str], str]:
    if tags is None:
        tags = {}
    if not isinstance(tags, dict):
        raise BusinessError(E_INPUT, "tags must be a JSON object")
    normalized: dict[str, str] = {}
    for key, value in tags.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise BusinessError(E_INPUT, "tag keys and values must be strings")
        normalized[key] = value
    return normalized, canonical_text(normalized)


def stable_public(value: Any) -> Any:
    """Round-trip through JSON to make protocol values plain and stable."""

    return json.loads(canonical_text(value))

