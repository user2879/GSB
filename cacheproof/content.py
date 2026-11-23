"""Deterministic entry content, entry digests and atomic file persistence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterator

from .canonical import canonical_bytes
from .errors import DiagnosticError


ENTRY_DOMAIN = b"CACHEPROOF-ENTRY-v1\0"
CONTENT_DOMAIN = b"CACHEPROOF-CONTENT-v1\0"
BLOCK_SIZE = 32
READ_SIZE = 128 * 1024


def iter_content(seed: str, size_bytes: int) -> Iterator[bytes]:
    """Yield deterministic content derived only from ``seed`` and size.

    Each 32-byte block is SHA-256(CONTENT_DOMAIN || seed_length_be64 ||
    seed_utf8 || counter_be64). The final block is truncated to the declared
    byte count. This definition is independent of Python's random module and
    host locale.
    """

    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")
    seed_bytes = seed.encode("utf-8")
    remaining = size_bytes
    counter = 0
    prefix = CONTENT_DOMAIN + len(seed_bytes).to_bytes(8, "big") + seed_bytes
    while remaining:
        block = hashlib.sha256(prefix + counter.to_bytes(8, "big")).digest()
        take = min(remaining, len(block))
        yield block[:take]
        remaining -= take
        counter += 1


def content_bytes(seed: str, size_bytes: int) -> bytes:
    return b"".join(iter_content(seed, size_bytes))


def _entry_hash_prefix(key: str, size_bytes: int) -> bytes:
    return ENTRY_DOMAIN + canonical_bytes(
        {"key": key, "size_bytes": size_bytes}
    ) + b"\0"


def entry_digest_from_chunks(
    key: str,
    size_bytes: int,
    chunks: Iterator[bytes],
) -> str:
    digest = hashlib.sha256()
    digest.update(_entry_hash_prefix(key, size_bytes))
    observed = 0
    for chunk in chunks:
        observed += len(chunk)
        digest.update(chunk)
    if observed != size_bytes:
        raise DiagnosticError(
            "E_INTEGRITY",
            "content size does not match declared size",
            entry_key=key,
            field="size_bytes",
            details={"expected_size": size_bytes, "actual_size": observed},
        )
    return "sha256:" + digest.hexdigest()


def entry_digest(key: str, size_bytes: int, seed: str) -> str:
    return entry_digest_from_chunks(
        key, size_bytes, iter_content(seed, size_bytes)
    )


def digest_file(path: str | Path, key: str, size_bytes: int | None = None) -> str:
    file_path = Path(path)
    try:
        actual_size = file_path.stat().st_size
    except OSError as exc:
        raise DiagnosticError(
            "E_INTEGRITY",
            "entry content file is not readable",
            entry_key=key,
            field="content_file",
            details={"path": str(file_path), "error": str(exc)},
        ) from exc
    size_for_digest = actual_size if size_bytes is None else size_bytes

    digest = hashlib.sha256()
    digest.update(_entry_hash_prefix(key, size_for_digest))
    observed = 0
    try:
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(READ_SIZE)
                if not chunk:
                    break
                observed += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise DiagnosticError(
            "E_INTEGRITY",
            "entry content file is not readable",
            entry_key=key,
            field="content_file",
            details={"path": str(file_path), "error": str(exc)},
        ) from exc
    if size_bytes is not None and observed != size_bytes:
        raise DiagnosticError(
            "E_INTEGRITY",
            "entry content file size does not match the ledger",
            entry_key=key,
            field="size_bytes",
            details={
                "path": str(file_path),
                "expected_size": size_bytes,
                "actual_size": observed,
            },
        )
    if size_bytes is None and observed != size_for_digest:
        size_for_digest = observed
    return "sha256:" + digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def persist_content_file(
    destination: str | Path,
    temp_directory: str | Path,
    *,
    key: str,
    size_bytes: int,
    seed: str,
    expected_digest: str,
) -> dict:
    """Persist deterministic content with temp-write, verify and replace.

    If the destination already exists, it is accepted only when both size and
    digest are exactly correct. The function never creates a second file for
    the same entry digest.
    """

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Path(temp_directory).mkdir(parents=True, exist_ok=True)

    if destination_path.exists():
        actual = digest_file(destination_path, key, size_bytes)
        if actual != expected_digest:
            raise DiagnosticError(
                "E_INTEGRITY",
                "existing content file has a different digest",
                entry_key=key,
                field="digest",
                details={
                    "path": str(destination_path),
                    "expected_digest": expected_digest,
                    "actual_digest": actual,
                },
            )
        return {
            "path": destination_path,
            "digest": actual,
            "written": False,
            "size_bytes": destination_path.stat().st_size,
        }

    handle, temp_name = tempfile.mkstemp(
        prefix=".cacheproof-write-",
        suffix=".tmp",
        dir=temp_directory,
    )
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    digest.update(_entry_hash_prefix(key, size_bytes))
    observed = 0
    try:
        with os.fdopen(handle, "wb") as output:
            for chunk in iter_content(seed, size_bytes):
                output.write(chunk)
                digest.update(chunk)
                observed += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = "sha256:" + digest.hexdigest()
        if observed != size_bytes:
            raise DiagnosticError(
                "E_INTEGRITY",
                "temporary content size mismatch",
                entry_key=key,
                field="size_bytes",
                details={"expected_size": size_bytes, "actual_size": observed},
            )
        if actual != expected_digest:
            raise DiagnosticError(
                "E_INTEGRITY",
                "temporary content digest mismatch",
                entry_key=key,
                field="digest",
                details={
                    "expected_digest": expected_digest,
                    "actual_digest": actual,
                },
            )
        os.replace(temp_path, destination_path)
        _fsync_directory(destination_path.parent)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise

    return {
        "path": destination_path,
        "digest": actual,
        "written": True,
        "size_bytes": size_bytes,
    }
