"""Raw block-device image handling with strict input validation.

A volume is a plain binary image of ``block_count * block_size`` bytes.
Metadata lives in SQLite rather than in a file header so the image stays a
raw device mirror that can be inspected with ordinary binary tools.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable

from .errors import DomainError, E_INPUT, E_SHAPE

SUPPORTED_BLOCK_SIZES = (512, 1024, 2048, 4096, 8192)
DIGEST_ALGO = "blake2b"


def block_digest(data: bytes) -> str:
    """Stable per-block content digest (hex, 32-byte BLAKE2b)."""
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def deterministic_block(seed: str, block_no: int, block_size: int, pattern: int = 0) -> bytes:
    """Reproducible block content derived from seed, block number and pattern.

    A BLAKE2b counter-mode keystream makes the content fully deterministic
    without any third-party dependency.  ``pattern`` rotates every byte so an
    injected change is visibly different from the baseline content.
    """
    out = bytearray()
    counter = 0
    base = f"{seed}|{block_no}|".encode()
    while len(out) < block_size:
        chunk = hashlib.blake2b(
            base + counter.to_bytes(4, "big"), digest_size=64, person=b"volblk"
        ).digest()
        out.extend(chunk)
        counter += 1
    data = bytes(byte ^ (pattern & 0xFF) for byte in out[:block_size])
    return data


def parse_change_spec(spec: str) -> list[tuple[int, int]]:
    """Parse a hex change spec such as ``"0a:ff,1c:03"``.

    Each item is ``<hex block no>:<hex pattern byte>``.  Empty items, bad
    hex, missing separators or duplicates are rejected as E_INPUT before any
    byte of the volume is touched.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise DomainError(E_INPUT, "change spec must be a non-empty hex string", field="change_spec")
    parsed: list[tuple[int, int]] = []
    seen: set[int] = set()
    for index, item in enumerate(spec.split(",")):
        token = item.strip()
        if not token or token.count(":") != 1:
            raise DomainError(
                E_INPUT,
                f"change item {index!r} must look like <hex block>:<hex byte>",
                field=f"change_spec[{index}]",
            )
        left, right = token.split(":")
        left, right = left.strip(), right.strip()
        try:
            block_no = int(left, 16)
        except ValueError:
            raise DomainError(
                E_INPUT, f"block number {left!r} is not hex", field=f"change_spec[{index}].block"
            )
        try:
            pattern = int(right, 16)
        except ValueError:
            raise DomainError(
                E_INPUT, f"pattern byte {right!r} is not hex", field=f"change_spec[{index}].pattern"
            )
        if not 0 <= pattern <= 0xFF:
            raise DomainError(
                E_INPUT, f"pattern {right!r} exceeds a byte", field=f"change_spec[{index}].pattern"
            )
        if block_no in seen:
            raise DomainError(
                E_INPUT, f"duplicate block {left!r} in change spec", field=f"change_spec"
            )
        seen.add(block_no)
        parsed.append((block_no, pattern))
    return parsed


class Volume:
    """Fixed-block binary image supporting block IO and shape validation."""

    def __init__(self, path: str, block_size: int, block_count: int, seed: str = "vol") -> None:
        if not isinstance(block_size, int) or block_size not in SUPPORTED_BLOCK_SIZES:
            raise DomainError(
                E_INPUT,
                f"block_size {block_size!r} not in supported set {SUPPORTED_BLOCK_SIZES}",
                field="block_size",
            )
        if not isinstance(block_count, int) or block_count <= 0:
            raise DomainError(
                E_INPUT, "block_count must be a positive integer", field="block_count"
            )
        self.path = os.path.abspath(path)
        self.block_size = block_size
        self.block_count = block_count
        self.seed = seed

    @property
    def expected_length(self) -> int:
        return self.block_size * self.block_count

    def ensure_parent_dir(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def create_initialized(self) -> None:
        """Write a deterministic baseline image, replacing any existing file."""
        self.ensure_parent_dir()
        with open(self.path, "wb") as handle:
            for block_no in range(self.block_count):
                handle.write(deterministic_block(self.seed, block_no, self.block_size))
            handle.flush()
            os.fsync(handle.fileno())

    def validate_shape(self) -> None:
        if not os.path.exists(self.path):
            raise DomainError(E_SHAPE, f"volume image missing: {self.path}", field="path")
        actual = os.path.getsize(self.path)
        if actual != self.expected_length:
            raise DomainError(
                E_SHAPE,
                f"image length {actual} != block_count {self.block_count} * block_size "
                f"{self.block_size} ({self.expected_length})",
                field="image_length",
            )

    def _check_block_no(self, block_no: int) -> None:
        if not isinstance(block_no, int) or not 0 <= block_no < self.block_count:
            raise DomainError(
                E_INPUT,
                f"block_no {block_no!r} out of range [0, {self.block_count})",
                field="block_no",
            )

    def read_block(self, block_no: int) -> bytes:
        self.validate_shape()
        self._check_block_no(block_no)
        with open(self.path, "rb") as handle:
            handle.seek(block_no * self.block_size)
            data = handle.read(self.block_size)
        if len(data) != self.block_size:
            raise DomainError(E_SHAPE, "short read on volume image", field="block_data", block_no=block_no)
        return data

    def write_block(self, block_no: int, data: bytes) -> None:
        self._check_block_no(block_no)
        if not isinstance(data, (bytes, bytearray)) or len(data) != self.block_size:
            raise DomainError(
                E_INPUT,
                f"block data must be exactly {self.block_size} bytes",
                field="block_data",
                block_no=block_no,
            )
        with open(self.path, "r+b") as handle:
            handle.seek(block_no * self.block_size)
            handle.write(bytes(data))
            handle.flush()
            os.fsync(handle.fileno())

    def iter_blocks(self) -> Iterable[tuple[int, bytes]]:
        self.validate_shape()
        with open(self.path, "rb") as handle:
            for block_no in range(self.block_count):
                data = handle.read(self.block_size)
                if len(data) != self.block_size:
                    raise DomainError(E_SHAPE, "short read while iterating", field="block_data", block_no=block_no)
                yield block_no, data

    def whole_digest(self) -> str:
        hasher = hashlib.blake2b(digest_size=32)
        for _, data in self.iter_blocks():
            hasher.update(data)
        return hasher.hexdigest()

    def block_digests(self) -> dict[int, str]:
        return {block_no: block_digest(data) for block_no, data in self.iter_blocks()}

    def inject_changes(self, spec: str) -> list[int]:
        """Validate then apply a hex change spec; returns changed block numbers.

        All parsing and range checks happen first so a rejected spec never
        performs a partial write.
        """
        parsed = parse_change_spec(spec)
        for block_no, _pattern in parsed:
            self._check_block_no(block_no)
        changed: list[int] = []
        for block_no, pattern in parsed:
            data = deterministic_block(self.seed, block_no, self.block_size, pattern)
            self.write_block(block_no, data)
            changed.append(block_no)
        return changed

    def snapshot_image(self, target_path: str) -> None:
        """Byte-copy the image into a temp/staging area."""
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(self.path, "rb") as src, open(target_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
