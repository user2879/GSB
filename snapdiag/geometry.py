"""Volume geometry validation and deterministic block I/O."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .errors import DiagnosticError, E_GEOM
from .util import CHUNK_SIZE, fsync_directory


SUPPORTED_BLOCK_SIZES = frozenset({512, 1024, 2048, 4096, 8192, 16384})


@dataclass(frozen=True)
class VolumeGeometry:
    image_path: Path
    block_size: int
    block_count: int

    @property
    def expected_size(self) -> int:
        return self.block_size * self.block_count

    @property
    def image_size(self) -> int:
        return self.image_path.stat().st_size

    def offset(self, block_index: int) -> int:
        self.validate_block_index(block_index)
        return block_index * self.block_size

    def validate_block_index(self, block_index: int) -> None:
        if isinstance(block_index, bool) or not isinstance(block_index, int):
            raise DiagnosticError(
                E_GEOM,
                "block number must be an integer",
                field="block_index",
                details={"actual": block_index, "block_count": self.block_count},
            )
        if block_index < 0 or block_index >= self.block_count:
            raise DiagnosticError(
                E_GEOM,
                "block number is outside the volume",
                field="block_index",
                details={
                    "block_index": block_index,
                    "block_count": self.block_count,
                    "valid_min": 0,
                    "valid_max": self.block_count - 1,
                },
            )

    def read_block(self, block_index: int) -> bytes:
        offset = self.offset(block_index)
        with open(self.image_path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(self.block_size)
        if len(data) != self.block_size:
            raise DiagnosticError(
                E_GEOM,
                "short read from volume image",
                field="image_length",
                details={"block_index": block_index, "expected": self.block_size, "actual": len(data)},
            )
        return data

    def read_all(self) -> bytes:
        with open(self.image_path, "rb") as handle:
            return handle.read()

    def write_block(self, block_index: int, data: bytes) -> str:
        """Atomically replace the whole image after changing one fixed block."""

        offset = self.offset(block_index)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise DiagnosticError(
                E_GEOM,
                "block payload must be bytes",
                field="block_data",
                details={"block_index": block_index},
            )
        payload = bytes(data)
        if len(payload) != self.block_size:
            raise DiagnosticError(
                E_GEOM,
                "block payload length must equal block_size",
                field="block_data",
                details={"expected": self.block_size, "actual": len(payload), "block_index": block_index},
            )
        fd, temporary = tempfile.mkstemp(prefix=f".{self.image_path.name}.", dir=str(self.image_path.parent))
        try:
            with open(self.image_path, "rb") as origin, os.fdopen(fd, "w+b") as destination:
                while True:
                    chunk = origin.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    destination.write(chunk)
                destination.seek(offset)
                destination.write(payload)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.image_path)
            fsync_directory(self.image_path.parent)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return hashlib.sha256(payload).hexdigest()


def _coerce_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiagnosticError(
            E_GEOM,
            f"{field_name} must be an integer",
            field=field_name,
            details={"actual": value},
        )
    if value <= 0:
        raise DiagnosticError(
            E_GEOM,
            f"{field_name} must be greater than zero",
            field=field_name,
            details={"actual": value},
        )
    return value


def validate_geometry(
    image_path: str | os.PathLike[str],
    block_size: object,
    block_count: object,
    *,
    require_exists: bool = True,
) -> VolumeGeometry:
    """Validate fields in a deterministic order and reject the whole volume atomically."""

    size = _coerce_positive_int(block_size, "block_size")
    if size not in SUPPORTED_BLOCK_SIZES:
        raise DiagnosticError(
            E_GEOM,
            "block_size is not in the supported set",
            field="block_size",
            details={
                "actual": size,
                "supported": sorted(SUPPORTED_BLOCK_SIZES),
            },
        )
    count = _coerce_positive_int(block_count, "block_count")
    image = Path(image_path)
    if require_exists:
        try:
            actual_size = image.stat().st_size
        except FileNotFoundError:
            raise DiagnosticError(
                E_GEOM,
                "volume image does not exist",
                field="image_path",
                details={"path": str(image)},
            )
    else:
        actual_size = image.stat().st_size if image.exists() else 0
    expected = size * count
    if actual_size != expected:
        raise DiagnosticError(
            E_GEOM,
            "volume image length must equal block_size * block_count",
            field="image_length",
            details={"expected": expected, "actual": actual_size, "block_size": size, "block_count": count},
        )
    return VolumeGeometry(image_path=image, block_size=size, block_count=count)


def decode_hex_block(hex_value: str, block_size: int) -> bytes:
    if not isinstance(hex_value, str):
        raise DiagnosticError(
            E_GEOM,
            "hex block payload must be a string",
            field="hex_data",
            details={"actual_type": type(hex_value).__name__},
        )
    text = hex_value.strip()
    if text.lower().startswith("0x"):
        text = text[2:]
    if len(text) % 2:
        raise DiagnosticError(
            E_GEOM,
            "hex block payload must have an even number of characters",
            field="hex_data",
            details={"characters": len(text)},
        )
    try:
        data = bytes.fromhex(text)
    except ValueError as exc:
        raise DiagnosticError(
            E_GEOM,
            "hex block payload contains non-hexadecimal characters",
            field="hex_data",
            details={"reason": str(exc)},
        )
    if len(data) != block_size:
        raise DiagnosticError(
            E_GEOM,
            "hex block payload must exactly fill one block",
            field="hex_data",
            details={"expected_bytes": block_size, "actual_bytes": len(data)},
        )
    return data


def deterministic_block(seed: int, block_index: int, block_size: int) -> bytes:
    material = f"snapdiag:{seed}:{block_index}".encode("ascii")
    result = bytearray()
    counter = 0
    while len(result) < block_size:
        result.extend(hashlib.sha256(material + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(result[:block_size])


def generate_volume(
    image_path: str | os.PathLike[str],
    block_size: int,
    block_count: int,
    *,
    seed: int = 20261123,
) -> VolumeGeometry:
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if block_size not in SUPPORTED_BLOCK_SIZES or block_size <= 0:
        raise DiagnosticError(
            E_GEOM,
            "block_size is not in the supported set",
            field="block_size",
            details={"actual": block_size, "supported": sorted(SUPPORTED_BLOCK_SIZES)},
        )
    if isinstance(block_count, bool) or not isinstance(block_count, int) or block_count <= 0:
        raise DiagnosticError(
            E_GEOM,
            "block_count must be a positive integer",
            field="block_count",
            details={"actual": block_count},
        )
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            for index in range(block_count):
                handle.write(deterministic_block(seed, index, block_size))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return validate_geometry(path, block_size, block_count)


def hex_preview(data: bytes, limit: int = 32) -> str:
    text = data[:limit].hex()
    return text + ("..." if len(data) > limit else "")

