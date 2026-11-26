"""Deterministic file I/O and PNG header validation."""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def png_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header = handle.read(33)
    if len(header) < 33 or header[:8] != PNG_SIGNATURE:
        raise ValueError("invalid PNG signature")
    chunk_length = struct.unpack(">I", header[8:12])[0]
    chunk_type = header[12:16]
    if chunk_length != 13 or chunk_type != b"IHDR":
        raise ValueError("missing first IHDR chunk")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", header[16:29])
    )
    if compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise ValueError("unsupported PNG coding parameters")
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "compression": compression,
        "filtering": filtering,
        "interlace": interlace,
    }


def png_pixel_mode(header: dict[str, int]) -> str:
    color_type = header["color_type"]
    bit_depth = header["bit_depth"]
    if bit_depth != 8:
        return f"bit{bit_depth}"
    return {0: "L", 2: "RGB", 3: "P", 4: "LA", 6: "RGBA"}.get(
        color_type, f"color{color_type}"
    )


def ensure_writable(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".write-probe"
    try:
        probe.write_text("ok", encoding="ascii")
        probe.read_text(encoding="ascii")
    finally:
        if probe.exists():
            probe.unlink()


def read_jsonl(path: Path) -> list[dict]:
    commands: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: command must be an object")
            value["_line"] = line_number
            commands.append(value)
    return commands
