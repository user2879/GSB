"""Canonical serialization, coordinate ordering and content hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

Coord = tuple[int, int]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON encoding used for hashes and files."""

    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_value(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_major_key(coord: Coord) -> tuple[int, int]:
    x, y = coord
    return (y, x)


def sorted_coords(coords: Iterable[Coord]) -> list[Coord]:
    return sorted(set(coords), key=row_major_key)


def coord_to_list(coord: Coord) -> list[int]:
    return [coord[0], coord[1]]


def list_to_coord(value: Any) -> Coord:
    return (int(value[0]), int(value[1]))


def grid_text(cells: list[list[str]]) -> str:
    """Canonical grid text: one space-separated row per line, final LF."""

    return "".join(" ".join(row) + "\n" for row in cells)


def grid_hash(cells: list[list[str]]) -> str:
    return sha256_bytes(grid_text(cells).encode("utf-8"))


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
