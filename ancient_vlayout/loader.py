"""PNG container validation, grayscale decoding and truth-file validation."""

from __future__ import annotations

import io
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import E_IMAGE_INVALID, E_TRUTH_INVALID
from .util import read_json, sha256_bytes, sha256_json


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SUPPORTED_MODES = {"L", "RGB", "RGBA"}


@dataclass
class LoadedImage:
    name: str
    path: Path
    array: np.ndarray
    source_mode: str
    raw_sha256: str
    content_fingerprint: str
    truth: dict[str, Any]
    truth_sha256: str


@dataclass
class Dataset:
    images: dict[str, LoadedImage] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def image_names(self) -> list[str]:
        return sorted(self.images)

    def image_set_hash(self) -> str:
        return sha256_json(self.entries)


def _png_chunks(raw: bytes) -> list[tuple[str, bytes]]:
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError("missing PNG signature")
    chunks: list[tuple[str, bytes]] = []
    position = len(PNG_SIGNATURE)
    saw_iend = False
    while position < len(raw):
        if position + 12 > len(raw):
            raise ValueError("truncated PNG chunk header")
        length = struct.unpack(">I", raw[position : position + 4])[0]
        chunk_type = raw[position + 4 : position + 8]
        data_start = position + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(raw):
            raise ValueError("truncated PNG chunk payload")
        data = raw[data_start:data_end]
        expected_crc = struct.unpack(">I", raw[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        try:
            name = chunk_type.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("invalid PNG chunk type") from exc
        chunks.append((name, data))
        position = crc_end
        if name == "IEND":
            saw_iend = True
            break
    if not chunks or chunks[0][0] != "IHDR":
        raise ValueError("PNG must begin with IHDR")
    if not saw_iend:
        raise ValueError("PNG has no IEND chunk")
    if any(chunk == "" for chunk, _ in chunks):
        raise ValueError("invalid empty PNG chunk")
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        raise ValueError("invalid IHDR length")
    width, height = struct.unpack(">II", ihdr[:8])
    if width <= 0 or height <= 0:
        raise ValueError("PNG dimensions must be positive")
    return chunks


def _load_grayscale(name: str, raw: bytes, config: dict[str, Any]) -> tuple[np.ndarray, str]:
    chunks = _png_chunks(raw)
    ihdr = chunks[0][1]
    width, height = struct.unpack(">II", ihdr[:8])
    if (width, height) != (int(config["expected_width"]), int(config["expected_height"])):
        raise ValueError(
            f"unexpected dimensions {width}x{height}; expected "
            f"{config['expected_width']}x{config['expected_height']}"
        )
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ValueError("decoder did not identify PNG")
            if image.size != (width, height):
                raise ValueError("decoder dimensions disagree with IHDR")
            if image.mode not in _SUPPORTED_MODES:
                raise ValueError(f"unsupported pixel mode {image.mode!r}")
            source_mode = image.mode
            image.load()
            gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"PNG decode failed: {exc}") from exc
    if gray.shape != (height, width):
        raise ValueError("decoded array shape disagrees with PNG dimensions")
    return gray, source_mode


def _find_truth_path(image_path: Path) -> Path:
    candidates = [
        image_path.with_name(f"{image_path.stem}.truth.json"),
        image_path.with_name(f"{image_path.stem}.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("truth JSON not found")


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _validate_bbox(
    bbox: Any,
    width: int,
    height: int,
    label: str,
) -> tuple[int, int, int, int]:
    if not _is_bbox(bbox):
        raise ValueError(f"{label} bbox must be four integers")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{label} bbox outside image or not positive")
    return x0, y0, x1, y1


def validate_truth(
    truth: Any,
    image_name: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    if not isinstance(truth, dict):
        raise ValueError("truth root must be an object")
    if "image" in truth and truth["image"] != image_name:
        raise ValueError("truth image name does not match PNG")
    for field in ("width", "height"):
        if field in truth and (
            not isinstance(truth[field], int)
            or isinstance(truth[field], bool)
            or truth[field] != (image_width if field == "width" else image_height)
        ):
            raise ValueError(f"truth {field} does not match decoded image")
    columns = truth.get("columns")
    if not isinstance(columns, list):
        raise ValueError("truth columns must be a list")
    seen_columns: set[int] = set()
    for column_position, column in enumerate(columns):
        if not isinstance(column, dict):
            raise ValueError(f"column {column_position} must be an object")
        index = column.get("index", column.get("column_index", column_position))
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"column {column_position} index must be a non-negative integer")
        if index in seen_columns:
            raise ValueError(f"duplicate column index {index}")
        seen_columns.add(index)
        column_box = _validate_bbox(column.get("bbox"), image_width, image_height, f"column {index}")
        chars = column.get("chars")
        if not isinstance(chars, list):
            raise ValueError(f"column {index} chars must be a list")
        reading_indices: set[int] = set()
        for char_position, char in enumerate(chars):
            if not isinstance(char, dict):
                raise ValueError(f"column {index} char {char_position} must be an object")
            reading_index = char.get("reading_index", char.get("index", char_position))
            if not isinstance(reading_index, int) or isinstance(reading_index, bool) or reading_index < 0:
                raise ValueError(f"column {index} char {char_position} reading index is invalid")
            if reading_index in reading_indices:
                raise ValueError(f"column {index} duplicate reading index {reading_index}")
            reading_indices.add(reading_index)
            char_box = _validate_bbox(
                char.get("bbox"),
                image_width,
                image_height,
                f"column {index} char {reading_index}",
            )
            if not (
                column_box[0] <= char_box[0] < char_box[2] <= column_box[2]
                and column_box[1] <= char_box[1] < char_box[3] <= column_box[3]
            ):
                raise ValueError(f"column {index} char {reading_index} bbox outside column bbox")
            char_text = char.get("char", char.get("text"))
            if not isinstance(char_text, str) or not char_text.strip():
                raise ValueError(f"column {index} char {reading_index} content must be non-empty")
        if reading_indices != set(range(len(reading_indices))):
            raise ValueError(f"column {index} reading indices must be contiguous from zero")
    if seen_columns != set(range(len(seen_columns))):
        raise ValueError("column indices must be contiguous from zero")
    return truth


def _failure(file_name: str, code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "file": file_name,
        "code": code,
        "message": message,
        "details": details or {},
    }


def load_dataset(
    samples_dir: Path,
    config: dict[str, Any],
    *,
    raw_overrides: Mapping[str, bytes] | None = None,
    truth_mutators: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    external_errors: Mapping[str, dict[str, Any]] | None = None,
) -> Dataset:
    raw_overrides = raw_overrides or {}
    truth_mutators = truth_mutators or {}
    external_errors = external_errors or {}
    dataset = Dataset()
    png_paths = sorted(
        (path for path in samples_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png"),
        key=lambda path: path.name,
    )
    for path in png_paths:
        name = path.name
        entry: dict[str, Any] = {"file": name}
        raw = raw_overrides.get(name)
        if raw is None:
            try:
                raw = path.read_bytes()
            except OSError as exc:
                failure = _failure(name, E_IMAGE_INVALID, f"cannot read PNG: {exc}")
                dataset.failures.append(failure)
                entry.update({"image_sha256": None, "truth_sha256": None, "code": E_IMAGE_INVALID})
                dataset.entries.append(entry)
                continue
        raw_hash = sha256_bytes(raw)
        entry["image_sha256"] = raw_hash

        if name in external_errors:
            err = external_errors[name]
            dataset.failures.append(_failure(name, err["code"], err["message"], err.get("details")))
            entry.update({"truth_sha256": None, "code": err["code"]})
            dataset.entries.append(entry)
            continue

        try:
            gray, source_mode = _load_grayscale(name, raw, config)
        except ValueError as exc:
            dataset.failures.append(_failure(name, E_IMAGE_INVALID, str(exc)))
            entry.update({"truth_sha256": None, "code": E_IMAGE_INVALID})
            dataset.entries.append(entry)
            continue

        try:
            truth_path = _find_truth_path(path)
            truth = read_json(truth_path)
            mutator = truth_mutators.get(name)
            if mutator is not None:
                truth = mutator(truth)
            truth = validate_truth(truth, name, gray.shape[1], gray.shape[0])
            truth_hash = sha256_json(truth)
        except (OSError, ValueError, TypeError) as exc:
            dataset.failures.append(_failure(name, E_TRUTH_INVALID, str(exc)))
            entry.update({"truth_sha256": None, "code": E_TRUTH_INVALID})
            dataset.entries.append(entry)
            continue

        content_fingerprint = sha256_bytes(
            raw_hash.encode("ascii") + f":{gray.shape[1]}x{gray.shape[0]}:{source_mode}".encode("ascii")
        )
        loaded = LoadedImage(
            name=name,
            path=path,
            array=gray,
            source_mode=source_mode,
            raw_sha256=raw_hash,
            content_fingerprint=content_fingerprint,
            truth=truth,
            truth_sha256=truth_hash,
        )
        dataset.images[name] = loaded
        entry.update(
            {
                "truth_sha256": truth_hash,
                "content_fingerprint": content_fingerprint,
                "source_mode": source_mode,
            }
        )
        dataset.entries.append(entry)

    dataset.failures.sort(key=lambda item: item["file"])
    dataset.entries.sort(key=lambda item: item["file"])
    return dataset
