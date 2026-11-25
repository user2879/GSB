"""Dataset scanning and structural validation."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from PIL import Image

from .constants import FIXED_PIXEL_MODE, FIXED_RESOLUTION, PNG_SIGNATURE
from .models import (
    ColumnTruth,
    Dataset,
    FailureLog,
    ImageEntry,
    ImageTruth,
    KeypointVersion,
    TruthKeypoint,
)
from .utils import file_hash, read_json, require_int, stable_hash


def load_dataset(root: str | Path, failures: FailureLog | None = None) -> Dataset:
    """Scan a dataset directory and append every structural validation failure."""

    if failures is None:
        failures = FailureLog()
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("root is not an object")
    except Exception as exc:
        failures.add("E_MANIFEST", "dataset", "manifest.json", f"cannot load manifest: {exc}")
        manifest = {}

    resolution_raw = manifest.get("resolution", {})
    if isinstance(resolution_raw, dict):
        resolution = (
            resolution_raw.get("width", FIXED_RESOLUTION[0]),
            resolution_raw.get("height", FIXED_RESOLUTION[1]),
        )
    else:
        resolution = FIXED_RESOLUTION
    pixel_mode = manifest.get("pixel_mode", FIXED_PIXEL_MODE)

    scanned = tuple(sorted(path.relative_to(root).as_posix() for path in root.rglob("*.png")))
    image_entries: list[ImageEntry] = []
    truths: dict[str, ImageTruth] = {}
    versions: dict[str, dict[str, KeypointVersion]] = {}
    listed_paths: set[str] = set()
    manifest_images = manifest.get("images", [])
    if not isinstance(manifest_images, list):
        failures.add("E_MANIFEST", "dataset", "manifest.images", "images must be an array")
        manifest_images = []

    for index, raw_entry in enumerate(manifest_images):
        if not isinstance(raw_entry, dict):
            failures.add("E_MANIFEST", "dataset", f"images[{index}]", "image entry must be an object")
            continue
        name = raw_entry.get("name")
        if not isinstance(name, str) or not name:
            failures.add("E_MANIFEST", "dataset", f"images[{index}].name", "image name must be a non-empty string")
            continue
        if any(entry.name == name for entry in image_entries):
            failures.add("E_MANIFEST", "dataset", name, "duplicate image name")
            continue
        relative_image = raw_entry.get("path", f"images/{name}")
        if not isinstance(relative_image, str):
            failures.add("E_MANIFEST", "dataset", name, "image path must be a string")
            relative_image = f"images/{name}"
        listed_paths.add(relative_image)
        image_path = (root / relative_image).resolve()
        if not image_path.is_file():
            failures.add("E_IMAGE_MISSING", "image", name, "listed image is missing", path=relative_image)

        _validate_png(image_path, name, failures)

        truth_data: Any = raw_entry.get("truth")
        truth_path_raw = raw_entry.get("truth_path", f"truth/{Path(name).stem}.json")
        truth_path = (root / truth_path_raw).resolve() if isinstance(truth_path_raw, str) else None
        if truth_data is None and truth_path is not None:
            try:
                truth_data = read_json(truth_path)
            except Exception as exc:
                failures.add("E_TRUTH", "image", name, f"cannot load truth: {exc}", path=str(truth_path_raw))
                truth_data = {}
        truth = _parse_truth(name, truth_data, failures)
        if truth is not None:
            truths[name] = truth

        version_map: dict[str, str] = {}
        raw_versions = raw_entry.get("keypoint_versions", {})
        if isinstance(raw_versions, dict):
            for version, relative in sorted(raw_versions.items()):
                if not isinstance(version, str) or not isinstance(relative, str):
                    failures.add("E_KEYPOINT_VERSION", "image", name, "keypoint version mapping must be string to string")
                    continue
                version_map[version] = relative
        else:
            failures.add("E_KEYPOINT_VERSION", "image", name, "keypoint_versions must be an object")

        parsed_versions: dict[str, KeypointVersion] = {}
        for version, relative in sorted(version_map.items()):
            version_path = (root / relative).resolve()
            try:
                raw_version = read_json(version_path)
            except Exception as exc:
                failures.add(
                    "E_KEYPOINT_VERSION",
                    "image",
                    f"{name}:{version}",
                    f"cannot load keypoint version: {exc}",
                    path=relative,
                )
                continue
            points = _parse_version_points(name, version, raw_version, truth, failures)
            if points is not None:
                parsed_versions[version] = KeypointVersion(
                    version=version,
                    image=name,
                    keypoints=points,
                    path=relative,
                )
        versions[name] = parsed_versions
        image_entries.append(
            ImageEntry(
                name=name,
                image_path=relative_image,
                truth_path=truth_path_raw if isinstance(truth_path_raw, str) else "",
                truth=truth,
                keypoint_versions=version_map,
                image_hash=file_hash(image_path) if image_path.is_file() else "",
            )
        )

    for relative in scanned:
        if relative not in listed_paths:
            failures.add("E_IMAGE_UNLISTED", "dataset", relative, "PNG is not listed in manifest.images")

    image_entries.sort(key=lambda entry: entry.name)
    content_payload = {
        "manifest": manifest,
        "images": [
            {
                "name": entry.name,
                "hash": entry.image_hash,
                "truth": _truth_payload(truths.get(entry.name)) if entry.name in truths else None,
                "versions": {
                    key: [point.as_dict() for point in value.keypoints]
                    for key, value in versions.get(entry.name, {}).items()
                },
            }
            for entry in image_entries
        ],
        "scanned_pngs": scanned,
    }
    content_hash = stable_hash(content_payload)
    return Dataset(
        root=str(root),
        manifest_path=str(manifest_path),
        manifest=manifest,
        images=tuple(image_entries),
        image_truths=truths,
        keypoint_versions=versions,
        scanned_pngs=scanned,
        resolution=(int(resolution[0]), int(resolution[1])),
        pixel_mode=str(pixel_mode),
        content_hash=content_hash,
    )


def _truth_payload(truth: ImageTruth | None) -> dict[str, Any] | None:
    if truth is None:
        return None
    return {
        "image": truth.image,
        "columns": [column.as_dict() for column in truth.columns],
        "keypoints": [point.as_dict() for point in truth.keypoints],
        "pixel_size": list(truth.pixel_size),
        "extra": truth.extra,
    }


def _validate_png(path: Path, name: str, failures: FailureLog) -> None:
    if not path.is_file():
        return
    try:
        with path.open("rb") as handle:
            header = handle.read(33)
        if len(header) < 33 or header[:8] != PNG_SIGNATURE:
            failures.add("E_IMAGE_HEADER", "image", name, "invalid PNG signature", path=str(path))
            return
        length = struct.unpack(">I", header[8:12])[0]
        chunk_type = header[12:16]
        if length != 13 or chunk_type != b"IHDR":
            failures.add("E_IMAGE_HEADER", "image", name, "PNG must begin with a 13-byte IHDR chunk", path=str(path))
            return
        width, height, bit_depth, color_type = struct.unpack(">IIBB", header[16:26])
        if (width, height) != FIXED_RESOLUTION:
            failures.add(
                "E_IMAGE_SIZE",
                "image",
                name,
                "image resolution does not match the fixed layout",
                expected=list(FIXED_RESOLUTION),
                actual=[width, height],
            )
        if bit_depth != 8 or color_type != 0:
            failures.add(
                "E_IMAGE_MODE",
                "image",
                name,
                "image must use 8-bit grayscale PNG storage",
                bit_depth=bit_depth,
                color_type=color_type,
            )
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode != FIXED_PIXEL_MODE:
                failures.add(
                    "E_IMAGE_MODE",
                    "image",
                    name,
                    "decoded pixel mode must be L",
                    expected=FIXED_PIXEL_MODE,
                    actual=image.mode,
                )
    except Exception as exc:
        failures.add("E_IMAGE_HEADER", "image", name, f"cannot decode PNG container: {exc}", path=str(path))


def _parse_truth(name: str, raw: Any, failures: FailureLog) -> ImageTruth | None:
    if not isinstance(raw, dict):
        failures.add("E_TRUTH", "image", name, "truth document must be an object")
        return None
    columns = _parse_columns(name, raw.get("columns", []), failures)
    if columns is None:
        columns = []
    keypoints = _parse_keypoints(name, raw.get("keypoints", []), columns, failures, context="truth")
    if keypoints is None:
        keypoints = []
    pixel_size_raw = raw.get("resolution", {})
    if isinstance(pixel_size_raw, dict):
        pixel_size = (
            pixel_size_raw.get("width", FIXED_RESOLUTION[0]),
            pixel_size_raw.get("height", FIXED_RESOLUTION[1]),
        )
    else:
        pixel_size = FIXED_RESOLUTION
    if tuple(pixel_size) != FIXED_RESOLUTION:
        failures.add(
            "E_IMAGE_SIZE",
            "image",
            name,
            "truth resolution does not match the fixed layout",
            expected=list(FIXED_RESOLUTION),
            actual=list(pixel_size),
        )
    _validate_reading_sequence(name, keypoints, columns, failures)
    return ImageTruth(
        image=name,
        columns=tuple(columns),
        keypoints=tuple(keypoints),
        pixel_size=(int(pixel_size[0]), int(pixel_size[1])),
        extra={key: value for key, value in raw.items() if key not in {"columns", "keypoints", "resolution"}},
    )


def _parse_columns(name: str, raw: Any, failures: FailureLog) -> list[ColumnTruth] | None:
    if not isinstance(raw, list):
        failures.add("E_TRUTH", "image", name, "truth columns must be an array")
        return None
    columns: list[ColumnTruth] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            failures.add("E_TRUTH", "image", name, f"column[{index}] must be an object")
            continue
        column_id = item.get("id")
        if not isinstance(column_id, str) or not column_id:
            failures.add("E_TRUTH", "image", name, f"column[{index}] has no valid id")
            continue
        if column_id in seen:
            failures.add("E_TRUTH", "image", name, "duplicate column id", column=column_id)
            continue
        seen.add(column_id)
        values = {key: item.get(key) for key in ("x0", "x1", "y0", "y1", "char_count", "start_reading_index")}
        invalid = [key for key, value in values.items() if not require_int(value)]
        if invalid:
            failures.add(
                "E_TRUTH",
                "image",
                name,
                "column geometry and counts must be integers",
                column=column_id,
                fields=invalid,
            )
            continue
        x0, x1, y0, y1 = values["x0"], values["x1"], values["y0"], values["y1"]
        if not (0 <= x0 < x1 <= FIXED_RESOLUTION[0] and 0 <= y0 < y1 <= FIXED_RESOLUTION[1]):
            failures.add(
                "E_COLUMN_INTERVAL",
                "image",
                name,
                "column interval is empty or outside the image",
                column=column_id,
                interval=[x0, y0, x1, y1],
            )
        if values["char_count"] < 0 or values["start_reading_index"] < 0:
            failures.add("E_TRUTH", "image", name, "column count and start index cannot be negative", column=column_id)
        order = item.get("order", len(columns))
        if not require_int(order):
            failures.add("E_TRUTH", "image", name, "column order must be an integer", column=column_id)
            order = len(columns)
        columns.append(
            ColumnTruth(
                id=column_id,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                char_count=values["char_count"],
                start_reading_index=values["start_reading_index"],
                order=order,
            )
        )

    # The business reading direction is geometry-first: rightmost column has order 0.
    # A declared order is retained only as input metadata and never overrides geometry.
    columns.sort(key=lambda column: (-column.center_x, column.id))
    columns = [
        ColumnTruth(
            id=column.id,
            x0=column.x0,
            x1=column.x1,
            y0=column.y0,
            y1=column.y1,
            char_count=column.char_count,
            start_reading_index=column.start_reading_index,
            order=index,
        )
        for index, column in enumerate(columns)
    ]
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            if left.x0 < right.x1 and right.x0 < left.x1:
                failures.add(
                    "E_COLUMN_OVERLAP",
                    "image",
                    name,
                    "column x-intervals overlap",
                    columns=[left.id, right.id],
                    intervals=[[left.x0, left.x1], [right.x0, right.x1]],
                )
    return columns


def _parse_keypoints(
    name: str,
    raw: Any,
    columns: list[ColumnTruth],
    failures: FailureLog,
    *,
    context: str,
) -> list[TruthKeypoint] | None:
    if not isinstance(raw, list):
        failures.add("E_TRUTH", "image", name, f"{context} keypoints must be an array")
        return None
    column_ids = {column.id for column in columns}
    points: list[TruthKeypoint] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            failures.add("E_TRUTH", "image", name, f"{context} keypoint[{index}] must be an object")
            continue
        point_id = item.get("id", f"k{index:06d}")
        if not isinstance(point_id, str) or not point_id:
            failures.add("E_TRUTH", "image", name, f"{context} keypoint[{index}] has an invalid id")
            continue
        if point_id in seen_ids:
            failures.add("E_TRUTH", "image", name, "duplicate keypoint id", keypoint=point_id, context=context)
            continue
        seen_ids.add(point_id)
        x, y = item.get("x"), item.get("y")
        reading_index = item.get("reading_index", index)
        if not all(require_int(value) for value in (x, y, reading_index)):
            failures.add(
                "E_KEYPOINT_NOT_INTEGER",
                "keypoint",
                point_id,
                "keypoint coordinates and reading index must be integers",
                image=name,
                context=context,
            )
            continue
        column = item.get("column")
        if not isinstance(column, str) or column not in column_ids:
            failures.add(
                "E_KEYPOINT_OOB",
                "keypoint",
                point_id,
                "keypoint references an unknown column",
                image=name,
                context=context,
                column=column,
                reason="unknown_column",
            )
        if not (0 <= x < FIXED_RESOLUTION[0] and 0 <= y < FIXED_RESOLUTION[1]):
            failures.add(
                "E_KEYPOINT_OOB",
                "keypoint",
                point_id,
                "keypoint coordinate is outside the image",
                image=name,
                context=context,
                coordinate=[x, y],
                reason="outside_image",
            )
        if reading_index < 0:
            failures.add(
                "E_READING_SEQUENCE",
                "keypoint",
                point_id,
                "reading index cannot be negative",
                image=name,
                context=context,
            )
        points.append(
            TruthKeypoint(
                id=point_id,
                column=column if isinstance(column, str) else "",
                x=x,
                y=y,
                reading_index=reading_index,
                char=str(item.get("char", "")),
            )
        )
    points.sort(key=lambda point: (point.reading_index, point.id))
    return points


def _parse_version_points(
    image: str,
    version: str,
    raw: Any,
    truth: ImageTruth | None,
    failures: FailureLog,
) -> tuple[TruthKeypoint, ...] | None:
    if not isinstance(raw, dict):
        failures.add("E_KEYPOINT_VERSION", "image", f"{image}:{version}", "version document must be an object")
        return None
    document_image = raw.get("image", image)
    if document_image != image:
        failures.add(
            "E_KEYPOINT_VERSION",
            "image",
            f"{image}:{version}",
            "version document image disagrees with manifest",
            expected=image,
            actual=document_image,
        )
    expected_version = raw.get("version", version)
    if expected_version != version:
        failures.add(
            "E_KEYPOINT_VERSION",
            "image",
            f"{image}:{version}",
            "version label disagrees with manifest",
            manifest=version,
            document=expected_version,
        )
    columns = list(truth.columns) if truth is not None else []
    points = _parse_keypoints(image, raw.get("keypoints", []), columns, failures, context=f"version:{version}")
    if points is None:
        return None
    _validate_reading_sequence(image, points, columns, failures, context=f"version:{version}")
    return tuple(points)


def _validate_reading_sequence(
    name: str,
    points: list[TruthKeypoint],
    columns: list[ColumnTruth],
    failures: FailureLog,
    *,
    context: str = "truth",
) -> None:
    indices = [point.reading_index for point in points]
    if len(set(indices)) != len(indices):
        failures.add(
            "E_READING_SEQUENCE",
            "image",
            name,
            "reading indices must be unique",
            context=context,
        )
    if points and sorted(indices) != list(range(len(points))):
        failures.add(
            "E_READING_SEQUENCE",
            "image",
            name,
            "reading indices must be continuous from zero",
            context=context,
            indices=sorted(indices),
        )
    column_map = {column.id: column for column in columns}
    for column in columns:
        own = sorted(point.reading_index for point in points if point.column == column.id)
        if own and own != list(range(own[0], own[0] + len(own))):
            failures.add(
                "E_READING_SEQUENCE",
                "image",
                name,
                "reading indices within a column must be continuous",
                context=context,
                column=column.id,
                indices=own,
            )
    for column in columns:
        actual = sum(1 for point in points if point.column == column.id)
        if column.char_count != actual:
            failures.add(
                "E_READING_SEQUENCE",
                "image",
                name,
                "column char_count does not match its keypoint count",
                context=context,
                column=column.id,
                declared=column.char_count,
                actual=actual,
            )
        if column.id in column_map and points:
            own = [point for point in points if point.column == column.id]
            if own and min(point.reading_index for point in own) != column.start_reading_index:
                failures.add(
                    "E_READING_SEQUENCE",
                    "image",
                    name,
                    "column start_reading_index is inconsistent",
                    context=context,
                    column=column.id,
                    declared=column.start_reading_index,
                    actual=min(point.reading_index for point in own),
                )
