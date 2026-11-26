"""Dataset, PNG, truth and annotation validation."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from gsb.config import RunConfig
from gsb.constants import E_CONFIG_INVALID, E_SAMPLE_FORMAT, SCHEMA_VERSION
from gsb.errors import GsbError, ValidationIssue, sort_issues
from gsb.models import (
    AnnotationVersion,
    BBox,
    ColumnSpec,
    DatasetState,
    SampleManifest,
    SampleRecord,
    TruthKeypoint,
)
from gsb.util import safe_relative_path, sha256_bytes, sha256_obj

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _issue(code: str, message: str, source: str, sample_id: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, source=source, sample_id=sample_id)


def validate_png_bytes(
    payload: bytes,
    *,
    expected_width: int,
    expected_height: int,
    expected_mode: str,
    source: str,
    sample_id: str | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not payload.startswith(PNG_SIGNATURE):
        return [_issue(E_SAMPLE_FORMAT, "PNG signature is missing", source, sample_id)]
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return [_issue(E_SAMPLE_FORMAT, f"PNG cannot be decoded: {exc}", source, sample_id)]
    if image.format != "PNG":
        issues.append(_issue(E_SAMPLE_FORMAT, "container format is not PNG", source, sample_id))
    if image.size != (expected_width, expected_height):
        issues.append(
            _issue(
                E_SAMPLE_FORMAT,
                f"resolution {image.width}x{image.height} does not match {expected_width}x{expected_height}",
                source,
                sample_id,
            )
        )
    if image.mode != expected_mode:
        issues.append(
            _issue(
                E_SAMPLE_FORMAT,
                f"pixel mode {image.mode} does not match {expected_mode}",
                source,
                sample_id,
            )
        )
    return issues


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def load_manifest(
    manifest_path: str | Path,
    *,
    config: RunConfig,
) -> tuple[SampleManifest | None, list[ValidationIssue]]:
    path = Path(manifest_path)
    source = str(path)
    issues: list[ValidationIssue] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [_issue(E_SAMPLE_FORMAT, "sample manifest not found", source)]
    except (OSError, json.JSONDecodeError) as exc:
        return None, [_issue(E_SAMPLE_FORMAT, f"sample manifest is invalid JSON: {exc}", source)]
    if not isinstance(raw, dict):
        return None, [_issue(E_SAMPLE_FORMAT, "sample manifest must be an object", source)]

    sample_id = raw.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        issues.append(_issue(E_SAMPLE_FORMAT, "sample_id must be a non-empty string", source))
        sample_id = None
    if raw.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue(E_SAMPLE_FORMAT, f"schema_version must be {SCHEMA_VERSION}", source, sample_id))

    image_name = raw.get("image")
    image_path: Path | None = None
    if not isinstance(image_name, str) or not image_name:
        issues.append(_issue(E_SAMPLE_FORMAT, "image path must be declared", source, sample_id))
    else:
        try:
            image_path = safe_relative_path(path.parent, image_name)
        except ValueError as exc:
            issues.append(_issue(E_SAMPLE_FORMAT, str(exc), source, sample_id))

    width = raw.get("width")
    height = raw.get("height")
    pixel_mode = raw.get("pixel_mode")
    for label, value in (("width", width), ("height", height)):
        if not _is_int(value) or value <= 0:
            issues.append(_issue(E_SAMPLE_FORMAT, f"{label} must be a positive integer", source, sample_id))
    if pixel_mode != config.layout.pixel_mode:
        issues.append(_issue(E_SAMPLE_FORMAT, "pixel_mode must be L", source, sample_id))
    if _is_int(width) and width != config.layout.width:
        issues.append(_issue(E_SAMPLE_FORMAT, "manifest width does not match configured layout", source, sample_id))
    if _is_int(height) and height != config.layout.height:
        issues.append(_issue(E_SAMPLE_FORMAT, "manifest height does not match configured layout", source, sample_id))

    template_version = raw.get("template_version")
    if not isinstance(template_version, str) or not template_version:
        issues.append(_issue(E_SAMPLE_FORMAT, "template_version must be declared", source, sample_id))
    elif template_version != config.recognition.template_version:
        issues.append(
            _issue(
                E_CONFIG_INVALID,
                "sample template_version does not match recognition configuration",
                source,
                sample_id,
            )
        )
    seed = raw.get("seed")
    if not _is_int(seed):
        issues.append(_issue(E_SAMPLE_FORMAT, "seed must be an integer", source, sample_id))

    columns: list[ColumnSpec] = []
    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, list) or not columns_raw:
        issues.append(_issue(E_SAMPLE_FORMAT, "columns must be a non-empty list", source, sample_id))
    else:
        seen_indices: set[int] = set()
        for index, item in enumerate(columns_raw):
            if not isinstance(item, dict):
                issues.append(_issue(E_SAMPLE_FORMAT, f"columns[{index}] must be an object", source, sample_id))
                continue
            column_index = item.get("index")
            x_min = item.get("x_min")
            x_max = item.get("x_max")
            if not all(_is_int(value) for value in (column_index, x_min, x_max)):
                issues.append(_issue(E_SAMPLE_FORMAT, f"columns[{index}] values must be integers", source, sample_id))
                continue
            if column_index in seen_indices:
                issues.append(_issue(E_SAMPLE_FORMAT, f"duplicate column index {column_index}", source, sample_id))
            seen_indices.add(column_index)
            if not (_is_int(width) and 0 <= x_min < x_max <= width):
                issues.append(_issue(E_SAMPLE_FORMAT, f"columns[{index}] interval is outside the image", source, sample_id))
                continue
            columns.append(ColumnSpec(column_index, x_min, x_max))
    ordered_columns = sorted(columns, key=lambda item: (item.x_min, item.index))
    for previous, current in zip(ordered_columns, ordered_columns[1:]):
        if current.x_min < previous.x_max:
            issues.append(
                _issue(
                    E_SAMPLE_FORMAT,
                    f"column x intervals overlap: {previous.index} and {current.index}",
                    source,
                    sample_id,
                )
            )

    keypoints: list[TruthKeypoint] = []
    keypoints_raw = raw.get("keypoints")
    column_lookup = {item.index: item for item in columns}
    if not isinstance(keypoints_raw, list) or not keypoints_raw:
        issues.append(_issue(E_SAMPLE_FORMAT, "keypoints must be a non-empty list", source, sample_id))
    else:
        seen_ids: set[str] = set()
        for index, item in enumerate(keypoints_raw):
            if not isinstance(item, dict):
                issues.append(_issue(E_SAMPLE_FORMAT, f"keypoints[{index}] must be an object", source, sample_id))
                continue
            keypoint_id = item.get("id")
            if not isinstance(keypoint_id, str) or not keypoint_id:
                issues.append(_issue(E_SAMPLE_FORMAT, f"keypoints[{index}].id must be a string", source, sample_id))
                continue
            if keypoint_id in seen_ids:
                issues.append(_issue(E_SAMPLE_FORMAT, f"duplicate keypoint id {keypoint_id}", source, sample_id))
            seen_ids.add(keypoint_id)
            x = item.get("x")
            y = item.get("y")
            column_index = item.get("column")
            reading_index = item.get("reading_index")
            bbox_raw = item.get("bbox")
            text = item.get("text")
            template_id = item.get("template_id")
            if not all(_is_int(value) for value in (x, y, column_index, reading_index)):
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} coordinates and indices must be integers", source, sample_id))
                continue
            if isinstance(width, int) and isinstance(height, int) and not (0 <= x < width and 0 <= y < height):
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} keypoint is outside the image", source, sample_id))
            column = column_lookup.get(column_index)
            if column is None:
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} refers to an unknown column", source, sample_id))
            elif not column.contains(x):
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} x coordinate is outside its column", source, sample_id))
            if reading_index < 0:
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} reading_index must be non-negative", source, sample_id))
            if (
                not isinstance(bbox_raw, list)
                or len(bbox_raw) != 4
                or any(not _is_int(value) for value in bbox_raw)
            ):
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} bbox must have four integers", source, sample_id))
                bbox = BBox(0, 0, 0, 0)
            else:
                bbox = BBox.from_list(bbox_raw)
                if not (
                    isinstance(width, int)
                    and isinstance(height, int)
                    and 0 <= bbox.left < bbox.right <= width
                    and 0 <= bbox.top < bbox.bottom <= height
                ):
                    issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} bbox is outside the image", source, sample_id))
                if not bbox.contains_point(x, y):
                    issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} keypoint is outside its bbox", source, sample_id))
            if not isinstance(text, str) or not text:
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} text must be declared", source, sample_id))
            if not isinstance(template_id, str) or not template_id:
                issues.append(_issue(E_SAMPLE_FORMAT, f"{keypoint_id} template_id must be declared", source, sample_id))
            keypoints.append(
                TruthKeypoint(
                    keypoint_id=keypoint_id,
                    x=x,
                    y=y,
                    bbox=bbox,
                    column=column_index,
                    reading_index=reading_index,
                    text=text if isinstance(text, str) else "",
                    template_id=template_id if isinstance(template_id, str) else "",
                )
            )

    reading_indices = sorted(item.reading_index for item in keypoints if item.reading_index >= 0)
    if reading_indices != list(range(len(keypoints))):
        issues.append(_issue(E_SAMPLE_FORMAT, "reading_index values must be continuous from zero", source, sample_id))
    expected_order = [
        item.keypoint_id
        for item in sorted(keypoints, key=lambda item: item.reading_index)
    ]
    reading_order_raw = raw.get("reading_order")
    if not isinstance(reading_order_raw, list) or reading_order_raw != expected_order:
        issues.append(_issue(E_SAMPLE_FORMAT, "reading_order must match continuous reading_index values", source, sample_id))

    image_sha256 = raw.get("image_sha256")
    if not isinstance(image_sha256, str) or len(image_sha256) != 64:
        issues.append(_issue(E_SAMPLE_FORMAT, "image_sha256 must be declared", source, sample_id))

    if issues or image_path is None or width is None or height is None:
        return None, sort_issues(issues)
    assert isinstance(sample_id, str)
    assert isinstance(template_version, str)
    assert isinstance(seed, int)
    assert isinstance(image_sha256, str)
    manifest = SampleManifest(
        sample_id=sample_id,
        manifest_path=path,
        image_path=image_path,
        width=width,
        height=height,
        pixel_mode="L",
        image_sha256=image_sha256,
        template_version=template_version,
        seed=seed,
        columns=tuple(sorted(columns, key=lambda item: item.index)),
        keypoints=tuple(sorted(keypoints, key=lambda item: item.reading_index)),
        reading_order=tuple(expected_order),
    )
    return manifest, sort_issues(issues)


def load_annotation_version(
    path: str | Path,
    *,
    manifest: SampleManifest,
    config: RunConfig,
) -> tuple[AnnotationVersion | None, list[ValidationIssue]]:
    annotation_path = Path(path)
    source = str(annotation_path)
    issues: list[ValidationIssue] = []
    try:
        raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [_issue(E_SAMPLE_FORMAT, "annotation version not found", source, manifest.sample_id)]
    except (OSError, json.JSONDecodeError) as exc:
        return None, [_issue(E_SAMPLE_FORMAT, f"annotation version is invalid JSON: {exc}", source, manifest.sample_id)]
    if not isinstance(raw, dict):
        return None, [_issue(E_SAMPLE_FORMAT, "annotation version must be an object", source, manifest.sample_id)]
    if raw.get("schema_version") != SCHEMA_VERSION:
        issues.append(_issue(E_SAMPLE_FORMAT, "annotation schema_version is invalid", source, manifest.sample_id))
    if raw.get("sample_id") != manifest.sample_id:
        issues.append(_issue(E_SAMPLE_FORMAT, "annotation sample_id does not match manifest", source, manifest.sample_id))
    version = raw.get("version")
    if not isinstance(version, str) or not version:
        issues.append(_issue(E_SAMPLE_FORMAT, "annotation version must be declared", source, manifest.sample_id))
    points_raw = raw.get("keypoints")
    keypoint_lookup = manifest.keypoint_by_id()
    points: dict[str, tuple[int, int]] = {}
    if not isinstance(points_raw, list) or not points_raw:
        issues.append(_issue(E_SAMPLE_FORMAT, "annotation keypoints must be a non-empty list", source, manifest.sample_id))
    else:
        for index, item in enumerate(points_raw):
            if not isinstance(item, dict):
                issues.append(_issue(E_SAMPLE_FORMAT, f"annotation keypoints[{index}] must be an object", source, manifest.sample_id))
                continue
            keypoint_id = item.get("id")
            x = item.get("x")
            y = item.get("y")
            if not isinstance(keypoint_id, str) or keypoint_id not in keypoint_lookup:
                issues.append(_issue(E_SAMPLE_FORMAT, f"annotation keypoints[{index}] id is unknown", source, manifest.sample_id))
                continue
            if not _is_int(x) or not _is_int(y):
                issues.append(_issue(E_SAMPLE_FORMAT, f"annotation {keypoint_id} coordinates must be integers", source, manifest.sample_id))
                continue
            if not (0 <= x < manifest.width and 0 <= y < manifest.height):
                issues.append(_issue(E_SAMPLE_FORMAT, f"annotation {keypoint_id} is outside the image", source, manifest.sample_id))
                continue
            if keypoint_id in points:
                issues.append(_issue(E_SAMPLE_FORMAT, f"duplicate annotation keypoint {keypoint_id}", source, manifest.sample_id))
                continue
            points[keypoint_id] = (x, y)
    if set(points) != set(keypoint_lookup):
        issues.append(_issue(E_SAMPLE_FORMAT, "annotation must replace every keypoint in the version", source, manifest.sample_id))
    if issues or not isinstance(version, str):
        return None, sort_issues(issues)
    content_fingerprint = sha256_obj(
        {
            "sample_id": manifest.sample_id,
            "version": version,
            "keypoints": {keypoint_id: list(value) for keypoint_id, value in sorted(points.items())},
        }
    )
    return (
        AnnotationVersion(
            sample_id=manifest.sample_id,
            version=version,
            path=annotation_path,
            content_fingerprint=content_fingerprint,
            keypoints=points,
        ),
        [],
    )


def load_sample_record(
    manifest_path: str | Path,
    *,
    config: RunConfig,
    annotation_path: str | Path | None = None,
) -> tuple[SampleRecord | None, list[ValidationIssue]]:
    manifest, issues = load_manifest(manifest_path, config=config)
    if manifest is None:
        return None, issues
    try:
        image_bytes = manifest.image_path.read_bytes()
    except OSError as exc:
        return None, [
            *_issue(E_SAMPLE_FORMAT, f"sample image cannot be read: {exc}", str(manifest.image_path), manifest.sample_id)
        ]
    image_issues = validate_png_bytes(
        image_bytes,
        expected_width=manifest.width,
        expected_height=manifest.height,
        expected_mode=manifest.pixel_mode,
        source=str(manifest.image_path),
        sample_id=manifest.sample_id,
    )
    image_fingerprint = sha256_bytes(image_bytes)
    if image_fingerprint != manifest.image_sha256:
        image_issues.append(
            _issue(E_SAMPLE_FORMAT, "image content fingerprint does not match manifest", str(manifest.image_path), manifest.sample_id)
        )
    issues.extend(image_issues)
    if issues:
        return None, sort_issues(issues)
    annotation: AnnotationVersion | None = None
    if annotation_path is not None:
        annotation, annotation_issues = load_annotation_version(
            annotation_path,
            manifest=manifest,
            config=config,
        )
        issues.extend(annotation_issues)
        if annotation_issues:
            return None, sort_issues(issues)
    return (
        SampleRecord(
            manifest=manifest,
            image_bytes=image_bytes,
            image_fingerprint=image_fingerprint,
            annotation=annotation,
        ),
        sort_issues(issues),
    )


def load_dataset(path: str | Path, *, config: RunConfig) -> tuple[DatasetState, list[ValidationIssue]]:
    dataset_path = Path(path)
    issues: list[ValidationIssue] = []
    try:
        raw = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DatasetState(dataset_path=dataset_path), [
            _issue(E_CONFIG_INVALID, "dataset file not found", str(dataset_path))
        ]
    except yaml.YAMLError as exc:
        return DatasetState(dataset_path=dataset_path), [
            _issue(E_CONFIG_INVALID, f"dataset YAML is invalid: {exc}", str(dataset_path))
        ]
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        return DatasetState(dataset_path=dataset_path), [
            _issue(E_CONFIG_INVALID, f"dataset schema_version must be {SCHEMA_VERSION}", str(dataset_path))
        ]
    samples_raw = raw.get("samples")
    state = DatasetState(dataset_path=dataset_path)
    if not isinstance(samples_raw, list) or not samples_raw:
        issues.append(_issue(E_CONFIG_INVALID, "dataset samples must be a non-empty list", str(dataset_path)))
        return state, sort_issues(issues)
    seen_ids: set[str] = set()
    for index, item in enumerate(samples_raw):
        if not isinstance(item, dict):
            issues.append(_issue(E_CONFIG_INVALID, f"samples[{index}] must be an object", str(dataset_path)))
            continue
        sample_id = item.get("id")
        manifest_name = item.get("manifest")
        annotation_name = item.get("annotation")
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(_issue(E_CONFIG_INVALID, f"samples[{index}].id must be a string", str(dataset_path)))
            continue
        if sample_id in seen_ids:
            issues.append(_issue(E_CONFIG_INVALID, f"duplicate sample id {sample_id}", str(dataset_path), sample_id))
            continue
        seen_ids.add(sample_id)
        if not isinstance(manifest_name, str) or not manifest_name:
            issues.append(_issue(E_CONFIG_INVALID, f"{sample_id} manifest must be declared", str(dataset_path), sample_id))
            continue
        try:
            manifest_path = safe_relative_path(dataset_path.parent, manifest_name)
        except ValueError as exc:
            issues.append(_issue(E_CONFIG_INVALID, str(exc), str(dataset_path), sample_id))
            continue
        annotation_path: Path | None = None
        if annotation_name is not None:
            if not isinstance(annotation_name, str) or not annotation_name:
                issues.append(_issue(E_CONFIG_INVALID, f"{sample_id} annotation must be a string", str(dataset_path), sample_id))
                continue
            try:
                annotation_path = safe_relative_path(dataset_path.parent, annotation_name)
            except ValueError as exc:
                issues.append(_issue(E_CONFIG_INVALID, str(exc), str(dataset_path), sample_id))
                continue
        record, sample_issues = load_sample_record(
            manifest_path,
            config=config,
            annotation_path=annotation_path,
        )
        issues.extend(sample_issues)
        if record is not None:
            if sample_id != record.sample_id:
                issues.append(
                    _issue(E_CONFIG_INVALID, "dataset id does not match manifest sample_id", str(dataset_path), sample_id)
                )
                continue
            state.sample_order.append(sample_id)
            state.samples[sample_id] = record
            if record.annotation is not None:
                state.annotation_versions[sample_id] = record.annotation.version
    return state, sort_issues(issues)


def scan_png_directory(directory: str | Path, *, config: RunConfig) -> list[ValidationIssue]:
    root = Path(directory)
    issues: list[ValidationIssue] = []
    for path in sorted(root.rglob("*.png"), key=lambda item: item.as_posix()):
        try:
            payload = path.read_bytes()
        except OSError as exc:
            issues.append(_issue(E_SAMPLE_FORMAT, f"PNG cannot be read: {exc}", str(path)))
            continue
        issues.extend(
            validate_png_bytes(
                payload,
                expected_width=config.layout.width,
                expected_height=config.layout.height,
                expected_mode=config.layout.pixel_mode,
                source=str(path),
            )
        )
    return sort_issues(issues)
