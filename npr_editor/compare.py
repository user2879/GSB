"""Persisted, reproducible version comparison and difference attribution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import NprError, require
from .geometry import flatten_shape
from .model import Project, Shape, canonical_hash, canonical_json, project_from_dict, version_stamp_key
from .png import encode_png, side_by_side
from .render import RenderConfig, RenderEngine
from .snapshots import COMMITTED_NAME, MANIFEST_NAME, atomic_write_bytes

COMPARISON_FORMAT = "npr-comparison-v1"


@dataclass
class VersionTrace:
    version_stamp: str
    source_schema_version: int | None = None
    edited_shapes: tuple[str, ...] = ()
    edit_actions: list[dict[str, Any]] = field(default_factory=list)
    migration_actions: list[dict[str, Any]] = field(default_factory=list)
    conversion_actions: list[dict[str, Any]] = field(default_factory=list)
    binary_format_version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_stamp": self.version_stamp,
            "source_schema_version": self.source_schema_version,
            "edited_shapes": sorted(set(self.edited_shapes)),
            "edit_actions": self.edit_actions,
            "migration_actions": self.migration_actions,
            "conversion_actions": self.conversion_actions,
            "binary_format_version": self.binary_format_version,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VersionTrace":
        require(
            isinstance(raw, dict),
            "E_SCHEMA_VERSION",
            "version trace must be an object",
        )
        stamp = raw.get("version_stamp")
        require(
            isinstance(stamp, str) and stamp.isdigit(),
            "E_SCHEMA_VERSION",
            "version trace stamp must be a decimal string",
        )
        edited = raw.get("edited_shapes", [])
        require(
            isinstance(edited, list) and all(isinstance(item, str) for item in edited),
            "E_SCHEMA_VERSION",
            "edited_shapes must be a string array",
        )
        actions = raw.get("edit_actions", [])
        migrations = raw.get("migration_actions", [])
        conversions = raw.get("conversion_actions", [])
        require(
            all(isinstance(items, list) for items in (actions, migrations, conversions)),
            "E_SCHEMA_VERSION",
            "version trace actions must be arrays",
        )
        return cls(
            version_stamp=stamp,
            source_schema_version=raw.get("source_schema_version"),
            edited_shapes=tuple(edited),
            edit_actions=list(actions),
            migration_actions=list(migrations),
            conversion_actions=list(conversions),
            binary_format_version=raw.get("binary_format_version"),
        )


@dataclass
class ComparisonResult:
    result: dict[str, Any]
    left_image: np.ndarray
    right_image: np.ndarray
    side_image: np.ndarray
    side_png: bytes
    comparison_hash: str


def trace_from_migration(migration: Any, *, edited_shapes: Iterable[str] = ()) -> VersionTrace:
    return VersionTrace(
        version_stamp=migration.project.version_stamp,
        source_schema_version=migration.source_version,
        edited_shapes=tuple(edited_shapes),
        migration_actions=[action.to_dict() for action in migration.actions],
    )


def _shape_control_entries(shape: Shape) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for contour_index, contour in enumerate(shape.contours):
        for segment_index, segment in enumerate(contour.segments):
            for point_index, point in enumerate(segment.points):
                entries.append(
                    {
                        "contour": contour_index,
                        "segment": segment_index,
                        "point": point_index,
                        "segment_type": segment.type,
                        "value": [point[0], point[1]],
                    }
                )
    return entries


def _shape_geometry_equal(left: Shape, right: Shape, left_project: Project, right_project: Project) -> bool:
    try:
        left_flat = flatten_shape(left, left_project.stroke_sample)
        right_flat = flatten_shape(right, right_project.stroke_sample)
    except NprError:
        return False
    if len(left_flat.contours) != len(right_flat.contours):
        return False
    for left_contour, right_contour in zip(left_flat.contours, right_flat.contours):
        if left_contour.closed != right_contour.closed:
            return False
        if left_contour.vertices.shape != right_contour.vertices.shape:
            return False
        if not np.allclose(left_contour.vertices, right_contour.vertices, rtol=0.0, atol=1e-6):
            return False
    return True


def _attribution(
    *,
    shape_id: str | None,
    left: VersionTrace,
    right: VersionTrace,
    flattened_equal: bool | None = None,
) -> list[str]:
    sources: list[str] = []
    edited = set(left.edited_shapes) | set(right.edited_shapes)
    if shape_id is not None and shape_id in edited:
        sources.append("edit")
    elif shape_id is None and (left.edit_actions or right.edit_actions):
        sources.append("edit")
    if left.migration_actions or right.migration_actions:
        sources.append("migration")
    if left.conversion_actions or right.conversion_actions:
        sources.append("format_conversion")
    if shape_id is not None and flattened_equal is True and "format_conversion" in sources:
        # A representation-only control-point change is exactly what the
        # binary skeleton round-trip can explain.
        return ["format_conversion"]
    if sources:
        # Migration/conversion never rewrite contour controls. If controls
        # changed without an edit, an attribution to those processes is false.
        if shape_id is not None and "edit" not in sources:
            return []
        return sorted(set(sources))
    return []


def _difference(
    *,
    kind: str,
    shape_id: str | None,
    sources: list[str],
    before: Any = None,
    after: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "shape_id": shape_id,
        "sources": sources,
        "suspicious": not sources,
        "before": before,
        "after": after,
    }
    record.update(extra)
    return record


def compare_versions(
    left_project: Project,
    right_project: Project,
    left_trace: VersionTrace,
    right_trace: VersionTrace,
    *,
    config: RenderConfig | None = None,
) -> ComparisonResult:
    require(
        left_project.project_id == right_project.project_id,
        "E_SCHEMA_VERSION",
        "cannot compare projects with different project ids",
        left=left_project.project_id,
        right=right_project.project_id,
    )
    require(
        left_trace.version_stamp == left_project.version_stamp
        and right_trace.version_stamp == right_project.version_stamp,
        "E_SCHEMA_VERSION",
        "version trace stamps do not match their projects",
    )
    config = config or RenderConfig()
    left_render = RenderEngine(left_project, config).render("lineart")
    right_render = RenderEngine(right_project, config).render("lineart")
    side_image = side_by_side(left_render.image, right_render.image)
    side_png = encode_png(side_image)
    differences: list[dict[str, Any]] = []

    left_schema_version = (
        left_trace.source_schema_version
        if left_trace.source_schema_version is not None
        else left_project.schema_version
    )
    right_schema_version = (
        right_trace.source_schema_version
        if right_trace.source_schema_version is not None
        else right_project.schema_version
    )
    schema_changed = left_schema_version != right_schema_version
    binary_changed = left_trace.binary_format_version != right_trace.binary_format_version
    if schema_changed or binary_changed:
        sources = _attribution(shape_id=None, left=left_trace, right=right_trace)
        if not sources:
            sources = []
        differences.append(
            _difference(
                kind="format_version",
                shape_id=None,
                sources=sources,
                before={
                    "schema_version": left_schema_version,
                    "binary_format_version": left_trace.binary_format_version,
                },
                after={
                    "schema_version": right_schema_version,
                    "binary_format_version": right_trace.binary_format_version,
                },
            )
        )

    if left_project.layer_order != right_project.layer_order:
        sources = _attribution(shape_id=None, left=left_trace, right=right_trace)
        differences.append(
            _difference(
                kind="layer_order",
                shape_id=None,
                sources=sources,
                before=left_project.layer_order,
                after=right_project.layer_order,
            )
        )

    left_style = left_project.style.to_dict()
    right_style = right_project.style.to_dict()
    for key in sorted(set(left_style) | set(right_style)):
        if left_style.get(key) != right_style.get(key):
            sources = _attribution(shape_id=None, left=left_trace, right=right_trace)
            differences.append(
                _difference(
                    kind="style",
                    shape_id=None,
                    sources=sources,
                    before=left_style.get(key),
                    after=right_style.get(key),
                    field=key,
                )
            )

    for key in sorted(set(left_project.stroke_sample.to_dict()) | set(right_project.stroke_sample.to_dict())):
        before = left_project.stroke_sample.to_dict().get(key)
        after = right_project.stroke_sample.to_dict().get(key)
        if before != after:
            sources = _attribution(shape_id=None, left=left_trace, right=right_trace)
            differences.append(
                _difference(
                    kind="stroke_sample",
                    shape_id=None,
                    sources=sources,
                    before=before,
                    after=after,
                    field=key,
                )
            )

    left_shapes = {shape.id: shape for shape in left_project.shapes}
    right_shapes = {shape.id: shape for shape in right_project.shapes}
    for shape_id in sorted(set(left_shapes) | set(right_shapes)):
        left_shape = left_shapes.get(shape_id)
        right_shape = right_shapes.get(shape_id)
        if left_shape is None or right_shape is None:
            sources = _attribution(shape_id=shape_id, left=left_trace, right=right_trace)
            differences.append(
                _difference(
                    kind="shape_presence",
                    shape_id=shape_id,
                    sources=sources,
                    before="absent" if left_shape is None else "present",
                    after="absent" if right_shape is None else "present",
                )
            )
            continue

        unchanged_geometry = _shape_geometry_equal(
            left_shape, right_shape, left_project, right_project
        )
        if left_shape.layer != right_shape.layer:
            sources = _attribution(
                shape_id=shape_id,
                left=left_trace,
                right=right_trace,
                flattened_equal=unchanged_geometry,
            )
            differences.append(
                _difference(
                    kind="layer",
                    shape_id=shape_id,
                    sources=sources,
                    before=left_shape.layer,
                    after=right_shape.layer,
                )
            )
        if left_shape.stroke_width != right_shape.stroke_width:
            sources = _attribution(
                shape_id=shape_id,
                left=left_trace,
                right=right_trace,
                flattened_equal=unchanged_geometry,
            )
            differences.append(
                _difference(
                    kind="stroke_width",
                    shape_id=shape_id,
                    sources=sources,
                    before=left_shape.stroke_width,
                    after=right_shape.stroke_width,
                )
            )
        if left_shape.style_tags != right_shape.style_tags:
            sources = _attribution(
                shape_id=shape_id,
                left=left_trace,
                right=right_trace,
                flattened_equal=unchanged_geometry,
            )
            differences.append(
                _difference(
                    kind="style_tags",
                    shape_id=shape_id,
                    sources=sources,
                    before=left_shape.style_tags,
                    after=right_shape.style_tags,
                )
            )

        left_entries = _shape_control_entries(left_shape)
        right_entries = _shape_control_entries(right_shape)
        if len(left_entries) != len(right_entries):
            sources = _attribution(
                shape_id=shape_id,
                left=left_trace,
                right=right_trace,
                flattened_equal=unchanged_geometry,
            )
            differences.append(
                _difference(
                    kind="control_structure",
                    shape_id=shape_id,
                    sources=sources,
                    before=len(left_entries),
                    after=len(right_entries),
                )
            )
        else:
            for left_entry, right_entry in zip(left_entries, right_entries):
                if (
                    left_entry["segment_type"] != right_entry["segment_type"]
                    or left_entry["value"] != right_entry["value"]
                ):
                    entry_equal = unchanged_geometry
                    sources = _attribution(
                        shape_id=shape_id,
                        left=left_trace,
                        right=right_trace,
                        flattened_equal=entry_equal,
                    )
                    differences.append(
                        _difference(
                            kind="control_point",
                            shape_id=shape_id,
                            sources=sources,
                            before=left_entry,
                            after=right_entry,
                            contour=left_entry["contour"],
                            segment=left_entry["segment"],
                            point=left_entry["point"],
                        )
                    )

    migration_summary = {
        "left": left_trace.migration_actions,
        "right": right_trace.migration_actions,
    }
    conversion_summary = {
        "left": left_trace.conversion_actions,
        "right": right_trace.conversion_actions,
    }
    suspicious_items = sorted(
        {
            f"{difference['kind']}:{difference.get('shape_id') or 'project'}"
            for difference in differences
            if difference["suspicious"]
        }
    )
    result: dict[str, Any] = {
        "format": COMPARISON_FORMAT,
        "project_id": left_project.project_id,
        "render_config": config.to_dict(),
        "left": {
            "version_stamp": left_project.version_stamp,
            "project_hash": left_project.content_hash(),
            "geometry_hash": left_project.geometry_hash(),
            "lineart_pixel_sha256": left_render.pixel_sha256,
            "lineart_png_sha256": left_render.png_sha256,
            "shape_render_hashes": left_render.shape_hashes,
        },
        "right": {
            "version_stamp": right_project.version_stamp,
            "project_hash": right_project.content_hash(),
            "geometry_hash": right_project.geometry_hash(),
            "lineart_pixel_sha256": right_render.pixel_sha256,
            "lineart_png_sha256": right_render.png_sha256,
            "shape_render_hashes": right_render.shape_hashes,
        },
        "differences": differences,
        "suspicious_items": suspicious_items,
        "migration_actions": migration_summary,
        "conversion_actions": conversion_summary,
        "side_by_side_png_sha256": hashlib.sha256(side_png).hexdigest(),
    }
    comparison_hash = canonical_hash(result)
    result["comparison_hash"] = comparison_hash
    return ComparisonResult(
        result=result,
        left_image=left_render.image,
        right_image=right_render.image,
        side_image=side_image,
        side_png=side_png,
        comparison_hash=comparison_hash,
    )


class ComparisonStore:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.root = self.workspace / "comparisons"
        self.root.mkdir(parents=True, exist_ok=True)

    def _directory(self, left_stamp: str, right_stamp: str) -> Path:
        return self.root / f"{left_stamp}__{right_stamp}"

    def save(
        self,
        left_project: Project,
        right_project: Project,
        left_trace: VersionTrace,
        right_trace: VersionTrace,
        *,
        config: RenderConfig | None = None,
    ) -> tuple[dict[str, Any], Path]:
        comparison = compare_versions(
            left_project,
            right_project,
            left_trace,
            right_trace,
            config=config,
        )
        destination = self._directory(left_project.version_stamp, right_project.version_stamp)
        staging = self.root / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        payload = {
            "left_project.json": (canonical_json(left_project.to_dict()) + "\n").encode("utf-8"),
            "right_project.json": (canonical_json(right_project.to_dict()) + "\n").encode("utf-8"),
            "left_trace.json": (canonical_json(left_trace.to_dict()) + "\n").encode("utf-8"),
            "right_trace.json": (canonical_json(right_trace.to_dict()) + "\n").encode("utf-8"),
            "comparison.json": (canonical_json(comparison.result) + "\n").encode("utf-8"),
            "side_by_side.png": comparison.side_png,
        }
        manifest = {
            "format": "npr-comparison-store-v1",
            "left_stamp": left_project.version_stamp,
            "right_stamp": right_project.version_stamp,
            "comparison_hash": comparison.comparison_hash,
            "left_project_hash": left_project.content_hash(),
            "right_project_hash": right_project.content_hash(),
            "files": {
                name: {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
                for name, data in sorted(payload.items())
            },
        }
        manifest["snapshot_hash"] = canonical_hash(manifest)
        try:
            for name, data in payload.items():
                atomic_write_bytes(staging / name, data)
            atomic_write_bytes(
                staging / MANIFEST_NAME,
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )
            atomic_write_bytes(
                staging / COMMITTED_NAME,
                (manifest["snapshot_hash"] + "\n").encode("ascii"),
            )
            if destination.exists():
                existing = json.loads((destination / MANIFEST_NAME).read_text(encoding="utf-8"))
                if existing.get("snapshot_hash") != manifest["snapshot_hash"]:
                    raise NprError(
                        "E_SNAPSHOT_STALE",
                        "comparison destination already exists with different content",
                        path=str(destination),
                    )
                shutil.rmtree(staging, ignore_errors=True)
            else:
                os.rename(staging, destination)
        except BaseException:
            raise
        return comparison.result, destination

    def recompute(self, left_stamp: str, right_stamp: str) -> dict[str, Any]:
        directory = self._directory(left_stamp, right_stamp)
        require(
            (directory / COMMITTED_NAME).is_file(),
            "E_SNAPSHOT_STALE",
            "comparison is not committed",
            path=str(directory),
        )
        manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        unhashed_manifest = dict(manifest)
        stored_snapshot_hash = unhashed_manifest.pop("snapshot_hash", None)
        require(
            stored_snapshot_hash is not None
            and canonical_hash(unhashed_manifest) == stored_snapshot_hash,
            "E_SNAPSHOT_STALE",
            "comparison manifest content hash is invalid",
        )
        require(
            (directory / COMMITTED_NAME).read_text(encoding="ascii").strip()
            == stored_snapshot_hash,
            "E_SNAPSHOT_STALE",
            "comparison commit marker does not match manifest",
        )
        stored_result = json.loads((directory / "comparison.json").read_text(encoding="utf-8"))
        left_project = project_from_dict(json.loads((directory / "left_project.json").read_text(encoding="utf-8")))
        right_project = project_from_dict(json.loads((directory / "right_project.json").read_text(encoding="utf-8")))
        left_trace = VersionTrace.from_dict(
            json.loads((directory / "left_trace.json").read_text(encoding="utf-8"))
        )
        right_trace = VersionTrace.from_dict(
            json.loads((directory / "right_trace.json").read_text(encoding="utf-8"))
        )
        file_manifest = manifest.get("files", {})
        for filename in (
            "left_project.json",
            "right_project.json",
            "left_trace.json",
            "right_trace.json",
            "comparison.json",
            "side_by_side.png",
        ):
            path = directory / filename
            require(
                path.is_file(),
                "E_SNAPSHOT_STALE",
                "comparison file is missing",
                file=filename,
            )
            expected = file_manifest.get(filename, {}).get("sha256")
            require(
                expected == hashlib.sha256(path.read_bytes()).hexdigest(),
                "E_SNAPSHOT_STALE",
                "comparison file hash mismatch",
                file=filename,
            )
        config_raw = stored_result.get("render_config", {})
        try:
            config = RenderConfig(**config_raw)
        except (TypeError, NprError) as exc:
            raise NprError(
                "E_SNAPSHOT_STALE", "stored comparison render config is invalid"
            ) from exc
        recomputed = compare_versions(
            left_project, right_project, left_trace, right_trace, config=config
        )
        require(
            recomputed.result == stored_result,
            "E_RENDER_MISMATCH",
            "comparison recomputation differs from persisted result",
        )
        require(
            manifest.get("comparison_hash") == recomputed.comparison_hash,
            "E_SNAPSHOT_STALE",
            "comparison manifest hash is stale",
        )
        return recomputed.result
