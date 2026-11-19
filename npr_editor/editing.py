"""Transactional contour editing and undo snapshots."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import NprError, require
from .geometry import GeometryEngine
from .model import (
    CURRENT_SCHEMA_VERSION,
    Project,
    bump_version_stamp,
    f32,
    project_from_dict,
)
from .render import RenderConfig, RenderEngine, RenderMode


@dataclass
class UndoSnapshot:
    project_document: dict[str, Any]
    affected_shapes: tuple[str, ...]
    label: str
    project_hash: str
    version_stamp: str


@dataclass
class UndoResult:
    restored_version_stamp: str
    project_hash: str
    remaining_undo: int


def _mark_provenance(project: Project, shape_id: str | None, action: str) -> None:
    provenance = copy.deepcopy(project.metadata.get("provenance", {}))
    edited = set(provenance.get("edited_shapes", []))
    if shape_id is not None:
        edited.add(shape_id)
    provenance["edited_shapes"] = sorted(edited)
    provenance["last_action"] = action
    project.metadata["provenance"] = provenance


class Editor:
    """A deterministic editing session sharing one geometry/render pipeline."""

    def __init__(
        self,
        project: Project,
        *,
        config: RenderConfig | None = None,
    ):
        self.project = project.clone()
        self.config = config or RenderConfig()
        self.geometry = GeometryEngine(self.project)
        self.renderer = RenderEngine(self.project, self.config, self.geometry)
        self._undo: list[UndoSnapshot] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def undo_depth(self) -> int:
        return len(self._undo)

    def _snapshot(self, *, affected_shapes: tuple[str, ...], label: str) -> UndoSnapshot:
        return UndoSnapshot(
            project_document=copy.deepcopy(self.project.to_dict()),
            affected_shapes=affected_shapes,
            label=label,
            project_hash=self.project.content_hash(),
            version_stamp=self.project.version_stamp,
        )

    def _activate(self, project: Project) -> None:
        self.project = project
        self.geometry.set_project(project)
        self.renderer.set_project(project)

    def _commit_document(self, document: dict[str, Any]) -> Project:
        project = project_from_dict(document)
        self._activate(project)
        return project

    def edit_control_point(
        self,
        shape_id: str,
        contour_index: int,
        segment_index: int,
        point_index: int,
        x: float,
        y: float,
    ) -> Project:
        try:
            shape = self.project.shape_by_id(shape_id)
        except KeyError as exc:
            raise NprError(
                "E_SCHEMA_VERSION", "shape id does not exist", shape_id=shape_id
            ) from exc
        require(
            0 <= contour_index < len(shape.contours),
            "E_SCHEMA_VERSION",
            "contour index out of range",
            shape=shape_id,
        )
        contour = shape.contours[contour_index]
        require(
            0 <= segment_index < len(contour.segments),
            "E_SCHEMA_VERSION",
            "segment index out of range",
            shape=shape_id,
        )
        segment = contour.segments[segment_index]
        require(
            0 <= point_index < len(segment.points),
            "E_SCHEMA_VERSION",
            "control point index out of range",
            shape=shape_id,
        )
        new_x = f32(x, field_name="control_point.x")
        new_y = f32(y, field_name="control_point.y")
        assert isinstance(new_x, float) and isinstance(new_y, float)
        if (new_x, new_y) == segment.points[point_index]:
            return self.project

        snapshot = self._snapshot(affected_shapes=(shape_id,), label="edit_control_point")
        document = self.project.to_dict()
        target = document["shapes"]
        for raw_shape in target:
            if raw_shape["id"] == shape_id:
                raw_shape["contours"][contour_index]["segments"][segment_index]["points"][
                    point_index
                ] = [float(np.float32(new_x)), float(np.float32(new_y))]
                break
        document["version_stamp"] = bump_version_stamp(self.project.version_stamp)
        project = project_from_dict(document)
        _mark_provenance(project, shape_id, "edit_control_point")
        self._undo.append(snapshot)
        self.geometry.invalidate(shape_id)
        self._activate(project)
        return project

    def set_shape_stroke_width(self, shape_id: str, width: float) -> Project:
        try:
            shape = self.project.shape_by_id(shape_id)
        except KeyError as exc:
            raise NprError(
                "E_SCHEMA_VERSION", "shape id does not exist", shape_id=shape_id
            ) from exc
        new_width = f32(width, field_name="stroke_width")
        require(new_width > 0.0, "E_SCHEMA_VERSION", "stroke width must be positive")
        if new_width == shape.stroke_width:
            return self.project
        snapshot = self._snapshot(affected_shapes=(shape_id,), label="set_stroke_width")
        document = self.project.to_dict()
        for raw_shape in document["shapes"]:
            if raw_shape["id"] == shape_id:
                raw_shape["stroke_width"] = new_width
                break
        document["version_stamp"] = bump_version_stamp(self.project.version_stamp)
        project = project_from_dict(document)
        _mark_provenance(project, shape_id, "set_stroke_width")
        self._undo.append(snapshot)
        self._activate(project)
        return project

    def set_style_parameter(self, name: str, value: Any) -> Project:
        if name not in self.project.style.to_dict():
            raise NprError(
                "E_SCHEMA_VERSION", "unknown style parameter", parameter=name
            )
        before = self.project.style.to_dict()
        if before[name] == value:
            return self.project
        snapshot = self._snapshot(affected_shapes=(), label=f"style:{name}")
        document = self.project.to_dict()
        document["style"][name] = copy.deepcopy(value)
        document["version_stamp"] = bump_version_stamp(self.project.version_stamp)
        project = project_from_dict(document)
        _mark_provenance(project, None, f"style:{name}")
        self._undo.append(snapshot)
        self._activate(project)
        return project

    def set_stroke_sample(self, **updates: Any) -> Project:
        if not updates:
            return self.project
        before = self.project.stroke_sample.to_dict()
        unknown = sorted(set(updates) - set(before))
        if unknown:
            raise NprError(
                "E_STROKE_SAMPLE", "unknown stroke sampling fields", fields=unknown
            )
        if all(before[key] == value for key, value in updates.items()):
            return self.project
        snapshot = self._snapshot(affected_shapes=(), label="stroke_sample")
        document = self.project.to_dict()
        document["stroke_sample"].update(updates)
        document["version_stamp"] = bump_version_stamp(self.project.version_stamp)
        try:
            project = project_from_dict(document)
        except NprError as exc:
            if exc.code == "E_SCHEMA_VERSION":
                raise NprError("E_STROKE_SAMPLE", exc.message, **exc.details) from exc
            raise
        _mark_provenance(project, None, "stroke_sample")
        self._undo.append(snapshot)
        self._activate(project)
        return project

    def undo(self) -> UndoResult:
        if not self._undo:
            raise NprError("E_UNDO_EMPTY", "undo stack is empty")
        snapshot = self._undo.pop()
        project = project_from_dict(copy.deepcopy(snapshot.project_document))
        require(
            project.content_hash() == snapshot.project_hash,
            "E_RENDER_MISMATCH",
            "undo snapshot content hash is corrupt",
        )
        for shape_id in snapshot.affected_shapes:
            self.geometry.invalidate(shape_id)
        self._activate(project)
        return UndoResult(
            restored_version_stamp=project.version_stamp,
            project_hash=project.content_hash(),
            remaining_undo=len(self._undo),
        )

    def render(self, mode: RenderMode = "lineart"):
        return self.renderer.render(mode)

    def geometry_hashes(self) -> dict[str, str]:
        return self.geometry.geometry_hashes()

    def render_hashes(self, mode: RenderMode = "lineart") -> dict[str, str]:
        return self.render(mode).shape_hashes
