"""Incremental editing engine with exact closure accounting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .errors import (
    E_CLOSURE_MISMATCH,
    E_EDIT_TARGET,
    E_SNAPSHOT_STALE,
    E_STROKE_SAMPLE,
    NPRError,
)
from .geometry import GeometryRecord, GeometryStore, StrokeSkeleton
from .model import Element, Point, Scene, StyleParams, validate_scene
from .raster import RenderOutcome, render_snapshot
from .snapshot import RenderSnapshot, RenderState, STYLE_MODES, SnapshotStore
from .strategy import StrategyDecision, StrategySelector, scene_render_stats


@dataclass(frozen=True, slots=True)
class EditRecord:
    operation: str
    changed: bool
    element_ids: tuple[str, ...]
    layer_ids: tuple[str, ...]
    recomputed_elements: tuple[str, ...]
    recomputed_layers: tuple[str, ...]
    geometry_rebuilt: tuple[str, ...]
    old_version: int
    new_version: int
    old_snapshot_hash: str
    new_snapshot_hash: str
    old_scene_hash: str
    new_scene_hash: str
    skeleton_hashes_unchanged_outside_closure: bool
    render_hashes_unchanged_outside_closure: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "changed": self.changed,
            "closure": {
                "elements": list(self.element_ids),
                "layers": list(self.layer_ids),
            },
            "actual_recomputed": {
                "elements": list(self.recomputed_elements),
                "layers": list(self.recomputed_layers),
                "geometry": list(self.geometry_rebuilt),
            },
            "old_version": self.old_version,
            "new_version": self.new_version,
            "old_snapshot_hash": self.old_snapshot_hash,
            "new_snapshot_hash": self.new_snapshot_hash,
            "old_scene_hash": self.old_scene_hash,
            "new_scene_hash": self.new_scene_hash,
            "skeleton_hashes_unchanged_outside_closure": self.skeleton_hashes_unchanged_outside_closure,
            "render_hashes_unchanged_outside_closure": self.render_hashes_unchanged_outside_closure,
        }


class NprEngine:
    """Mutable edit session whose readers only see immutable snapshots."""

    def __init__(
        self,
        scene: Scene,
        *,
        viewport: tuple[int, int] = (256, 192),
        mode: str = "outline",
        tier: int | None = None,
        budget: Mapping[str, float] | None = None,
        gl_backend: object | None = None,
        geometry_records: Mapping[str, GeometryRecord] | None = None,
    ) -> None:
        validate_scene(scene)
        if mode not in STYLE_MODES:
            raise NPRError(E_STROKE_SAMPLE, "unsupported style mode", mode=mode)
        self.scene = scene
        self.viewport = (int(viewport[0]), int(viewport[1]))
        self.mode = mode
        self.tier = int(tier if tier is not None else len(scene.elements))
        self.budget = dict(budget) if budget else None
        self.gl_backend = gl_backend
        self.geometry = GeometryStore.from_records(geometry_records) if geometry_records else GeometryStore()
        self.selector = StrategySelector()
        self.snapshots = SnapshotStore()
        self.strategy: StrategyDecision | None = None
        self._initialised = False

    @classmethod
    def from_snapshot(
        cls,
        scene: Scene,
        snapshot: RenderSnapshot,
        *,
        tier: int | None = None,
        budget: Mapping[str, float] | None = None,
    ) -> "NprEngine":
        snapshot.verify()
        if snapshot.scene_hash != scene.content_hash:
            raise NPRError(E_SNAPSHOT_STALE, "snapshot belongs to a different scene", snapshot_version=snapshot.version)
        engine = cls(scene, viewport=snapshot.viewport, mode=snapshot.mode, tier=tier, budget=budget)
        records: dict[str, GeometryRecord] = {}
        for element_id, flat in snapshot.flat_geometry.items():
            skeleton = StrokeSkeleton(
                element_id=element_id,
                vertices=flat.points,
                widths=tuple(flat.stroke_width for _ in flat.points),
                indices=flat.indices,
                content_hash=flat.skeleton_hash,
            )
            records[element_id] = GeometryRecord(flat.element_hash, flat.tolerance, flat, skeleton)
        engine.geometry = GeometryStore.from_records(records)
        engine.snapshots = SnapshotStore(initial=snapshot)
        engine.mode = snapshot.mode
        engine._initialised = True
        engine.strategy = engine._select_strategy({element_id: record.flat for element_id, record in records.items()})
        return engine

    @property
    def render_state(self) -> RenderState:
        return RenderState(self.scene, self.current_snapshot)

    @property
    def current_snapshot(self) -> RenderSnapshot:
        snapshot = self.snapshots.current
        if snapshot is None:
            raise NPRError(E_SNAPSHOT_STALE, "engine has not rendered a snapshot yet")
        return snapshot

    def _select_strategy(self, flats: Mapping[str, Any]) -> StrategyDecision:
        stats = scene_render_stats(self.scene, flats, self.viewport)
        return self.selector.select(self.tier, stats, self.budget)

    def full_render(self, *, version: int = 1, mode: str | None = None) -> RenderOutcome:
        if mode is not None:
            if mode not in STYLE_MODES:
                raise NPRError(E_STROKE_SAMPLE, "unsupported style mode", mode=mode)
            self.mode = mode
        built = self.geometry.sync_scene(self.scene, self.viewport)
        flats = {element_id: record.flat for element_id, record in self.geometry.records.items()}
        self.strategy = self._select_strategy(flats)
        outcome = render_snapshot(
            self.scene,
            self.geometry,
            self.mode,
            self.viewport,
            version,
            strategy=self.strategy,
        )
        if self.snapshots.current is None:
            self.snapshots.initialize(outcome.snapshot)
        else:
            self.snapshots.publish(outcome.snapshot)
        self._initialised = True
        return outcome

    def set_style_mode(
        self,
        mode: str,
        *,
        expected_version: int | None = None,
        expected_hash: str | None = None,
    ) -> EditRecord:
        if mode not in STYLE_MODES:
            raise NPRError(E_STROKE_SAMPLE, "unsupported style mode", mode=mode)
        prior = self._read_expected(expected_version, expected_hash)
        if mode == self.mode:
            return self._unchanged_record("set_style_mode", prior)
        geometry_before = dict(self.geometry.build_counts)
        skeleton_before = dict(self.geometry.skeleton_hashes())
        dirty = {element.id for element in self.scene.elements}
        layers = {layer.id for layer in self.scene.layers}
        self.mode = mode
        outcome = render_snapshot(
            self.scene,
            self.geometry,
            self.mode,
            self.viewport,
            prior.version + 1,
            prior=prior,
            dirty_elements=dirty,
            dirty_layers=layers,
            strategy=self.strategy,
            gl_backend=self.gl_backend,
        )
        if outcome.snapshot.skeleton_hashes != skeleton_before:
            raise NPRError(E_CLOSURE_MISMATCH, "style switch changed outline geometry")
        geometry_rebuilt = tuple(
            sorted(element_id for element_id, count in self.geometry.build_counts.items() if geometry_before.get(element_id, 0) != count)
        )
        if geometry_rebuilt:
            raise NPRError(E_CLOSURE_MISMATCH, "style switch rebuilt flattened geometry", elements=list(geometry_rebuilt))
        self.snapshots.publish(outcome.snapshot, expected_version=prior.version)
        return EditRecord(
            operation="set_style_mode",
            changed=True,
            element_ids=tuple(sorted(dirty)),
            layer_ids=tuple(sorted(layers)),
            recomputed_elements=tuple(sorted(outcome.actual_elements)),
            recomputed_layers=tuple(sorted(outcome.actual_layers)),
            geometry_rebuilt=(),
            old_version=prior.version,
            new_version=outcome.snapshot.version,
            old_snapshot_hash=prior.snapshot_hash,
            new_snapshot_hash=outcome.snapshot.snapshot_hash,
            old_scene_hash=self.scene.content_hash,
            new_scene_hash=self.scene.content_hash,
            skeleton_hashes_unchanged_outside_closure=True,
            render_hashes_unchanged_outside_closure=True,
        )

    def _read_expected(self, expected_version: int | None, expected_hash: str | None) -> RenderSnapshot:
        if not self._initialised or self.snapshots.current is None:
            raise NPRError(E_SNAPSHOT_STALE, "engine has no committed render snapshot")
        return self.snapshots.read(expected_version, expected_hash)

    def _closure(self, element: Element) -> tuple[set[str], set[str]]:
        return {element.id}, {element.layer_id}

    def _unchanged_record(self, operation: str, prior: RenderSnapshot) -> EditRecord:
        return EditRecord(
            operation=operation,
            changed=False,
            element_ids=(),
            layer_ids=(),
            recomputed_elements=(),
            recomputed_layers=(),
            geometry_rebuilt=(),
            old_version=prior.version,
            new_version=prior.version,
            old_snapshot_hash=prior.snapshot_hash,
            new_snapshot_hash=prior.snapshot_hash,
            old_scene_hash=self.scene.content_hash,
            new_scene_hash=self.scene.content_hash,
            skeleton_hashes_unchanged_outside_closure=True,
            render_hashes_unchanged_outside_closure=True,
        )

    def _commit_element_edit(
        self,
        new_element: Element,
        operation: str,
        expected_version: int | None,
        expected_hash: str | None,
    ) -> EditRecord:
        prior = self._read_expected(expected_version, expected_hash)
        old_scene = self.scene
        old_element = old_scene.element_map().get(new_element.id)
        if old_element is None:
            raise NPRError(E_EDIT_TARGET, "element not found", element_id=new_element.id)
        if old_element.as_dict() == new_element.as_dict():
            return self._unchanged_record(operation, prior)
        new_scene = old_scene.with_element(new_element)
        element_ids, layer_ids = self._closure(new_element)
        geometry_before = dict(self.geometry.build_counts)
        skeleton_before = dict(prior.skeleton_hashes)
        rebuild_set = self.geometry.sync_scene(new_scene, self.viewport, dirty_elements=element_ids)
        if rebuild_set != element_ids:
            raise NPRError(
                E_CLOSURE_MISMATCH,
                "geometry rebuild set differs from the element closure",
                closure=sorted(element_ids),
                actual=sorted(rebuild_set),
            )
        flats = {element_id: record.flat for element_id, record in self.geometry.records.items()}
        self.strategy = self._select_strategy(flats)
        outcome = render_snapshot(
            new_scene,
            self.geometry,
            self.mode,
            self.viewport,
            prior.version + 1,
            prior=prior,
            dirty_elements=element_ids,
            dirty_layers=layer_ids,
            strategy=self.strategy,
            gl_backend=self.gl_backend,
        )
        if outcome.actual_elements != element_ids or outcome.actual_layers != layer_ids:
            raise NPRError(
                E_CLOSURE_MISMATCH,
                "actual recomputation set differs from the closure",
                expected_elements=sorted(element_ids),
                actual_elements=sorted(outcome.actual_elements),
                expected_layers=sorted(layer_ids),
                actual_layers=sorted(outcome.actual_layers),
            )
        skeleton_after = dict(outcome.snapshot.skeleton_hashes)
        render_after = dict(outcome.snapshot.element_hashes)
        skeleton_outside_ok = all(
            skeleton_before.get(element_id) == value
            for element_id, value in skeleton_after.items()
            if element_id not in element_ids
        )
        render_outside_ok = all(
            prior.element_hashes.get(element_id) == value
            for element_id, value in render_after.items()
            if element_id not in element_ids
        )
        if not skeleton_outside_ok or not render_outside_ok:
            raise NPRError(E_CLOSURE_MISMATCH, "hash changed outside the edit closure")
        geometry_rebuilt = tuple(
            sorted(element_id for element_id, count in self.geometry.build_counts.items() if geometry_before.get(element_id, 0) != count)
        )
        self.snapshots.publish(outcome.snapshot, expected_version=prior.version)
        self.scene = new_scene
        return EditRecord(
            operation=operation,
            changed=True,
            element_ids=tuple(sorted(element_ids)),
            layer_ids=tuple(sorted(layer_ids)),
            recomputed_elements=tuple(sorted(outcome.actual_elements)),
            recomputed_layers=tuple(sorted(outcome.actual_layers)),
            geometry_rebuilt=geometry_rebuilt,
            old_version=prior.version,
            new_version=outcome.snapshot.version,
            old_snapshot_hash=prior.snapshot_hash,
            new_snapshot_hash=outcome.snapshot.snapshot_hash,
            old_scene_hash=old_scene.content_hash,
            new_scene_hash=new_scene.content_hash,
            skeleton_hashes_unchanged_outside_closure=skeleton_outside_ok,
            render_hashes_unchanged_outside_closure=render_outside_ok,
        )

    def edit_control_point(
        self,
        element_id: str,
        segment_index: int,
        point_index: int,
        x: float,
        y: float,
        *,
        expected_version: int | None = None,
        expected_hash: str | None = None,
    ) -> EditRecord:
        element = self.scene.element_map().get(element_id)
        if element is None:
            raise NPRError(E_EDIT_TARGET, "element not found", element_id=element_id)
        if segment_index < 0 or segment_index >= len(element.segments):
            raise NPRError(E_EDIT_TARGET, "segment index is outside the element", element_id=element_id, segment_index=segment_index)
        segment = element.segments[segment_index]
        if point_index < 0 or point_index >= len(segment.points):
            raise NPRError(E_EDIT_TARGET, "point index is outside the segment", element_id=element_id, point_index=point_index)
        replacement = Point(float(x), float(y))
        new_points = tuple(replacement if index == point_index else point for index, point in enumerate(segment.points))
        new_segment = replace(segment, points=new_points)
        new_segments = tuple(new_segment if index == segment_index else current for index, current in enumerate(element.segments))
        new_element = replace(element, segments=new_segments)
        return self._commit_element_edit(new_element, "edit_control_point", expected_version, expected_hash)

    def edit_stroke_width(
        self,
        element_id: str,
        width: float,
        *,
        expected_version: int | None = None,
        expected_hash: str | None = None,
    ) -> EditRecord:
        if not (float(width) > 0.0):
            raise NPRError(E_STROKE_SAMPLE, "stroke width must be > 0", width=width)
        element = self.scene.element_map().get(element_id)
        if element is None:
            raise NPRError(E_EDIT_TARGET, "element not found", element_id=element_id)
        new_element = replace(element, stroke_width=float(width))
        return self._commit_element_edit(new_element, "edit_stroke_width", expected_version, expected_hash)

    def edit_style_param(
        self,
        element_id: str,
        name: str,
        value: Any,
        *,
        expected_version: int | None = None,
        expected_hash: str | None = None,
    ) -> EditRecord:
        element = self.scene.element_map().get(element_id)
        if element is None:
            raise NPRError(E_EDIT_TARGET, "element not found", element_id=element_id)
        style = element.style
        if name == "tag":
            style = replace(style, tag=str(value))
        elif name in {"tone", "roughness"}:
            numeric = float(value)
            if numeric < 0.0 or numeric > 1.0:
                raise NPRError(E_STROKE_SAMPLE, "style scalar is outside [0, 1]", name=name, value=value)
            style = replace(style, **{name: numeric})
        elif name == "tint":
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise NPRError(E_STROKE_SAMPLE, "tint must contain three channels", value=value)
            tint = tuple(max(0, min(255, int(channel))) for channel in value)
            style = replace(style, tint=(int(tint[0]), int(tint[1]), int(tint[2])))
        else:
            raise NPRError(E_EDIT_TARGET, "unknown style parameter", name=name)
        return self._commit_element_edit(replace(element, style=style), "edit_style_param", expected_version, expected_hash)

    def read_snapshot(self, expected_version: int | None = None, expected_hash: str | None = None) -> RenderSnapshot:
        return self.snapshots.read(expected_version, expected_hash)
