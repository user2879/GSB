"""Application facade for versioned CAD editing and recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .errors import DomainError, ErrorCode
from .geometry import PatchGeometry
from .history import EditOperation, HistoryManager, StateLoader
from .model import generate_model
from .persistence import LoadedCommit, PersistenceReport, PersistentStore
from .render import DISPLAY_MODES, RenderBuffer
from .state import GeometryState, build_initial_state, derive_edited_state


@dataclass(frozen=True)
class ReconstructionReport:
    target_version: int
    patch_key: str
    applied_operations: tuple[int, ...]
    vertex_evaluations: int
    total_patch_vertices: int
    patches_rebuilt: int
    full_model_rebuild: bool
    expected_geometry_hash: str
    reconstructed_geometry_hash: str
    expected_normals_hash: str
    reconstructed_normals_hash: str
    max_abs_deviation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_version": self.target_version,
            "patch_key": self.patch_key,
            "applied_operations": list(self.applied_operations),
            "vertex_evaluations": self.vertex_evaluations,
            "total_patch_vertices": self.total_patch_vertices,
            "patches_rebuilt": self.patches_rebuilt,
            "full_model_rebuild": self.full_model_rebuild,
            "expected_geometry_hash": self.expected_geometry_hash,
            "reconstructed_geometry_hash": self.reconstructed_geometry_hash,
            "expected_normals_hash": self.expected_normals_hash,
            "reconstructed_normals_hash": self.reconstructed_normals_hash,
            "max_abs_deviation": self.max_abs_deviation,
            "match": (
                self.expected_geometry_hash == self.reconstructed_geometry_hash
                and self.expected_normals_hash == self.reconstructed_normals_hash
                and self.max_abs_deviation == 0.0
            ),
        }


class CadApplication:
    def __init__(
        self,
        *,
        root: Path,
        store: PersistentStore,
        history: HistoryManager,
        recovery: PersistenceReport | None,
    ) -> None:
        self.root = Path(root)
        self.store = store
        self.history = history
        self.recovery = recovery
        self._closed = False
        self._closed_report: dict[str, Any] | None = None

    @property
    def current(self) -> GeometryState:
        return self.history.current

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        seed: int = 20261119,
        max_undo_depth: int = 32,
        grid_resolution: int = 16,
    ) -> "CadApplication":
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise DomainError(
                ErrorCode.SNAPSHOT_STALE,
                "target workspace is not empty; choose a fresh or reset workspace",
                root=str(root),
            )
        first = generate_model(seed)
        second = generate_model(seed)
        if first.to_bytes() != second.to_bytes():
            raise DomainError(
                ErrorCode.GEOM_STALE,
                "deterministic sample generation produced different bytes",
                seed=seed,
            )
        state = build_initial_state(first, version=0, grid_resolution=grid_resolution)
        store = PersistentStore(root)
        store.initialize(state)
        history = HistoryManager(state, max_depth=max_undo_depth)
        return cls(root=root, store=store, history=history, recovery=None)

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        max_undo_depth: int = 32,
    ) -> "CadApplication":
        root = Path(root)
        store = PersistentStore(root)
        recovery = store.open()
        manifest = store.manifest()

        def load_verified(version: int) -> GeometryState:
            loaded = store.load_state(version)
            loaded.state.verify_display()
            return loaded.state

        current = load_verified(recovery.recovered_version)
        operations = store.load_all_operations()
        history = HistoryManager(
            current,
            max_depth=max_undo_depth,
            state_loader=load_verified,
            operations=operations.values(),
            undo_versions=manifest.get("undo_stack", []),
            redo_versions=manifest.get("redo_stack", []),
        )
        store.mark_active()
        app = cls(root=root, store=store, history=history, recovery=recovery)
        return app

    def _load_state(self, version: int) -> GeometryState:
        if version in self.history.states:
            return self.history.states[version]
        loaded = self.store.load_state(version)
        loaded.state.verify_display()
        self.history.states[version] = loaded.state
        return loaded.state

    def edit_control_point(
        self,
        patch_key: str,
        index_u: int,
        index_v: int,
        value: Sequence[float],
    ) -> GeometryState:
        if self._closed:
            raise DomainError(ErrorCode.GEOM_STALE, "application is closed")
        if len(value) != 3:
            raise DomainError(ErrorCode.SCHEMA_VERSION, "control point value must have three coordinates")
        patch = self.current.model.patch(patch_key)
        patch.validate_control_index(index_u, index_v)
        before = tuple(float(item) for item in patch.control_points[index_u, index_v])
        after = tuple(float(item) for item in value)
        new_version = self.store.next_version()
        state, _, _ = derive_edited_state(
            self.current,
            new_version=new_version,
            operation_version=new_version,
            patch_key=patch_key,
            index_u=index_u,
            index_v=index_v,
            new_value=after,
        )
        operation = EditOperation(
            kind="set_control_point",
            op_version=new_version,
            parent_version=self.current.version,
            patch_key=patch_key,
            index_u=index_u,
            index_v=index_v,
            before=before,
            after=after,
            result_state_hash=state.state_hash,
        )
        prospective_undo = [*self.history.undo_stack, self.current.version]
        while len(prospective_undo) > self.history.max_depth:
            prospective_undo.pop(0)
        self.store.persist_commit(state, operation, undo_stack=prospective_undo, redo_stack=[])
        self.history.record_commit(state, operation)
        return state

    def undo(self) -> GeometryState:
        def persist(version: int, undo: list[int], redo: list[int]) -> None:
            target = self._load_state(version)
            self.store.save_session(
                current_version=version,
                current_state_hash=target.state_hash,
                undo_stack=undo,
                redo_stack=redo,
                closed=False,
            )

        return self.history.undo(persist_callback=persist)

    def redo(self) -> GeometryState:
        def persist(version: int, undo: list[int], redo: list[int]) -> None:
            target = self._load_state(version)
            self.store.save_session(
                current_version=version,
                current_state_hash=target.state_hash,
                undo_stack=undo,
                redo_stack=redo,
                closed=False,
            )

        return self.history.redo(persist_callback=persist)

    def switch_to(self, version: int) -> GeometryState:
        def persist(target_version: int, undo: list[int], redo: list[int]) -> None:
            target = self._load_state(target_version)
            self.store.save_session(
                current_version=target_version,
                current_state_hash=target.state_hash,
                undo_stack=undo,
                redo_stack=redo,
                closed=False,
            )

        return self.history.switch_to(version, persist_callback=persist)

    def close(self) -> dict[str, Any]:
        if self._closed:
            return dict(self._closed_report or {})
        self.store.close(
            self.current,
            undo_stack=self.history.undo_stack,
            redo_stack=self.history.redo_stack,
        )
        self._closed = True
        manifest = self.store.manifest()
        self._closed_report = {
            "exit_kind": "controlled",
            "final_version": self.current.version,
            "final_state_hash": self.current.state_hash,
            "closed": bool(manifest.get("closed")),
        }
        return dict(self._closed_report)

    def render_current(self) -> dict[str, RenderBuffer]:
        return self.current.render_all()

    def compare_versions(self, first_version: int, second_version: int) -> dict[str, Any]:
        first = self._load_state(first_version)
        second = self._load_state(second_version)
        first_buffers = first.render_all()
        second_buffers = second.render_all()
        displays: dict[str, Any] = {}
        for mode in DISPLAY_MODES:
            first_rgb = first_buffers[mode].rgb.astype(np.int16)
            second_rgb = second_buffers[mode].rgb.astype(np.int16)
            delta = np.abs(first_rgb - second_rgb)
            displays[mode] = {
                "first_hash": first_buffers[mode].content_hash,
                "second_hash": second_buffers[mode].content_hash,
                "identical": first_buffers[mode].content_hash == second_buffers[mode].content_hash,
                "pixels_changed": int(np.count_nonzero(np.any(delta != 0, axis=2))),
                "max_channel_deviation": int(np.max(delta)) if delta.size else 0,
            }
        return {
            "first_version": first.version,
            "second_version": second.version,
            "first_geometry_hash": first.geometry_hash,
            "second_geometry_hash": second.geometry_hash,
            "geometry_identical": first.geometry_hash == second.geometry_hash,
            "first_normals_hash": first.normals_hash,
            "second_normals_hash": second.normals_hash,
            "normals_identical": first.normals_hash == second.normals_hash,
            "display": displays,
        }

    def reconstruct_patch_at_version(self, version: int, patch_key: str) -> ReconstructionReport:
        """Replay only the selected patch and only each operation's local closure."""

        target_loaded = self.store.load_state(version)
        target_state = target_loaded.state
        target_geometry = target_state.geometries[patch_key]
        self.current.model.patch(patch_key)
        lineage = self.history.lineage_to(version)
        base_state = self._load_state(0)
        patch = base_state.model.patch(patch_key)
        geometry = base_state.geometries[patch_key]
        applied: list[int] = []
        vertex_evaluations = 0
        for operation_version in lineage[1:]:
            operation = self.history.operations.get(operation_version)
            if operation is None:
                raise DomainError(
                    ErrorCode.GEOM_STALE,
                    "historical operation record is missing",
                    version=operation_version,
                )
            if operation.patch_key != patch_key:
                continue
            current_value = tuple(float(item) for item in patch.control_points[operation.index_u, operation.index_v])
            if current_value != operation.before:
                raise DomainError(
                    ErrorCode.GEOM_STALE,
                    "operation before-value does not match replayed history",
                    version=operation_version,
                    expected=list(operation.before),
                    actual=list(current_value),
                )
            patch = patch.with_control_point(operation.index_u, operation.index_v, operation.after)
            closure = geometry.closure_for_control_point(operation.index_u, operation.index_v)
            geometry = geometry.apply_control_edit(
                patch,
                new_version=operation.op_version,
                index_u=operation.index_u,
                index_v=operation.index_v,
                closure=closure,
                operation_version=operation.op_version,
            )
            vertex_evaluations += len(closure.vertices)
            applied.append(operation_version)
        if geometry.version != target_geometry.version:
            geometry = PatchGeometry(
                version=target_geometry.version,
                patch_key=geometry.patch_key,
                control_hash=geometry.control_hash,
                u_values=geometry.u_values,
                v_values=geometry.v_values,
                points=geometry.points,
                du=geometry.du,
                dv=geometry.dv,
                normals=geometry.normals,
                span_u=geometry.span_u,
                span_v=geometry.span_v,
                triangles=geometry.triangles,
                cp_to_vertices=geometry.cp_to_vertices,
                evaluation_record=geometry.evaluation_record,
                recompute_trace=geometry.recompute_trace,
            )
        deviation = 0.0
        if geometry.points.tobytes(order="C") != target_geometry.points.tobytes(order="C"):
            deviation = max(
                deviation,
                float(np.max(np.abs(geometry.points - target_geometry.points))),
            )
        if geometry.normals.tobytes(order="C") != target_geometry.normals.tobytes(order="C"):
            deviation = max(
                deviation,
                float(np.max(np.abs(geometry.normals - target_geometry.normals))),
            )
        return ReconstructionReport(
            target_version=int(version),
            patch_key=patch_key,
            applied_operations=tuple(applied),
            vertex_evaluations=vertex_evaluations,
            total_patch_vertices=target_geometry.vertex_count,
            patches_rebuilt=1,
            full_model_rebuild=False,
            expected_geometry_hash=target_geometry.geometry_hash,
            reconstructed_geometry_hash=geometry.geometry_hash,
            expected_normals_hash=target_geometry.normals_hash,
            reconstructed_normals_hash=geometry.normals_hash,
            max_abs_deviation=deviation,
        )

    def stage_crash_probe(
        self,
        patch_key: str,
        index_u: int,
        index_v: int,
        value: Sequence[float],
    ) -> dict[str, Any]:
        """Build a prospective commit in staging without updating the manifest."""

        patch = self.current.model.patch(patch_key)
        patch.validate_control_index(index_u, index_v)
        before = tuple(float(item) for item in patch.control_points[index_u, index_v])
        after = tuple(float(item) for item in value)
        prospective_version = self.store.next_version()
        state, _, _ = derive_edited_state(
            self.current,
            new_version=prospective_version,
            operation_version=prospective_version,
            patch_key=patch_key,
            index_u=index_u,
            index_v=index_v,
            new_value=after,
        )
        operation = EditOperation(
            kind="set_control_point",
            op_version=prospective_version,
            parent_version=self.current.version,
            patch_key=patch_key,
            index_u=index_u,
            index_v=index_v,
            before=before,
            after=after,
            result_state_hash=state.state_hash,
        )
        staging_path = self.store.stage_uncommitted_probe(state, operation)
        return {
            "staging_path": staging_path,
            "prospective_version": prospective_version,
            "last_committed_version": self.current.version,
            "last_committed_state_hash": self.current.state_hash,
        }

    def export_ppm(self, version: int, mode: str, path: Path) -> Path:
        state = self._load_state(version)
        buffer = state.render_all()[mode]
        from .render import write_ppm

        write_ppm(Path(path), buffer)
        return Path(path)

    def state_summary(self, version: int | None = None) -> dict[str, Any]:
        state = self.current if version is None else self._load_state(version)
        return {
            "version": state.version,
            "parent_version": state.parent_version,
            "state_hash": state.state_hash,
            "geometry_hash": state.geometry_hash,
            "normals_hash": state.normals_hash,
            "iso_hash": state.hashes.iso_hash,
            "tessellation_hash": state.hashes.tessellation_hash,
            "display_hashes": state.display_hashes,
            "model_hash": state.model.content_hash(),
            "undo_depth": len(self.history.undo_stack),
            "redo_depth": len(self.history.redo_stack),
        }
