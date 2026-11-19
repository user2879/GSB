from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PySide6.QtCore import QPointF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .engine import CancellationToken, RunEngine, RunResult
from .errors import BusinessError, ErrorCode
from .mesh import MeshState
from .persistence import StateStore, read_json_strict
from .sample import write_sample
from .schema import MeshInput, RepairScript, load_mesh_file, load_repair_script, make_version_id


@dataclass(slots=True)
class StepView:
    step: int
    operator_type: str
    valid: bool
    reason: str | None
    payload: dict[str, Any] | None
    mesh: MeshState | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    error_lookup: dict[int, dict[str, Any]] = field(default_factory=dict)
    business_codes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        names = {
            "input": "Input mesh",
            "hole_triangulation": "Hole triangulation",
            "remove_degenerate_faces": "Remove degenerate faces",
            "reorient_normals": "Reorient normals",
            "laplacian_smooth": "Laplacian smoothing",
        }
        name = names.get(self.operator_type, self.operator_type)
        return f"{self.step:02d}  {name}"


@dataclass(slots=True)
class WorkspaceView:
    steps: list[StepView]
    report: dict[str, Any] | None
    inspection: dict[str, Any] | None
    expected_version: str | None
    manifest_version: str | None
    load_codes: list[str]

    @property
    def valid_steps(self) -> list[StepView]:
        return [step for step in self.steps if step.valid and step.mesh is not None]


def _mark_invalid(step: StepView, reason: str) -> StepView:
    return StepView(
        step=step.step,
        operator_type=step.operator_type,
        valid=False,
        reason=reason,
        payload=None,
    )


def load_workspace_view(
    mesh_path: str,
    script_path: str,
    state_dir: str,
) -> WorkspaceView:
    load_codes: list[str] = []
    expected_version: str | None = None
    script: RepairScript | None = None
    try:
        mesh = load_mesh_file(mesh_path)
        script = load_repair_script(script_path)
        expected_version = make_version_id(mesh.mesh_digest, script.script_digest)
    except BusinessError as error:
        load_codes.append(error.code)

    inspection_dict: dict[str, Any] | None = None
    manifest_version: str | None = None
    steps: list[StepView] = []
    try:
        inspection = StateStore(state_dir).inspect()
        inspection_dict = inspection.to_dict()
        manifest_version = (
            str(inspection.manifest.get("version_id")) if inspection.manifest else None
        )
        for record in inspection.snapshots:
            operator_type = "unknown"
            reason = record.reason
            payload = record.payload
            if payload is not None:
                operator_type = str(payload.get("operator_type", operator_type))
            step = StepView(
                step=record.step,
                operator_type=operator_type,
                valid=record.valid,
                reason=reason,
                payload=payload,
            )
            if step.valid and payload is not None:
                if expected_version is None:
                    step = _mark_invalid(step, "cannot derive a version digest from current mesh/script")
                elif payload.get("version_id") != expected_version:
                    step = _mark_invalid(step, "snapshot version_id differs from current mesh/script digest")
                elif script is not None and int(payload.get("mesh", {}).get("scale", -1)) != script.fixed_scale:
                    step = _mark_invalid(step, "snapshot fixed-point scale differs from current script")
                else:
                    try:
                        step.mesh = MeshState.from_payload(payload["mesh"])
                        step.errors = list(payload.get("error", {}).get("vertex_errors", []))
                        step.error_lookup = {
                            int(row["vertex"]): row for row in step.errors
                        }
                        step.business_codes = [str(code) for code in payload.get("business_codes", [])]
                    except (KeyError, TypeError, ValueError) as exc:
                        step = _mark_invalid(step, f"snapshot mesh arrays are malformed: {exc}")
            if not step.valid:
                load_codes.append(ErrorCode.E_SNAPSHOT_INVALID)
            steps.append(step)
    except BusinessError as error:
        load_codes.append(error.code)

    return WorkspaceView(
        steps=steps,
        report=None,
        inspection=inspection_dict,
        expected_version=expected_version,
        manifest_version=manifest_version,
        load_codes=sorted(set(load_codes)),
    )


class RunWorker(QThread):
    completed = Signal(object)
    failed = Signal(object)
    progressed = Signal(int, int, str)

    def __init__(
        self,
        mesh: MeshInput,
        script: RepairScript,
        state_dir: str,
        output_dir: str,
        restart: bool,
    ) -> None:
        super().__init__()
        self.mesh = mesh
        self.script = script
        self.state_dir = state_dir
        self.output_dir = output_dir
        self.restart = restart
        self.token = CancellationToken()

    def run(self) -> None:
        try:
            engine = RunEngine(self.mesh, self.script, self.state_dir, self.output_dir)
            result = engine.run(
                self.token,
                restart=self.restart,
                progress=lambda step, total, label: self.progressed.emit(step, total, label),
            )
            self.completed.emit(result)
        except BusinessError as error:
            self.failed.emit(error.to_dict())
        except Exception as exc:  # pragma: no cover - defensive GUI boundary
            self.failed.emit(
                {
                    "ok": False,
                    "error": {"code": ErrorCode.E_INTERNAL, "message": str(exc), "details": {}},
                }
            )


def _error_color(ratio: float) -> QColor:
    if not math.isfinite(ratio) or ratio <= 0.0:
        return QColor("#4f7cac")
    if ratio < 0.33:
        return QColor("#2aa198")
    if ratio < 0.66:
        return QColor("#d6b300")
    if ratio < 1.0:
        return QColor("#e8832c")
    return QColor("#d83b3b")


class MeshCanvas(QWidget):
    vertex_selected = Signal(int)

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.mesh: MeshState | None = None
        self.error_lookup: dict[int, dict[str, Any]] = {}
        self.tolerance = 1.0e-7
        self.view_mode = "iso"
        self.source_vertices: set[int] = set()
        self.affected_vertices: set[int] = set()
        self.selected_vertex: int | None = None
        self._projected: dict[int, QPointF] = {}
        self.setMinimumSize(280, 250)
        self.setMouseTracking(True)

    def set_data(
        self,
        mesh: MeshState | None,
        error_lookup: dict[int, dict[str, Any]],
        tolerance: float,
    ) -> None:
        self.mesh = mesh
        self.error_lookup = error_lookup
        self.tolerance = max(float(tolerance), 1.0e-30)
        self.update()

    def set_view_mode(self, mode: str) -> None:
        self.view_mode = mode
        self.update()

    def set_highlights(
        self,
        sources: Sequence[int],
        affected: Sequence[int],
        selected_vertex: int | None = None,
    ) -> None:
        self.source_vertices = set(int(value) for value in sources)
        self.affected_vertices = set(int(value) for value in affected)
        self.selected_vertex = selected_vertex
        self.update()

    def _raw_project(self, coordinates: np.ndarray) -> np.ndarray:
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        if self.view_mode == "top":
            return np.column_stack((x, z))
        if self.view_mode == "front":
            return np.column_stack((x, y))
        cosine = math.cos(math.radians(30.0))
        sine = math.sin(math.radians(30.0))
        return np.column_stack(((x - z) * cosine, y + (x + z) * sine))

    def _project(self) -> tuple[np.ndarray, float, float]:
        if self.mesh is None:
            return np.zeros((0, 2)), 1.0, 1.0
        coordinates = self.mesh.fixed.astype(np.float64) / float(self.mesh.scale)
        raw = self._raw_project(coordinates)
        incident = np.unique(self.mesh.faces.reshape(-1)) if self.mesh.faces.size else np.arange(len(raw))
        fitted = raw[incident] if len(incident) else raw
        minimum = np.min(fitted, axis=0)
        maximum = np.max(fitted, axis=0)
        span = np.maximum(maximum - minimum, 1.0e-12)
        margin = 22.0
        width = max(1.0, self.width() - 2.0 * margin)
        height = max(1.0, self.height() - 2.0 * margin)
        scale = min(width / float(span[0]), height / float(span[1]))
        center = (minimum + maximum) * 0.5
        pixel_center = np.asarray([self.width() * 0.5, self.height() * 0.5])
        pixel = (raw - center) * scale + pixel_center
        pixel[:, 1] = self.height() - pixel[:, 1]
        return pixel, float(scale), float(minimum[0])

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#101820"))
        painter.setPen(QColor("#d9e2ec"))
        title_font = QFont(self.font())
        title_font.setPointSize(max(10, title_font.pointSize() + 1))
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(14, 24, self.title)
        if self.mesh is None:
            painter.setPen(QColor("#7f8c9a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No persisted step selected")
            painter.end()
            return
        projected, _scale, _minimum = self._project()
        self._projected = {
            index: QPointF(float(point[0]), float(point[1]))
            for index, point in enumerate(projected)
        }
        depth_axis = 1
        face_depth = np.mean(projected[self.mesh.faces, depth_axis], axis=1)
        face_order = np.argsort(face_depth, kind="stable")
        for face_index in face_order:
            face = self.mesh.faces[int(face_index)]
            path = QPainterPath(self._projected[int(face[0])])
            path.lineTo(self._projected[int(face[1])])
            path.lineTo(self._projected[int(face[2])])
            path.closeSubpath()
            affected = any(int(vertex) in self.affected_vertices for vertex in face)
            fill = QColor(42, 66, 82, 95) if affected else QColor(31, 48, 61, 63)
            painter.fillPath(path, QBrush(fill))
            painter.setPen(QPen(QColor("#456273"), 0.8))
            painter.drawPath(path)

        for vertex in range(self.mesh.vertex_count):
            point = self._projected.get(vertex)
            if point is None:
                continue
            error = float(self.error_lookup.get(vertex, {}).get("max_abs_error", 0.0))
            color = _error_color(error / self.tolerance)
            radius = 3.3 if vertex not in self.source_vertices else 5.0
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#0b1116"), 0.7))
            painter.drawEllipse(point, radius, radius)
            if vertex in self.source_vertices:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#ff56c2"), 1.6))
                painter.drawEllipse(point, 7.5, 7.5)
            if vertex == self.selected_vertex:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#ffffff"), 2.0))
                painter.drawEllipse(point, 10.5, 10.5)
        self._draw_legend(painter)
        painter.end()

    def _draw_legend(self, painter: QPainter) -> None:
        x = 14
        y = self.height() - 18
        painter.setFont(QFont(self.font().family(), 8))
        for index, (label, color) in enumerate(
            (("≤0", "#4f7cac"), ("low", "#2aa198"), ("mid", "#d6b300"), ("high", "#e8832c"), ("over", "#d83b3b"))
        ):
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(x + index * 49, y - 9, 10, 10)
            painter.setPen(QColor("#bcccdc"))
            painter.drawText(x + index * 49 + 14, y, label)
        painter.setPen(QColor("#8aa0b3"))
        painter.drawText(
            max(14, self.width() - 205),
            24,
            f"{self.view_mode.upper()} projection",
        )

    def mousePressEvent(self, event: Any) -> None:
        if self.mesh is None or not self._projected:
            return
        position = event.position()
        nearest: tuple[float, int] | None = None
        for vertex, point in self._projected.items():
            distance = (point.x() - position.x()) ** 2 + (point.y() - position.y()) ** 2
            if nearest is None or distance < nearest[0]:
                nearest = (distance, vertex)
        if nearest is not None and nearest[0] <= 12.0**2:
            self.selected_vertex = nearest[1]
            self.vertex_selected.emit(nearest[1])
            self.update()


class MainWindow(QMainWindow):
    def __init__(
        self,
        mesh_path: str | None = None,
        script_path: str | None = None,
        state_dir: str | None = None,
        output_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Mesh Audit Workbench · Fixed-Point Repair Diagnostics")
        self.resize(1440, 900)
        self.mesh_path = str(mesh_path or "samples/sample_defective.mesh.json")
        self.script_path = str(script_path or "samples/sample_repair_script.json")
        self.state_dir = str(state_dir or "state/sample")
        self.output_dir = str(output_dir or "output/sample")
        self.workspace = WorkspaceView([], None, None, None, None, [])
        self.current_step_index = 0
        self.current_source_vertex: int | None = None
        self.worker: RunWorker | None = None
        self.transient_codes: set[str] = set()
        self._build_ui()
        self._update_path_label()
        self.refresh_view()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Workbench")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        open_mesh = QAction("Open mesh", self)
        open_mesh.triggered.connect(self._choose_mesh)
        toolbar.addAction(open_mesh)
        open_script = QAction("Open script", self)
        open_script.triggered.connect(self._choose_script)
        toolbar.addAction(open_script)
        choose_state = QAction("State dir", self)
        choose_state.triggered.connect(self._choose_state)
        toolbar.addAction(choose_state)
        choose_output = QAction("Output dir", self)
        choose_output.triggered.connect(self._choose_output)
        toolbar.addAction(choose_output)
        generate = QAction("Generate sample", self)
        generate.triggered.connect(self._generate_sample)
        toolbar.addAction(generate)
        toolbar.addSeparator()
        self.run_action = QAction("Run / Resume", self)
        self.run_action.setShortcut(QKeySequence("Ctrl+R"))
        self.run_action.triggered.connect(lambda: self._start_run(False))
        toolbar.addAction(self.run_action)
        self.restart_action = QAction("Archive & rerun", self)
        self.restart_action.triggered.connect(lambda: self._start_run(True))
        toolbar.addAction(self.restart_action)
        self.stop_action = QAction("Stop at boundary", self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self._stop_run)
        toolbar.addAction(self.stop_action)
        refresh = QAction("Refresh snapshots", self)
        refresh.triggered.connect(self.refresh_view)
        toolbar.addAction(refresh)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Projection"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["iso", "top", "front"])
        self.view_combo.currentTextChanged.connect(self._change_view)
        toolbar.addWidget(self.view_combo)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 6, 8, 8)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setFont(QFont(self.font().family(), 9))
        layout.addWidget(self.path_label)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        canvas_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_canvas = MeshCanvas("Input mesh (persisted step 00)")
        self.current_canvas = MeshCanvas("Selected repair step")
        self.original_canvas.vertex_selected.connect(self._select_vertex)
        self.current_canvas.vertex_selected.connect(self._select_vertex)
        canvas_splitter.addWidget(self.original_canvas)
        canvas_splitter.addWidget(self.current_canvas)
        canvas_splitter.setSizes([550, 550])
        left_layout.addWidget(canvas_splitter, 1)
        main_splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.timeline = QListWidget()
        self.timeline.currentRowChanged.connect(self._timeline_changed)
        self.tabs.addTab(self.timeline, "Operator timeline")

        self.diagnostics_table = QTableWidget()
        self.diagnostics_table.setColumnCount(3)
        self.diagnostics_table.setHorizontalHeaderLabels(["Metric", "Input", "Current / final"])
        self.diagnostics_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.diagnostics_table, "Topology")

        self.propagation_table = QTableWidget()
        self.propagation_table.setColumnCount(6)
        self.propagation_table.setHorizontalHeaderLabels(
            ["Source", "Layer", "Vertices", "Max error", "Mean error", "Conclusion"]
        )
        self.propagation_table.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.propagation_table, "Propagation layers")

        self.vertex_table = QTableWidget()
        self.vertex_table.setColumnCount(5)
        self.vertex_table.setHorizontalHeaderLabels(
            ["Vertex", "Fixed X/Y/Z", "float64 max error", "Over tol.", "Direct source"]
        )
        self.vertex_table.horizontalHeader().setStretchLastSection(True)
        self.vertex_table.cellClicked.connect(self._vertex_table_clicked)
        self.tabs.addTab(self.vertex_table, "Vertex errors")

        self.codes_list = QListWidget()
        self.tabs.addTab(self.codes_list, "Business codes")

        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("monospace", 9))
        self.tabs.addTab(self.report_text, "Report summary")
        right_layout.addWidget(self.tabs, 1)
        main_splitter.addWidget(right)
        main_splitter.setSizes([920, 500])
        layout.addWidget(main_splitter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _update_path_label(self) -> None:
        self.path_label.setText(
            f"<b>Mesh</b> {self.mesh_path} &nbsp; | &nbsp; <b>Script</b> {self.script_path}<br>"
            f"<b>State</b> {self.state_dir} &nbsp; | &nbsp; <b>Output</b> {self.output_dir}"
        )

    def _choose_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select triangle mesh", str(Path(self.mesh_path).parent), "Mesh (*.json *.obj);;All files (*)"
        )
        if path:
            self.mesh_path = path
            self._update_path_label()
            self.refresh_view()

    def _choose_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select repair script", str(Path(self.script_path).parent), "JSON (*.json);;All files (*)"
        )
        if path:
            self.script_path = path
            self._update_path_label()
            self.refresh_view()

    def _choose_state(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select state directory", self.state_dir)
        if path:
            self.state_dir = path
            self._update_path_label()
            self.refresh_view()

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir)
        if path:
            self.output_dir = path
            self._update_path_label()

    def _generate_sample(self) -> None:
        mesh_path, script_path = write_sample("samples")
        self.mesh_path = str(mesh_path)
        self.script_path = str(script_path)
        self.state_dir = "state/sample"
        self.output_dir = "output/sample"
        self._update_path_label()
        self.refresh_view()
        self.statusBar().showMessage("Deterministic sample generated; click Run / Resume")

    def refresh_view(self) -> None:
        self.workspace = load_workspace_view(self.mesh_path, self.script_path, self.state_dir)
        report_path = Path(self.output_dir) / "report.json"
        if report_path.exists():
            try:
                candidate = read_json_strict(report_path)
                if isinstance(candidate, dict):
                    if (
                        self.workspace.expected_version is None
                        or candidate.get("version_id") == self.workspace.expected_version
                    ):
                        self.workspace.report = candidate
                    else:
                        self.workspace.report = None
                        self.workspace.load_codes.append(ErrorCode.E_SNAPSHOT_INVALID)
            except BusinessError:
                self.workspace.load_codes.append(ErrorCode.E_SNAPSHOT_INVALID)
        self._render_workspace()

    def _render_workspace(self) -> None:
        self.timeline.blockSignals(True)
        self.timeline.clear()
        for index, step in enumerate(self.workspace.steps):
            marker = "" if step.valid else "  [INVALID]"
            item = QListWidgetItem(step.label + marker)
            item.setData(Qt.ItemDataRole.UserRole, index)
            if not step.valid:
                item.setForeground(QColor("#d83b3b"))
                item.setToolTip(step.reason or "invalid snapshot")
            self.timeline.addItem(item)
        self.timeline.blockSignals(False)
        selected = next(
            (index for index, step in enumerate(self.workspace.steps) if step.valid),
            0,
        )
        if self.workspace.steps:
            valid_indices = [index for index, step in enumerate(self.workspace.steps) if step.valid]
            if valid_indices:
                selected = valid_indices[-1]
            self.timeline.setCurrentRow(selected)
        original = self._step(0)
        if original is not None and original.valid:
            self.original_canvas.set_data(
                original.mesh,
                original.error_lookup,
                float(original.payload.get("error", {}).get("tolerance", 1.0e-7))
                if original.payload
                else 1.0e-7,
            )
        else:
            self.original_canvas.set_data(None, {}, 1.0e-7)
        self._timeline_changed(self.timeline.currentRow())
        self._render_diagnostics()
        self._render_codes()
        self._render_report()
        if self.worker is None:
            if self.workspace.steps:
                self.statusBar().showMessage(
                    f"Loaded {len(self.workspace.steps)} persisted steps, "
                    f"{sum(1 for step in self.workspace.steps if not step.valid)} invalid"
                )
            else:
                self.statusBar().showMessage("No persisted steps yet; views render only from snapshots")

    def _step(self, step: int) -> StepView | None:
        for candidate in self.workspace.steps:
            if candidate.step == step:
                return candidate
        return None

    def _timeline_changed(self, row: int) -> None:
        if row < 0 or row >= len(self.workspace.steps):
            return
        step = self.workspace.steps[row]
        self.current_step_index = row
        if not step.valid or step.mesh is None or step.payload is None:
            self.current_canvas.set_data(None, {}, 1.0e-7)
            self.current_canvas.set_highlights([], [], None)
            self.statusBar().showMessage(f"step {step.step:02d} invalid: {step.reason or 'digest mismatch'}")
        else:
            tolerance = float(step.payload.get("error", {}).get("tolerance", 1.0e-7))
            self.current_canvas.set_data(step.mesh, step.error_lookup, tolerance)
            sources = step.payload.get("propagation", {}).get("source_vertices", [])
            affected = step.payload.get("propagation", {}).get("affected_vertices", [])
            self.current_canvas.set_highlights(sources, affected, self.current_source_vertex)
        self._render_diagnostics()
        self._render_propagation()
        self._render_vertex_table()

    def _change_view(self, mode: str) -> None:
        self.original_canvas.set_view_mode(mode)
        self.current_canvas.set_view_mode(mode)

    def _render_diagnostics(self) -> None:
        initial = self._step(0)
        current = self._step(self._selected_step_number())
        if current is None:
            current = self.workspace.valid_steps[-1] if self.workspace.valid_steps else None
        initial_diag = initial.payload.get("diagnostics", {}) if initial and initial.payload else {}
        current_diag = current.payload.get("diagnostics", {}) if current and current.payload else {}
        keys = [
            "vertex_count",
            "face_count",
            "edge_count",
            "nonmanifold_edge_count",
            "boundary_edge_count",
            "boundary_loop_count",
            "open_boundary_chain_count",
            "degenerate_face_count",
            "isolated_vertex_count",
            "euler_characteristic",
            "rescan_matches",
        ]
        self.diagnostics_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            self.diagnostics_table.setItem(row, 0, QTableWidgetItem(key))
            self.diagnostics_table.setItem(
                row, 1, QTableWidgetItem(str(initial_diag.get(key, "-")))
            )
            self.diagnostics_table.setItem(
                row, 2, QTableWidgetItem(str(current_diag.get(key, "-")))
            )
        self.diagnostics_table.resizeColumnsToContents()

    def _selected_step_number(self) -> int:
        if 0 <= self.current_step_index < len(self.workspace.steps):
            return self.workspace.steps[self.current_step_index].step
        return 0

    def _render_propagation(self) -> None:
        step = self._step(self._selected_step_number())
        self.propagation_table.setRowCount(0)
        if step is None or not step.valid or step.payload is None:
            return
        traces = list(step.payload.get("propagation_sources", []))
        if self.current_source_vertex is not None:
            selected = [
                trace
                for trace in traces
                if int(trace.get("source_vertex", -1)) == self.current_source_vertex
            ]
            if selected:
                traces = selected
        rows: list[tuple[int, int, int, float, float, str]] = []
        for trace in traces:
            source = int(trace.get("source_vertex", -1))
            for layer in trace.get("layers", []):
                rows.append(
                    (
                        source,
                        int(layer.get("layer", 0)),
                        int(layer.get("vertex_count", 0)),
                        float(layer.get("max_abs_error", 0.0)),
                        float(layer.get("mean_abs_error", 0.0)),
                        str(trace.get("decay_conclusion", "")),
                    )
                )
        rows.sort(key=lambda row: (row[0], row[1]))
        self.propagation_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                text = f"{value:.3e}" if isinstance(value, float) else str(value)
                self.propagation_table.setItem(row_index, column, QTableWidgetItem(text))
        self.propagation_table.resizeColumnsToContents()

    def _render_vertex_table(self) -> None:
        step = self._step(self._selected_step_number())
        self.vertex_table.setRowCount(0)
        if step is None or not step.valid or step.payload is None:
            return
        sources = set(int(value) for value in step.payload.get("error", {}).get("direct_source_vertices", []))
        rows = sorted(step.errors, key=lambda row: int(row.get("vertex", 0)))
        self.vertex_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            vertex = int(row["vertex"])
            fixed = row.get("fixed", [0, 0, 0])
            self.vertex_table.setItem(row_index, 0, QTableWidgetItem(str(vertex)))
            self.vertex_table.setItem(
                row_index, 1, QTableWidgetItem(", ".join(str(value) for value in fixed))
            )
            self.vertex_table.setItem(
                row_index, 2, QTableWidgetItem(f"{float(row.get('max_abs_error', 0.0)):.6e}")
            )
            self.vertex_table.setItem(
                row_index,
                3,
                QTableWidgetItem("YES" if row.get("exceeds_tolerance") else "no"),
            )
            self.vertex_table.setItem(
                row_index, 4, QTableWidgetItem("YES" if vertex in sources else "")
            )
        self.vertex_table.resizeColumnsToContents()

    def _vertex_table_clicked(self, row: int, _column: int) -> None:
        item = self.vertex_table.item(row, 0)
        if item is not None:
            try:
                self._select_vertex(int(item.text()))
            except ValueError:
                pass

    def _select_vertex(self, vertex: int) -> None:
        self.current_source_vertex = vertex
        step = self._step(self._selected_step_number())
        if step is None or not step.valid or step.payload is None:
            return
        traces = list(step.payload.get("propagation_sources", []))
        selected_trace = next(
            (trace for trace in traces if int(trace.get("source_vertex", -1)) == vertex),
            None,
        )
        if selected_trace is None:
            selected_trace = next(
                (
                    trace
                    for trace in traces
                    if vertex in {int(value) for value in trace.get("affected_vertices", [])}
                ),
                None,
            )
        if selected_trace is not None:
            source = int(selected_trace.get("source_vertex", vertex))
            affected = [int(value) for value in selected_trace.get("affected_vertices", [])]
            self.current_source_vertex = source
            self.current_canvas.set_highlights([source], affected, vertex)
            self.original_canvas.set_highlights([source], affected, vertex)
        else:
            self.current_canvas.set_highlights([], [], vertex)
        self._render_propagation()
        self.statusBar().showMessage(
            f"Selected vertex {vertex}; propagation source {self.current_source_vertex}"
        )

    def _render_codes(self) -> None:
        self.codes_list.clear()
        codes: set[str] = set(self.workspace.load_codes) | set(self.transient_codes)
        for step in self.workspace.steps:
            codes.update(step.business_codes)
        if self.workspace.report:
            codes.update(str(code) for code in self.workspace.report.get("business_codes", []))
        for code in sorted(codes):
            item = QListWidgetItem(code)
            if code.startswith("E_") or "INVALID" in code or "MISMATCH" in code:
                item.setForeground(QColor("#d83b3b"))
            elif "TOLERANCE" in code:
                item.setForeground(QColor("#e8832c"))
            self.codes_list.addItem(item)

    def _render_report(self) -> None:
        if not self.workspace.report:
            self.report_text.setPlainText(
                "No valid report.json yet. A completed run writes the authoritative summary; views never use in-memory run results."
            )
            return
        report = self.workspace.report
        summary = report.get("summary", {})
        text = {
            "version_id": report.get("version_id"),
            "verdict": summary.get("fixed_point_verdict"),
            "first_tolerance_exceedance": summary.get("first_tolerance_exceedance"),
            "resume": report.get("resume"),
            "business_codes": report.get("business_codes"),
            "initial_diagnostics": {
                key: summary.get("initial_diagnostics", {}).get(key)
                for key in (
                    "vertex_count",
                    "face_count",
                    "nonmanifold_edge_count",
                    "boundary_loop_count",
                    "degenerate_face_count",
                    "isolated_vertex_count",
                    "euler_characteristic",
                )
            },
            "final_diagnostics": {
                key: summary.get("final_diagnostics", {}).get(key)
                for key in (
                    "vertex_count",
                    "face_count",
                    "nonmanifold_edge_count",
                    "boundary_loop_count",
                    "degenerate_face_count",
                    "isolated_vertex_count",
                    "euler_characteristic",
                )
            },
        }
        self.report_text.setPlainText(json.dumps(text, indent=2, ensure_ascii=False))

    def _start_run(self, restart: bool) -> None:
        if self.worker is not None:
            return
        try:
            mesh = load_mesh_file(self.mesh_path)
            script = load_repair_script(self.script_path)
        except BusinessError as error:
            self.transient_codes.add(error.code)
            self._render_codes()
            QMessageBox.critical(self, error.code, error.to_json())
            return
        self.worker = RunWorker(
            mesh,
            script,
            self.state_dir,
            self.output_dir,
            restart,
        )
        self.worker.completed.connect(self._run_completed)
        self.worker.failed.connect(self._run_failed)
        self.worker.progressed.connect(self._run_progress)
        self.worker.finished.connect(self._worker_finished)
        self.run_action.setEnabled(False)
        self.restart_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.statusBar().showMessage("Running; stop requests take effect only at an operator boundary")
        self.worker.start()

    def _stop_run(self) -> None:
        if self.worker is not None:
            self.worker.token.request(None, "UI_STOP")
            self.stop_action.setEnabled(False)
            self.statusBar().showMessage("Stop requested: waiting for the current operator to persist")

    def _run_progress(self, step: int, total: int, label: str) -> None:
        self.statusBar().showMessage(f"Step {step}/{total} · {label}")

    def _run_completed(self, result: RunResult) -> None:
        if result.exit_code == 0:
            self.transient_codes.clear()
            self.statusBar().showMessage(f"Complete: {result.report_path}")
        else:
            self.statusBar().showMessage(
                f"Stopped at operator boundary with exit code {result.exit_code}; run again to resume"
            )
        self.refresh_view()

    def _run_failed(self, payload: dict[str, Any]) -> None:
        error = payload.get("error", {})
        self.transient_codes.add(str(error.get("code", ErrorCode.E_INTERNAL)))
        self._render_codes()
        QMessageBox.critical(
            self,
            str(error.get("code", ErrorCode.E_INTERNAL)),
            json.dumps(payload, indent=2, ensure_ascii=False),
        )
        self.statusBar().showMessage(f"Failed: {error.get('code')} {error.get('message')}")
        self.refresh_view()

    def _worker_finished(self) -> None:
        self.worker = None
        self.run_action.setEnabled(True)
        self.restart_action.setEnabled(True)
        self.stop_action.setEnabled(False)


def main(
    mesh_path: str | None = None,
    script_path: str | None = None,
    state_dir: str | None = None,
    output_dir: str | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(mesh_path, script_path, state_dir, output_dir)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
