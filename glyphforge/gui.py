"""PyQt5 read-only diagnostic view plus a thin external editor controller."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import Any

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .application import GlyphApplication
from .diagnostics import DiagnosticService, DiagnosticSnapshot, DiagnosticViewState
from .errors import GlyphForgeError
from .model import SUPPORTED_SIZES, Point
from .samples import sample_glyphs


def _grayscale_image(pixels: bytes, width: int, height: int) -> QImage:
    image = QImage(pixels, width, height, width, QImage.Format_Grayscale8).copy()
    return image


class BitmapCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._caption = "No bitmap"
        self.setMinimumSize(QSize(280, 280))

    def set_bitmap(
        self, pixels: bytes | None, width: int, height: int, caption: str
    ) -> None:
        self._image = (
            _grayscale_image(pixels, width, height) if pixels is not None else None
        )
        self._caption = caption
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#15191f"))
        painter.setPen(QColor("#9fb3c8"))
        painter.drawText(
            QRect(8, 8, self.width() - 16, 20), Qt.AlignLeft, self._caption
        )
        if self._image is None:
            painter.drawText(self.rect(), Qt.AlignCenter, "No rendered bitmap")
            return
        available = self.rect().adjusted(12, 32, -12, -12)
        scaled = self._image.size().scaled(available.size(), Qt.KeepAspectRatio)
        target = QRect(
            available.x() + (available.width() - scaled.width()) // 2,
            available.y() + (available.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )
        painter.fillRect(target, QColor("#f1f3f5"))
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(target, self._image)


class AtlasPageCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._page = None
        self.setMinimumSize(QSize(300, 300))

    def set_page(self, page: Any) -> None:
        self._page = page
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101419"))
        if self._page is None:
            painter.setPen(QColor("#9fb3c8"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No atlas page")
            return
        page = self._page
        margin = 8
        side = min(self.width(), self.height()) - margin * 2 - 20
        target = QRect(margin, margin + 20, side, side)
        image = _grayscale_image(page.pixels, page.width, page.height)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(target, image)
        painter.setPen(QColor("#dbe7f3"))
        painter.drawText(
            QRect(margin, margin, self.width(), 18),
            Qt.AlignLeft,
            f"Atlas page {page.index}",
        )
        sx = side / float(page.width)
        sy = side / float(page.height)
        for region in page.regions:
            rect = QRect(
                target.x() + int(region.x * sx),
                target.y() + int(region.y * sy),
                max(1, int(region.width * sx)),
                max(1, int(region.height * sy)),
            )
            color = QColor("#ffb454") if region.reused_slot else QColor("#5ee0a0")
            painter.setPen(QPen(color, 1))
            painter.drawRect(rect)
        painter.setPen(QPen(QColor("#5aa9ff"), 1, Qt.DashLine))
        for x, y, width, height in page.free_rectangles:
            rect = QRect(
                target.x() + int(x * sx),
                target.y() + int(y * sy),
                max(1, int(width * sx)),
                max(1, int(height * sy)),
            )
            painter.drawRect(rect)


class DiagnosticView(QWidget):
    """Read-only widget. It has no application/session reference by design."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = DiagnosticViewState()
        self._snapshot: DiagnosticSnapshot | None = None
        self._bitmap_canvas = BitmapCanvas()
        self._atlas_container = QWidget()
        self._atlas_layout = QVBoxLayout(self._atlas_container)
        self._atlas_layout.setContentsMargins(0, 0, 0, 0)
        self._atlas_pages: list[AtlasPageCanvas] = []
        atlas_scroll = QScrollArea()
        atlas_scroll.setWidgetResizable(True)
        atlas_scroll.setWidget(self._atlas_container)
        self._glyph_selector = QComboBox()
        self._glyph_selector.currentIndexChanged.connect(self._refresh_bitmap)
        self._hash_label = QLabel("No snapshot")
        self._hash_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top = QHBoxLayout()
        top.addWidget(QLabel("Glyph:"))
        top.addWidget(self._glyph_selector)
        top.addWidget(self._hash_label, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addLayout(top)
        left_layout.addWidget(self._bitmap_canvas, 1)
        atlas_box = QGroupBox("Atlas pages (green=new, orange=reused, blue=free)")
        atlas_box_layout = QVBoxLayout(atlas_box)
        atlas_box_layout.addWidget(atlas_scroll)
        self._budget_view = QPlainTextEdit()
        self._budget_view.setReadOnly(True)
        budget_box = QGroupBox("Thread budget and degradation ledger")
        budget_layout = QVBoxLayout(budget_box)
        budget_layout.addWidget(self._budget_view)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(atlas_box)
        splitter.setSizes([420, 620])
        lower_splitter = QSplitter(Qt.Vertical)
        lower_splitter.addWidget(splitter)
        lower_splitter.addWidget(budget_box)
        lower_splitter.setSizes([620, 220])
        layout = QVBoxLayout(self)
        layout.addWidget(lower_splitter)

    @property
    def snapshot_version(self) -> int | None:
        return self._state.current_version

    def apply_snapshot(self, snapshot: DiagnosticSnapshot) -> bool:
        changed = self._state.apply(snapshot)
        if not changed:
            return False
        self._snapshot = snapshot
        current_key = self._glyph_selector.currentData()
        self._glyph_selector.blockSignals(True)
        self._glyph_selector.clear()
        for key, _bitmap in snapshot.bitmaps:
            self._glyph_selector.addItem(key.display(), key)
        if current_key is not None:
            index = self._glyph_selector.findData(current_key)
            if index >= 0:
                self._glyph_selector.setCurrentIndex(index)
        self._glyph_selector.blockSignals(False)
        self._refresh_bitmap()

        while self._atlas_layout.count():
            item = self._atlas_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._atlas_pages = []
        for page in snapshot.atlas_pages:
            canvas = AtlasPageCanvas()
            canvas.set_page(page)
            self._atlas_layout.addWidget(canvas)
            self._atlas_pages.append(canvas)
        self._atlas_layout.addStretch(1)

        decisions = json.loads(snapshot.degrade_decisions_json)
        budget_payload = {
            "version": snapshot.version,
            "thread_budget": snapshot.thread_budget.to_dict()
            if snapshot.thread_budget
            else None,
            "degrade_decisions": decisions,
            "atlas_regions": [region.to_dict() for region in snapshot.atlas_regions],
            "pending_keys": list(snapshot.pending_keys),
            "change_closure": list(snapshot.change_closure),
        }
        self._budget_view.setPlainText(
            json.dumps(budget_payload, ensure_ascii=False, indent=2, sort_keys=True)
        )
        return True

    def _refresh_bitmap(self) -> None:
        key = self._glyph_selector.currentData()
        if self._snapshot is None or key is None:
            self._bitmap_canvas.set_bitmap(None, 1, 1, "No bitmap")
            return
        result = self._snapshot.bitmap_map.get(key)
        if result is None:
            self._bitmap_canvas.set_bitmap(None, 1, 1, key.display())
            self._hash_label.setText("bitmap unavailable")
            return
        self._bitmap_canvas.set_bitmap(
            result.pixels,
            result.width,
            result.height,
            f"{key.display()}  {result.width}x{result.height}  vertices={result.flattened.vertex_count}",
        )
        self._hash_label.setText(f"sha256 {result.content_hash}")


class EditorWindow(QMainWindow):
    """Control panel that delegates edits, then replaces one whole snapshot."""

    def __init__(
        self,
        diagnostics: DiagnosticService,
        command: Callable[[str, dict[str, Any]], DiagnosticSnapshot],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._diagnostics = diagnostics
        self._command = command
        self._view = DiagnosticView()
        self.setCentralWidget(self._view)
        self._status = QLabel("Ready")
        self.statusBar().addWidget(self._status)
        self._build_editor_dock()
        self.apply_snapshot(diagnostics.snapshot())

    def _build_editor_dock(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self._glyph = QComboBox()
        self._glyph.currentIndexChanged.connect(self._sync_glyph_controls)
        self._fill = QComboBox()
        self._fill.addItems(["nonzero", "evenodd"])
        self._size = QComboBox()
        for size in SUPPORTED_SIZES:
            self._size.addItem(str(size), size)
        self._contour = QSpinBox()
        self._contour.setRange(0, 999)
        self._segment = QSpinBox()
        self._segment.setRange(-1, 9999)
        self._role = QComboBox()
        self._role.addItems(["start", "to", "control1", "control2"])
        self._x = QDoubleSpinBox()
        self._y = QDoubleSpinBox()
        for spin in (self._x, self._y):
            spin.setRange(-1_000_000, 1_000_000)
            spin.setDecimals(3)
            spin.setSingleStep(5.0)
        form = QFormLayout()
        form.addRow("Glyph", self._glyph)
        form.addRow("Fill rule", self._fill)
        form.addRow("Size tier", self._size)
        form.addRow("Contour", self._contour)
        form.addRow("Segment (-1 start)", self._segment)
        form.addRow("Point role", self._role)
        form.addRow("X", self._x)
        form.addRow("Y", self._y)
        layout.addLayout(form)
        apply_point = QPushButton("Apply point")
        apply_point.clicked.connect(self._apply_point)
        apply_fill = QPushButton("Apply fill rule")
        apply_fill.clicked.connect(self._apply_fill)
        apply_size = QPushButton("Apply size tier")
        apply_size.clicked.connect(self._apply_size)
        refresh = QPushButton("Refresh snapshot")
        refresh.clicked.connect(self._refresh_only)
        for button in (apply_point, apply_fill, apply_size, refresh):
            layout.addWidget(button)
        layout.addStretch(1)
        dock_widget = QWidget()
        dock_widget.setLayout(QVBoxLayout())
        dock_widget.layout().addWidget(panel)
        from PyQt5.QtWidgets import QDockWidget

        dock = QDockWidget("Editor", self)
        dock.setWidget(dock_widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def apply_snapshot(self, snapshot: DiagnosticSnapshot) -> bool:
        changed = self._view.apply_snapshot(snapshot)
        glyph_ids = snapshot.glyph_ids
        self._glyph.blockSignals(True)
        self._glyph.clear()
        self._glyph.addItems(list(glyph_ids))
        self._glyph.blockSignals(False)
        if snapshot.fill_rules:
            self._fill.setCurrentText(
                dict(snapshot.fill_rules).get(self._glyph.currentText(), "nonzero")
            )
        self._sync_glyph_controls()
        return changed

    def _sync_glyph_controls(self) -> None:
        snapshot = self._view._snapshot
        if snapshot is None or not self._glyph.currentText():
            return
        glyph_id = self._glyph.currentText()
        fill = dict(snapshot.fill_rules).get(glyph_id, "nonzero")
        self._fill.setCurrentText(fill)
        sizes = dict(snapshot.active_sizes).get(glyph_id, ())
        index = self._size.findData(sizes[0]) if sizes else -1
        if index >= 0:
            self._size.setCurrentIndex(index)

    def _run_command(self, name: str, payload: dict[str, Any]) -> None:
        try:
            snapshot = self._command(name, payload)
            self.apply_snapshot(snapshot)
            closure = ", ".join(snapshot.change_closure) or "(none)"
            self._status.setText(f"Committed {name}; recomputed closure: {closure}")
        except GlyphForgeError as exc:
            self._status.setText(str(exc))
            QMessageBox.warning(self, "GlyphForge", str(exc))

    def _apply_point(self) -> None:
        self._run_command(
            "point",
            {
                "glyph_id": self._glyph.currentText(),
                "contour_index": self._contour.value(),
                "segment_index": self._segment.value(),
                "role": self._role.currentText(),
                "x": self._x.value(),
                "y": self._y.value(),
            },
        )

    def _apply_fill(self) -> None:
        self._run_command(
            "fill",
            {
                "glyph_id": self._glyph.currentText(),
                "fill_rule": self._fill.currentText(),
            },
        )

    def _apply_size(self) -> None:
        self._run_command(
            "size",
            {
                "glyph_id": self._glyph.currentText(),
                "size": int(self._size.currentData()),
            },
        )

    def _refresh_only(self) -> None:
        try:
            snapshot = self._diagnostics.snapshot()
            self._diagnostics.require_current(snapshot)
            self.apply_snapshot(snapshot)
            self._status.setText(
                f"Read snapshot version {snapshot.version}; no recompute requested"
            )
        except GlyphForgeError as exc:
            self._status.setText(str(exc))


def launch_gui(application: GlyphApplication | None = None) -> int:
    if application is None:
        glyphs = sample_glyphs()
        application = GlyphApplication(
            glyphs,
            active_sizes={glyph_id: (12, 16, 24, 32, 48) for glyph_id in glyphs},
            worker_budget=4,
            memory_budget_bytes=1_300_000,
        )
        application.render_all(step="gui_initial_render")
    diagnostics = DiagnosticService(application)

    def command(name: str, payload: dict[str, Any]) -> DiagnosticSnapshot:
        if name == "point":
            application.edit_point(
                payload["glyph_id"],
                contour_index=payload["contour_index"],
                segment_index=payload["segment_index"],
                role=payload["role"],
                point=Point(payload["x"], payload["y"]),
            )
        elif name == "fill":
            application.set_fill_rule(payload["glyph_id"], payload["fill_rule"])
        elif name == "size":
            application.switch_size(payload["glyph_id"], payload["size"])
        else:
            raise ValueError(f"unknown command {name}")
        return diagnostics.snapshot()

    qt_app = QApplication.instance() or QApplication(sys.argv)
    window = EditorWindow(diagnostics, command)
    window.setWindowTitle("GlyphForge diagnostic editor")
    window.resize(1180, 820)
    window.show()
    return qt_app.exec_()


__all__ = ["DiagnosticView", "EditorWindow", "launch_gui"]
