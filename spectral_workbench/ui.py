"""PyQt5 diagnostic view that renders only published immutable snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QWidget

from .snapshot import DiagnosticSnapshot, SnapshotStore, load_snapshot


_PALETTE = (
    QColor(255, 72, 54),
    QColor(38, 224, 92),
    QColor(58, 112, 255),
    QColor(255, 196, 32),
    QColor(206, 68, 234),
    QColor(38, 222, 222),
    QColor(255, 122, 24),
    QColor(190, 190, 196),
)


class SnapshotImageWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_snapshot(self, snapshot: DiagnosticSnapshot) -> None:
        rgb = np.ascontiguousarray(snapshot.pseudocolor, dtype=np.uint8)
        height, width, _channels = rgb.shape
        overlay = np.array(rgb, copy=True)
        mask = snapshot.ill_conditioned_mask.astype(bool)
        if np.any(mask):
            red = np.array([255, 32, 32], dtype=np.float64)
            overlay[mask] = np.rint(0.55 * overlay[mask] + 0.45 * red).astype(np.uint8)
        self._image = QImage(overlay.data, width, height, width * 3, QImage.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(16, 18, 22))
        if self._image is None:
            painter.setPen(QColor(180, 184, 194))
            painter.drawText(self.rect(), Qt.AlignCenter, "No published snapshot")
            return
        target = self.rect().adjusted(8, 8, -8, -8)
        scaled = self._image.size().scaled(target.size(), Qt.KeepAspectRatio)
        x = target.x() + (target.width() - scaled.width()) // 2
        y = target.y() + (target.height() - scaled.height()) // 2
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.drawImage(QRectF(x, y, scaled.width(), scaled.height()), self._image)
        painter.setPen(QPen(QColor(90, 96, 108), 1))
        painter.drawRect(QRectF(x, y, scaled.width(), scaled.height()))


class ErrorBandCurveWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.snapshot: Optional[DiagnosticSnapshot] = None
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_snapshot(self, snapshot: DiagnosticSnapshot) -> None:
        self.snapshot = snapshot
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 22, 28))
        painter.setRenderHint(QPainter.Antialiasing, True)
        frame = self.rect().adjusted(52, 28, -18, -52)
        painter.setPen(QPen(QColor(80, 86, 98), 1))
        painter.drawRect(frame)
        if self.snapshot is None:
            painter.setPen(QColor(180, 184, 194))
            painter.drawText(frame, Qt.AlignCenter, "No snapshot")
            return

        positions = self.snapshot.band_positions.astype(np.float64)
        curves = self.snapshot.concentration_curves.astype(np.float64)
        low = self.snapshot.error_low.astype(np.float64)
        high = self.snapshot.error_high.astype(np.float64)
        minimum = float(min(np.min(low), 0.0))
        maximum = float(np.max(high))
        if maximum <= minimum:
            maximum = minimum + 1.0
        span = maximum - minimum
        minimum -= 0.05 * span
        maximum += 0.05 * span

        def map_point(index: int, value: float) -> QPointF:
            x = frame.left() + positions[index] * frame.width()
            y = frame.bottom() - ((value - minimum) / (maximum - minimum)) * frame.height()
            return QPointF(x, y)

        for band in range(positions.size):
            if self.snapshot.ill_conditioned_bands[band]:
                center = frame.left() + positions[band] * frame.width()
                width = max(3.0, frame.width() / max(1, positions.size - 1))
                painter.fillRect(
                    QRectF(center - width / 2.0, frame.top(), width, frame.height()),
                    QColor(220, 44, 44, 46),
                )

        for k in range(curves.shape[0]):
            color = _PALETTE[k % len(_PALETTE)]
            upper = QPolygonF([map_point(index, high[k, index]) for index in range(positions.size)])
            lower_points = [map_point(index, low[k, index]) for index in range(positions.size - 1, -1, -1)]
            polygon = upper
            for point in lower_points:
                polygon.append(point)
            fill = QColor(color)
            fill.setAlpha(34)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill)
            painter.drawPolygon(polygon)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(color, 2.0))
            points = [map_point(index, curves[k, index]) for index in range(positions.size)]
            for first, second in zip(points[:-1], points[1:]):
                painter.drawLine(first, second)

        painter.setPen(QPen(QColor(130, 136, 148), 1))
        for tick in range(5):
            value = minimum + (maximum - minimum) * tick / 4.0
            point = map_point(0, value)
            painter.drawLine(QPointF(frame.left() - 4, point.y()), QPointF(frame.left(), point.y()))
            painter.setPen(QColor(170, 176, 188))
            painter.drawText(QRectF(2, point.y() - 8, 46, 16), Qt.AlignRight | Qt.AlignVCenter, f"{value:.3f}")
            painter.setPen(QPen(QColor(130, 136, 148), 1))

        for first_band, label in ((0, "band 0"), (positions.size - 1, f"band {positions.size - 1}")):
            point = map_point(first_band, minimum)
            painter.setPen(QColor(170, 176, 188))
            alignment = Qt.AlignLeft if first_band == 0 else Qt.AlignRight
            painter.drawText(QRectF(point.x() - 30, frame.bottom() + 6, 60, 18), alignment, label)

        legend_x = frame.left() + 8
        legend_y = frame.top() + 8
        for k, name in enumerate(self.snapshot.fluorophore_names):
            color = _PALETTE[k % len(_PALETTE)]
            painter.fillRect(QRectF(legend_x, legend_y + k * 18, 16, 3), color)
            painter.setPen(QColor(210, 214, 222))
            painter.drawText(QRectF(legend_x + 22, legend_y + k * 18 - 7, 150, 16), Qt.AlignLeft, name)

        painter.setPen(QColor(220, 92, 82))
        painter.drawText(QRectF(frame.right() - 180, frame.top() + 8, 174, 18), Qt.AlignRight, "red: ill-conditioned")


class DiagnosticWindow(QMainWindow):
    """Read-only view. It deliberately has no pipeline/service reference."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Spectral Imaging Diagnostic Workbench")
        self.resize(1120, 620)
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.image_widget = SnapshotImageWidget(central)
        self.curve_widget = ErrorBandCurveWidget(central)
        layout.addWidget(self.image_widget, 1)
        layout.addWidget(self.curve_widget, 1)
        self.setCentralWidget(central)
        self.status = QLabel("Waiting for immutable snapshot", self)
        self.status.setStyleSheet("color: #c8ccd4; padding: 4px; background: #14161c;")
        self.statusBar().addWidget(self.status, 1)
        self._snapshot: Optional[DiagnosticSnapshot] = None

    def render_snapshot(self, snapshot: DiagnosticSnapshot, store: SnapshotStore) -> None:
        store.validate_render(snapshot)
        self._snapshot = snapshot
        self.image_widget.set_snapshot(snapshot)
        self.curve_widget.set_snapshot(snapshot)
        self.setWindowTitle(
            f"Spectral Imaging Diagnostic Workbench | v{snapshot.version} | {snapshot.content_hash[:12]}"
        )
        self.status.setText(
            f"snapshot v{snapshot.version}  hash={snapshot.content_hash[:16]}  "
            f"source_run={snapshot.source_run_hash[:16]}  render-only"
        )
        self.update()


def create_offscreen_window(snapshot: DiagnosticSnapshot, store: SnapshotStore) -> DiagnosticWindow:
    window = DiagnosticWindow()
    window.render_snapshot(snapshot, store)
    return window


def show_snapshot_directory(directory: str | Path, store: Optional[SnapshotStore] = None) -> int:
    from PyQt5.QtWidgets import QApplication

    snapshot = load_snapshot(directory)
    active_store = store or SnapshotStore()
    active_store.publish(snapshot)
    app = QApplication.instance() or QApplication([])
    window = DiagnosticWindow()
    window.render_snapshot(snapshot, active_store)
    window.show()
    return int(app.exec_())
