"""PyQt6 diagnostic workbench.

The plotting widgets consume only the immutable diagnostic snapshot.  They do
not estimate IRs, delays, mismatch intervals, filters or residuals.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .hashing import canonical_json, write_text_atomic
from .models import WorkbenchConfig, config_from_mapping, default_config
from .pipeline import PipelineRun, WorkbenchPipeline
from .report import build_report, render_markdown
from .snapshot import Snapshot, SnapshotStore

_COLORS = (
    QColor("#2563eb"),
    QColor("#d97706"),
    QColor("#059669"),
    QColor("#dc2626"),
    QColor("#7c3aed"),
    QColor("#0891b2"),
)


def _min_max(values: Sequence[float], fallback: tuple[float, float] = (-1.0, 1.0)) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return fallback
    low, high = min(finite), max(finite)
    if high <= low:
        return low - 0.5, high + 0.5
    margin = (high - low) * 0.08
    return low - margin, high + margin


class _PlotBase(QWidget):
    def __init__(self, snapshot: Snapshot, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot = snapshot
        self.setMinimumSize(720, 420)

    def set_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def _frame(self, painter: QPainter, rect: QRectF, title: str) -> None:
        painter.fillRect(self.rect(), QColor("#f8fafc"))
        painter.setPen(QPen(QColor("#0f172a"), 1.2))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.drawText(QRectF(rect.left(), 8, rect.width(), 24), Qt.AlignmentFlag.AlignLeft, title)
        painter.setPen(QPen(QColor("#cbd5e1"), 1))
        painter.drawRect(rect)

    def _grid(self, painter: QPainter, rect: QRectF, y_low: float, y_high: float, x_count: int = 8) -> None:
        painter.setPen(QPen(QColor("#e2e8f0"), 1))
        for index in range(1, x_count):
            x = rect.left() + rect.width() * index / x_count
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        for index in range(1, 6):
            y = rect.top() + rect.height() * index / 6.0
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
        painter.setPen(QPen(QColor("#64748b"), 1))
        painter.setFont(QFont("Segoe UI", 8))
        for index in range(7):
            value = y_high - (y_high - y_low) * index / 6.0
            y = rect.top() + rect.height() * index / 6.0
            painter.drawText(QRectF(2, y - 8, rect.left() - 6, 16), Qt.AlignmentFlag.AlignRight, f"{value:.2f}")


class FrequencyDiagnosticsWidget(_PlotBase):
    """Render amplitude/phase, uncertainty bands and calibrated residuals."""

    def diagnostic_numbers(self) -> dict[str, Any]:
        data = self._snapshot.data["frequency_view"]
        return {
            "frequencies_hz": tuple(data["frequencies_hz"]),
            "channels": tuple(data["channels"]),
        }

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        data = self._snapshot.data["frequency_view"]
        frequencies = [float(value) for value in data["frequencies_hz"]]
        channels = list(data["channels"])
        width = self.width()
        height = self.height()
        left, right, top = 64.0, 18.0, 36.0
        gap, bottom = 48.0, 58.0
        panel_height = max(140.0, (height - top - bottom - gap) / 2.0)
        amplitude_rect = QRectF(left, top, width - left - right, panel_height)
        phase_rect = QRectF(left, amplitude_rect.bottom() + gap, width - left - right, panel_height)
        self._frame(painter, amplitude_rect, "Amplitude ratio, uncertainty band and calibrated residual (dB)")
        self._frame(painter, phase_rect, "Phase difference, uncertainty band and calibrated residual (deg)")

        amp_values: list[float] = []
        phase_values: list[float] = []
        for channel in channels:
            amp_values.extend(channel["amplitude_low_db"])
            amp_values.extend(channel["amplitude_high_db"])
            amp_values.extend(channel["residual_amplitude_db"])
            phase_values.extend(channel["phase_low_deg"])
            phase_values.extend(channel["phase_high_deg"])
            phase_values.extend(channel["residual_phase_deg"])
        amp_low, amp_high = _min_max(amp_values)
        phase_low, phase_high = _min_max(phase_values, (-180.0, 180.0))
        self._grid(painter, amplitude_rect, amp_low, amp_high)
        self._grid(painter, phase_rect, phase_low, phase_high)
        x_low, x_high = frequencies[0], frequencies[-1]

        def x_map(freq: float) -> float:
            return amplitude_rect.left() + (freq - x_low) / max(x_high - x_low, 1.0e-12) * amplitude_rect.width()

        def y_map(value: float, rect: QRectF, low: float, high: float) -> float:
            return rect.bottom() - (value - low) / max(high - low, 1.0e-12) * rect.height()

        painter.save()
        for channel_index, channel in enumerate(channels):
            color = _COLORS[channel_index % len(_COLORS)]
            amp = [float(value) for value in channel["amplitude_ratio_db"]]
            amp_lo = [float(value) for value in channel["amplitude_low_db"]]
            amp_hi = [float(value) for value in channel["amplitude_high_db"]]
            residual_amp = [float(value) for value in channel["residual_amplitude_db"]]
            phase = [float(value) for value in channel["phase_difference_deg"]]
            phase_lo = [float(value) for value in channel["phase_low_deg"]]
            phase_hi = [float(value) for value in channel["phase_high_deg"]]
            residual_phase = [float(value) for value in channel["residual_phase_deg"]]

            band_brush = QColor(color)
            band_brush.setAlpha(25)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(band_brush)
            polygon_amp = QPolygonF(
                [QPointF(x_map(f), y_map(v, amplitude_rect, amp_low, amp_high)) for f, v in zip(frequencies, amp_lo)]
                + [QPointF(x_map(f), y_map(v, amplitude_rect, amp_low, amp_high)) for f, v in reversed(list(zip(frequencies, amp_hi)))]
            )
            painter.drawPolygon(polygon_amp)
            polygon_phase = QPolygonF(
                [QPointF(x_map(f), y_map(v, phase_rect, phase_low, phase_high)) for f, v in zip(frequencies, phase_lo)]
                + [QPointF(x_map(f), y_map(v, phase_rect, phase_low, phase_high)) for f, v in reversed(list(zip(frequencies, phase_hi)))]
            )
            painter.drawPolygon(polygon_phase)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color, 2.0))
            painter.drawPolyline(QPolygonF([QPointF(x_map(f), y_map(v, amplitude_rect, amp_low, amp_high)) for f, v in zip(frequencies, amp)]))
            painter.drawPolyline(QPolygonF([QPointF(x_map(f), y_map(v, phase_rect, phase_low, phase_high)) for f, v in zip(frequencies, phase)]))
            residual_pen = QPen(color.darker(150), 1.2, Qt.PenStyle.DashDotLine)
            painter.setPen(residual_pen)
            painter.drawPolyline(QPolygonF([QPointF(x_map(f), y_map(v, amplitude_rect, amp_low, amp_high)) for f, v in zip(frequencies, residual_amp)]))
            painter.drawPolyline(QPolygonF([QPointF(x_map(f), y_map(v, phase_rect, phase_low, phase_high)) for f, v in zip(frequencies, residual_phase)]))
        painter.restore()

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#475569"), 1))
        for index in range(6):
            freq = x_low + (x_high - x_low) * index / 5.0
            x = x_map(freq)
            painter.drawText(QRectF(x - 34, amplitude_rect.bottom() + 4, 68, 16), Qt.AlignmentFlag.AlignCenter, f"{freq:.0f} Hz")
        text = "solid=estimate  shaded=synthetic interval  dash-dot=calibrated residual"
        for index, channel in enumerate(channels):
            color = _COLORS[index % len(_COLORS)]
            painter.setPen(QPen(color, 2))
            x = left + index * 160.0
            painter.drawLine(QPointF(x, height - 30), QPointF(x + 20, height - 30))
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.drawText(QRectF(x + 24, height - 39, 135, 18), Qt.AlignmentFlag.AlignLeft, channel["channel"])
        painter.drawText(QRectF(left, height - 17, width - left, 15), Qt.AlignmentFlag.AlignLeft, text)


class TimeDiagnosticsWidget(_PlotBase):
    """Render snapshot IRs, peak markers and sync-marker truth."""

    def diagnostic_numbers(self) -> dict[str, Any]:
        data = self._snapshot.data["time_view"]
        return {
            "sample_axis": tuple(data["sample_axis"]),
            "time_ms": tuple(data["time_ms"]),
            "channels": tuple(data["channels"]),
        }

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        data = self._snapshot.data["time_view"]
        times = [float(value) for value in data["time_ms"]]
        channels = list(data["channels"])
        rect = QRectF(70, 38, self.width() - 88, self.height() - 92)
        self._frame(painter, rect, "Impulse responses, peak positions and sync-marker truth")
        self._grid(painter, rect, 0.0, float(max(1, len(channels))), x_count=8)
        x_low, x_high = times[0], times[-1]

        def x_map(time_ms: float) -> float:
            return rect.left() + (time_ms - x_low) / max(x_high - x_low, 1.0e-12) * rect.width()

        lane_height = rect.height() / max(len(channels), 1)
        painter.setFont(QFont("Segoe UI", 8))
        for channel_index, channel in enumerate(channels):
            values = [float(value) for value in channel["ir"]]
            scale = max((abs(value) for value in values), default=0.0) or 1.0
            center = rect.top() + lane_height * (channel_index + 0.5)
            color = _COLORS[channel_index % len(_COLORS)]
            painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(rect.left(), center), QPointF(rect.right(), center))
            polyline_values = []
            for time_ms, value in zip(times, values):
                y = center - value / scale * lane_height * 0.36
                polyline_values.append(QPointF(x_map(time_ms), y))
            painter.setPen(QPen(color, 1.4))
            painter.drawPolyline(QPolygonF(polyline_values))
            sample_rate = float(self._snapshot.data["audio"]["sample_rate"])
            peak_time_ms = float(channel["peak_relative_sample"]) * 1000.0 / sample_rate
            peak_x = x_map(peak_time_ms)
            painter.setPen(QPen(color.darker(130), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(peak_x, center - lane_height * 0.43), QPointF(peak_x, center + lane_height * 0.43))
            painter.setPen(QPen(QColor("#15803d"), 1.2, Qt.PenStyle.DotLine))
            for marker_ms in channel["sync_marker_relative_ms"]:
                marker_x = x_map(float(marker_ms))
                painter.drawLine(QPointF(marker_x, center - lane_height * 0.40), QPointF(marker_x, center + lane_height * 0.40))
            painter.setPen(QPen(QColor("#0f172a"), 1))
            painter.drawText(
                QRectF(rect.left() + 4, center - lane_height * 0.45, 210, 16),
                Qt.AlignmentFlag.AlignLeft,
                f"{channel['channel']}  peak {channel['peak_relative_sample']:.4f} smp / {channel['peak_level_db']:.2f} dB",
            )
        painter.setPen(QPen(QColor("#475569"), 1))
        for index in range(6):
            value = x_low + (x_high - x_low) * index / 5.0
            x = x_map(value)
            painter.drawText(QRectF(x - 45, rect.bottom() + 5, 90, 16), Qt.AlignmentFlag.AlignCenter, f"{value:.1f} ms")
        painter.drawText(QRectF(rect.left(), self.height() - 25, rect.width(), 18), Qt.AlignmentFlag.AlignLeft, "green dotted = sync truth; colored dashed = IR peak")


class SnapshotSummaryWidget(QTextBrowser):
    def __init__(self, snapshot: Snapshot, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.set_snapshot(snapshot)

    def set_snapshot(self, snapshot: Snapshot) -> None:
        data = snapshot.data
        delay_rows = "".join(
            f"<tr><td>{item['channel']}</td><td>{item['truth_relative_samples']:.6f}</td>"
            f"<td>{item['estimated_relative_samples']:.6f}</td><td>{item['bias_samples']:.6f}</td>"
            f"<td>{item['error_bound_samples']:.6f}</td><td>{item['status']}</td></tr>"
            for item in data["delay"]["channels"]
        )
        mismatch_rows = "".join(
            f"<tr><td>{item['channel']}</td><td>{item['weighted_gain_db']:.4f}</td>"
            f"<td>{item['truth_gain_db']:.4f}</td><td>{item['gain_bias_db']:.4f}</td>"
            f"<td>[{item['interval_low_db']:.4f}, {item['interval_high_db']:.4f}]</td></tr>"
            for item in data["mismatch"]["channels"]
        )
        residual_rows = "".join(
            f"<tr><td>{item['channel']}</td><td>{item['residual_gain_db']:.6f}</td>"
            f"<td>{item['max_amplitude_excess_db']:.6f}</td><td>{item['max_phase_excess_deg']:.6f}</td>"
            f"<td>{item['compatible']}</td></tr>"
            for item in data["verification"]["channels"]
        )
        errors = "<br>".join(
            f"<b>{item['code']}</b> ({item['stage']}): {item['message']}" for item in data["errors"]
        ) or "无"
        html = f"""
        <h2>诊断快照摘要</h2>
        <p><b>版本戳</b> {data['version_stamp']}<br><b>内容哈希</b> {data['content_hash']}</p>
        <h3>同步/延迟</h3>
        <table border='1' cellspacing='0' cellpadding='4'><tr><th>通道</th><th>真值</th><th>估计</th><th>偏差</th><th>上界</th><th>状态</th></tr>{delay_rows}</table>
        <h3>增益失配与区间</h3>
        <table border='1' cellspacing='0' cellpadding='4'><tr><th>通道</th><th>估计 dB</th><th>真值 dB</th><th>偏差 dB</th><th>区间</th></tr>{mismatch_rows}</table>
        <h3>校准重测</h3>
        <table border='1' cellspacing='0' cellpadding='4'><tr><th>通道</th><th>残余 dB</th><th>幅度超界 dB</th><th>相位超界 deg</th><th>相容</th></tr>{residual_rows}</table>
        <h3>业务错误</h3><p>{errors}</p>
        """
        self.setHtml(html)


class StageTable(QTableWidget):
    def __init__(self, snapshot: Snapshot, parent: QWidget | None = None) -> None:
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["阶段", "内容哈希", "复用来源", "是否复用", "缓存键"])
        self.set_snapshot(snapshot)

    def set_snapshot(self, snapshot: Snapshot) -> None:
        reuse = list(snapshot.data["reuse"].items())
        self.setRowCount(len(reuse))
        for row, (stage, record) in enumerate(reuse):
            values = [
                stage,
                record["content_hash"][:20],
                record["compute_source"],
                str(record["reused"]),
                record["key"][:20],
            ]
            for column, value in enumerate(values):
                self.setItem(row, column, QTableWidgetItem(str(value)))
        self.resizeColumnsToContents()


class PipelineWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: WorkbenchConfig,
        cache_dir: Path,
        snapshot_path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._cache_dir = cache_dir
        self._snapshot_path = snapshot_path

    def run(self) -> None:
        try:
            pipeline = WorkbenchPipeline(self._cache_dir, self._snapshot_path)
            result = pipeline.run(self._config)
            self.completed.emit(result)
        except Exception as exc:  # GUI boundary converts exceptions to text
            self.failed.emit(str(exc))


class DiagnosticWindow(QMainWindow):
    def __init__(
        self,
        snapshot: Snapshot | None,
        *,
        output_dir: str | Path = "output/gui",
        auto_run: bool = False,
    ) -> None:
        super().__init__()
        self.setWindowTitle("声学测量与校准诊断工作台")
        self.resize(1440, 900)
        self.output_dir = Path(output_dir)
        self.snapshot_path = self.output_dir / "current_snapshot.json"
        self.cache_dir = self.output_dir / "state" / "cache"
        self._snapshot = snapshot
        self._worker: PipelineWorker | None = None

        root = QWidget()
        root_layout = QVBoxLayout(root)
        controls = QHBoxLayout()
        form = QFormLayout()
        form.setContentsMargins(8, 4, 8, 4)
        self.sweep_duration = QDoubleSpinBox()
        self.sweep_duration.setRange(0.1, 30.0)
        self.sweep_duration.setDecimals(3)
        self.sweep_duration.setSuffix(" s")
        self.regularization = QLineEdit()
        self.band_low = QSpinBox()
        self.band_low.setRange(20, 20_000)
        self.band_high = QSpinBox()
        self.band_high.setRange(30, 23_900)
        self.filter_length = QSpinBox()
        self.filter_length.setRange(33, 8_193)
        self.filter_length.setSingleStep(2)
        form.addRow("扫频长度", self.sweep_duration)
        form.addRow("正则化 λ", self.regularization)
        form.addRow("分析低端 Hz", self.band_low)
        form.addRow("分析高端 Hz", self.band_high)
        form.addRow("FIR 长度", self.filter_length)
        controls.addLayout(form)
        buttons = QVBoxLayout()
        self.run_button = QPushButton("增量重测")
        self.reload_button = QPushButton("重新载入快照")
        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.reload_button)
        buttons.addWidget(self.status_label)
        buttons.addStretch(1)
        controls.addLayout(buttons)
        root_layout.addLayout(controls)

        self.tabs = QTabWidget()
        self.frequency_widget: FrequencyDiagnosticsWidget | None = None
        self.time_widget: TimeDiagnosticsWidget | None = None
        self.summary_widget: SnapshotSummaryWidget | None = None
        self.stage_table: StageTable | None = None
        root_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(root)
        self.run_button.clicked.connect(self._run_incremental)
        self.reload_button.clicked.connect(self._reload_snapshot)
        if snapshot is None and self.snapshot_path.exists():
            snapshot = SnapshotStore(self.snapshot_path).load()
        if snapshot is not None:
            self._apply_snapshot(snapshot)
        elif auto_run:
            QTimer.singleShot(150, self._run_incremental)

    def _apply_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot
        config = config_from_mapping(snapshot.data["config"])
        self.sweep_duration.setValue(config.measurement.sweep_duration_s)
        self.regularization.setText(f"{config.deconvolution.regularization:.12g}")
        self.band_low.setValue(int(round(config.analysis.band_low_hz)))
        self.band_high.setValue(int(round(config.analysis.band_high_hz)))
        self.filter_length.setValue(config.calibration.filter_length)
        self.frequency_widget = FrequencyDiagnosticsWidget(snapshot)
        self.time_widget = TimeDiagnosticsWidget(snapshot)
        self.summary_widget = SnapshotSummaryWidget(snapshot)
        self.stage_table = StageTable(snapshot)
        self.tabs.clear()
        self.tabs.addTab(self.frequency_widget, "频域诊断")
        self.tabs.addTab(self.time_widget, "时域冲激响应")
        self.tabs.addTab(self.summary_widget, "摘要与错误")
        self.tabs.addTab(self.stage_table, "阶段与复用")
        self.status_label.setText(f"快照 {snapshot.version_stamp}")
        self._write_current_report(snapshot)

    def _current_config(self) -> WorkbenchConfig:
        if self._snapshot is None:
            return default_config()
        config = config_from_mapping(self._snapshot.data["config"])
        try:
            regularization = float(self.regularization.text())
        except ValueError:
            regularization = config.deconvolution.regularization
        return config.with_measurement(sweep_duration_s=float(self.sweep_duration.value())).with_deconvolution(
            regularization=regularization
        ).with_analysis(
            band_low_hz=float(self.band_low.value()),
            band_high_hz=float(self.band_high.value()),
        ).with_calibration(filter_length=int(self.filter_length.value()))

    def _run_incremental(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            config = self._current_config()
        except Exception as exc:
            QMessageBox.critical(self, "参数错误", str(exc))
            return
        self.run_button.setEnabled(False)
        self.status_label.setText("正在进行增量重测…")
        self._worker = PipelineWorker(config, self.cache_dir, self.snapshot_path, self)
        self._worker.completed.connect(self._on_pipeline_complete)
        self._worker.failed.connect(self._on_pipeline_failed)
        self._worker.start()

    def _on_pipeline_complete(self, result: PipelineRun) -> None:
        self.run_button.setEnabled(True)
        self._apply_snapshot(result.snapshot)

    def _on_pipeline_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.status_label.setText(message)
        QMessageBox.critical(self, "运行失败", message)

    def _reload_snapshot(self) -> None:
        try:
            snapshot = SnapshotStore(self.snapshot_path).load()
            self._apply_snapshot(snapshot)
        except Exception as exc:
            QMessageBox.critical(self, "快照不可用", str(exc))

    def _write_current_report(self, snapshot: Snapshot) -> None:
        report = build_report(snapshot)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self.output_dir / "diagnostic_report.json", canonical_json(report, pretty=True))
        write_text_atomic(self.output_dir / "diagnostic_report.md", render_markdown(report))


def render_snapshot_png(snapshot: Snapshot, path: str | Path, *, width: int = 1440, height: int = 720) -> None:
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    widget = QWidget()
    widget.resize(width, height)
    layout = QVBoxLayout(widget)
    frequency = FrequencyDiagnosticsWidget(snapshot)
    time_view = TimeDiagnosticsWidget(snapshot)
    layout.addWidget(frequency)
    layout.addWidget(time_view)
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor("#ffffff"))
    painter = QPainter(image)
    widget.render(painter)
    painter.end()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(path), "PNG"):
        raise RuntimeError("failed to save diagnostic PNG")
    widget.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PyQt6 acoustic diagnostics workbench")
    parser.add_argument("snapshot", nargs="?", help="snapshot JSON path")
    parser.add_argument("--output", default="output/gui", help="output/cache directory")
    parser.add_argument("--auto-run", action="store_true", help="run a default measurement if no snapshot exists")
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    snapshot = None
    if args.snapshot:
        snapshot = SnapshotStore(args.snapshot).load()
    window = DiagnosticWindow(snapshot, output_dir=args.output, auto_run=args.auto_run)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
