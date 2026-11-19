"""PySide6 desktop workbench for deterministic mocap cleaning."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .constants import (
    COLOR_JITTER,
    COLOR_JUMP,
    COLOR_KEYFRAME,
    COLOR_MISSING,
    COLOR_ROOT,
    DEFAULT_SEED,
    DEFECT_JITTER,
    DEFECT_JUMP,
    DEFECT_MISSING,
    DEFECT_ROOT_SLIP,
    FRAME_COUNTS,
    PREVIEW_BUDGET_MS,
    SKELETON,
    THRESHOLD_RANGES,
)
from .errors import BusinessError, ErrorCode
from .model import Snapshot
from .preview import PreviewResult
from .service import WorkbenchService


DEFECT_COLORS = {
    DEFECT_JUMP: COLOR_JUMP,
    DEFECT_JITTER: COLOR_JITTER,
    DEFECT_MISSING: COLOR_MISSING,
    DEFECT_ROOT_SLIP: COLOR_ROOT,
}


class TimelineWidget(QWidget):
    frameChanged = Signal(int)
    anchorSelected = Signal(int)
    anchorMovePreview = Signal(int, int)
    anchorMoveFinished = Signal(int, int)
    addKeyframeRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._snapshot: Snapshot | None = None
        self._candidate = False
        self._current_frame = 0
        self._selected_anchor: int | None = None
        self._drag_old: int | None = None
        self._drag_target: int | None = None
        self._defect_runs: list[tuple[int, int, str]] = []
        self._root_points: list[float] = []
        self._error_points: list[float] = []
        self._keyframes: list[int] = []
        self._dirty_ranges: list[tuple[int, int]] = []
        self._margin_left = 18
        self._margin_right = 18

    def set_snapshot(self, snapshot: Snapshot, candidate: bool = False) -> None:
        self._snapshot = snapshot
        self._candidate = candidate
        self._current_frame = min(self._current_frame, snapshot.frame_count - 1)
        self._prepare_cache()
        self.update()

    def set_current_frame(self, frame: int) -> None:
        if self._snapshot is None:
            return
        self._current_frame = max(0, min(self._snapshot.frame_count - 1, int(frame)))
        self.update()

    def set_selected_anchor(self, frame: int | None) -> None:
        self._selected_anchor = frame
        self.update()

    def _prepare_cache(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return
        runs: list[tuple[int, int, str]] = []
        current_start = None
        current_type = None
        for frame in range(snapshot.frame_count):
            types = []
            for joint in range(snapshot.record.joint_count):
                if snapshot.detection.joint_bad[frame, joint]:
                    types.append(str(snapshot.detection.joint_type[frame, joint]))
            if snapshot.detection.root_bad[frame]:
                types.append(DEFECT_ROOT_SLIP)
            chosen = None
            for candidate in (DEFECT_MISSING, DEFECT_JUMP, DEFECT_JITTER, DEFECT_ROOT_SLIP):
                if candidate in types:
                    chosen = candidate
                    break
            if chosen is None:
                if current_start is not None and current_type is not None:
                    runs.append((current_start, frame - 1, current_type))
                    current_start = None
                    current_type = None
            elif current_start is None:
                current_start = frame
                current_type = chosen
            elif chosen != current_type:
                runs.append((current_start, frame - 1, current_type))
                current_start = frame
                current_type = chosen
        if current_start is not None and current_type is not None:
            runs.append((current_start, snapshot.frame_count - 1, current_type))
        self._defect_runs = runs

        root_y = snapshot.repair.roots[:, 1]
        low = float(root_y.min())
        high = float(root_y.max())
        span = max(1e-9, high - low)
        if snapshot.frame_count <= 2000:
            indices = range(snapshot.frame_count)
        else:
            step = max(1, snapshot.frame_count // 2000)
            indices = range(0, snapshot.frame_count, step)
        self._root_points = [float((root_y[index] - low) / span) for index in indices]
        errors: dict[int, float] = {}
        for detail in snapshot.evaluation.details:
            frame = detail.frame
            value = max(
                float(detail.joint_angle_error_deg or 0.0) / 2.0,
                float(detail.root_error_m or 0.0) / 0.02,
            )
            errors[frame] = max(errors.get(frame, 0.0), min(1.0, value))
        self._error_points = [errors.get(frame, 0.0) for frame in range(snapshot.frame_count)]
        anchors: set[int] = set(snapshot.keyframes)
        for joint_anchors in snapshot.anchors.joint_auto:
            anchors.update(joint_anchors)
        anchors.update(snapshot.anchors.root_auto)
        anchors.difference_update(snapshot.suppressed_auto)
        self._keyframes = sorted(anchors)
        self._dirty_ranges = []
        if snapshot.closure:
            frames = {frame for frame, _ in snapshot.closure.dirty_joint_coords}
            frames.update(snapshot.closure.dirty_root_frames)
            ordered = sorted(frames)
            if ordered:
                start = previous = ordered[0]
                for frame in ordered[1:]:
                    if frame != previous + 1:
                        self._dirty_ranges.append((start, previous))
                        start = frame
                    previous = frame
                self._dirty_ranges.append((start, previous))

    def _frame_x(self, frame: int) -> float:
        assert self._snapshot is not None
        width = max(1, self.width() - self._margin_left - self._margin_right)
        if self._snapshot.frame_count <= 1:
            return float(self._margin_left)
        return self._margin_left + frame / (self._snapshot.frame_count - 1) * width

    def _frame_at_x(self, x: float) -> int:
        assert self._snapshot is not None
        width = max(1, self.width() - self._margin_left - self._margin_right)
        ratio = (x - self._margin_left) / width
        return max(0, min(self._snapshot.frame_count - 1, int(round(ratio * (self._snapshot.frame_count - 1)))))

    def paintEvent(self, event) -> None:  # noqa: N802
        snapshot = self._snapshot
        if snapshot is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = QRectF(0, 0, self.width(), self.height())
        painter.fillRect(rect, QColor("#0f151d"))
        band_defects = QRectF(self._margin_left, 12, self.width() - self._margin_left - self._margin_right, 74)
        band_root = QRectF(self._margin_left, 96, self.width() - self._margin_left - self._margin_right, 58)
        band_errors = QRectF(self._margin_left, 160, self.width() - self._margin_left - self._margin_right, 28)
        band_keys = QRectF(self._margin_left, 196, self.width() - self._margin_left - self._margin_right, 30)
        painter.fillRect(band_defects, QColor("#151d28"))
        painter.fillRect(band_root, QColor("#111923"))
        painter.fillRect(band_errors, QColor("#151d28"))
        painter.fillRect(band_keys, QColor("#111923"))

        # Fine frame grid at a bounded density.
        painter.setPen(QPen(QColor(255, 255, 255, 18), 1))
        tick = max(1, snapshot.frame_count // 120)
        for frame in range(0, snapshot.frame_count, tick):
            x = self._frame_x(frame)
            painter.drawLine(QPointF(x, 12), QPointF(x, 226))

        for start, end, defect_type in self._defect_runs:
            x0 = self._frame_x(start)
            x1 = self._frame_x(end) + max(2.0, 1.0)
            color = QColor(DEFECT_COLORS.get(defect_type, "#ffffff"))
            color.setAlpha(170)
            painter.fillRect(QRectF(x0, 16, max(2.0, x1 - x0), 66), color)
        painter.setPen(QColor("#9fb0c3"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(4, 26, "缺陷")
        painter.drawText(4, 118, "根 Y")
        painter.drawText(4, 178, "误差")
        painter.drawText(4, 214, "关键帧")

        if self._root_points:
            path = QPainterPath()
            count = len(self._root_points)
            for index, value in enumerate(self._root_points):
                frame_ratio = index / max(1, count - 1)
                x = band_root.left() + frame_ratio * band_root.width()
                y = band_root.bottom() - value * band_root.height()
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor(COLOR_ROOT), 1.6))
            painter.drawPath(path)

        for frame, value in enumerate(self._error_points):
            if value <= 0.0:
                continue
            x = self._frame_x(frame)
            color = QColor(COLOR_JUMP if value >= 1.0 else "#e4c45a")
            color.setAlpha(150)
            height = max(2.0, value * band_errors.height())
            painter.fillRect(QRectF(x, band_errors.bottom() - height, max(2.0, band_errors.width() / max(1, snapshot.frame_count)), height), color)

        for start, end in self._dirty_ranges:
            x0 = self._frame_x(start)
            x1 = self._frame_x(end) + 2
            color = QColor("#55d6be")
            color.setAlpha(35)
            painter.fillRect(QRectF(x0, 12, max(2.0, x1 - x0), 214), color)

        painter.setPen(QPen(QColor("#243346"), 1))
        painter.drawLine(QPointF(band_keys.left(), band_keys.center().y()), QPointF(band_keys.right(), band_keys.center().y()))
        for frame in self._keyframes:
            x = self._frame_x(frame)
            y = band_keys.center().y()
            color = QColor(COLOR_KEYFRAME if frame in snapshot.keyframes else "#63b3ed")
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#e6f0ff") if frame == self._selected_anchor else color, 1))
            painter.drawPolygon(
                [
                    QPointF(x, y - 5),
                    QPointF(x + 5, y),
                    QPointF(x, y + 5),
                    QPointF(x - 5, y),
                ]
            )
        if self._drag_target is not None:
            x = self._frame_x(self._drag_target)
            painter.setBrush(QColor("#ffd166"))
            painter.setPen(QPen(QColor("#fff2bd"), 1))
            painter.drawEllipse(QPointF(x, band_keys.center().y()), 5, 5)

        x_current = self._frame_x(self._current_frame)
        painter.setPen(QPen(QColor("#f8fafc"), 1.2))
        painter.drawLine(QPointF(x_current, 8), QPointF(x_current, self.height() - 18))
        painter.setPen(QColor("#d9e4f1"))
        painter.drawText(
            QRectF(max(0, x_current - 50), self.height() - 18, 100, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"帧 {self._current_frame}",
        )
        status = "候选预览（未提交）" if self._candidate else f"已提交 {snapshot.version.label}"
        painter.setPen(QColor("#ffd166") if self._candidate else QColor("#7dd3fc"))
        painter.drawText(
            QRectF(self.width() - 270, 2, 250, 16),
            Qt.AlignmentFlag.AlignRight,
            status,
        )

    def _nearest_anchor(self, x: float) -> int | None:
        if not self._keyframes:
            return None
        best = min(self._keyframes, key=lambda frame: abs(self._frame_x(frame) - x))
        return best if abs(self._frame_x(best) - x) <= 7.0 else None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._snapshot is None or event.button() != Qt.MouseButton.LeftButton:
            return
        position = event.position()
        anchor = self._nearest_anchor(position.x())
        if anchor is not None:
            self._selected_anchor = anchor
            self._drag_old = anchor
            self._drag_target = anchor
            self.anchorSelected.emit(anchor)
            self.update()
            return
        frame = self._frame_at_x(position.x())
        self._current_frame = frame
        self._selected_anchor = None
        self.frameChanged.emit(frame)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._snapshot is None:
            return
        if self._drag_old is not None:
            target = self._frame_at_x(event.position().x())
            if target != self._drag_target:
                self._drag_target = target
                self.anchorMovePreview.emit(self._drag_old, target)
                self.update()
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            frame = self._frame_at_x(event.position().x())
            if frame != self._current_frame:
                self._current_frame = frame
                self.frameChanged.emit(frame)
                self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_old is not None and self._drag_target is not None:
            old, target = self._drag_old, self._drag_target
            self._drag_old = None
            self._drag_target = None
            self.anchorMoveFinished.emit(old, target)
            self.update()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._snapshot is None:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.addKeyframeRequested.emit(self._frame_at_x(event.position().x()))


class WorkbenchWindow(QMainWindow):
    def __init__(self, workspace: Path, frame_count: int, seed: int) -> None:
        super().__init__()
        self.workspace = Path(workspace).resolve()
        self.frame_count = frame_count
        self.seed = seed
        self.service: WorkbenchService | None = None
        self._display_snapshot: Snapshot | None = None
        self._display_is_candidate = False
        self._preview_result: PreviewResult | None = None
        self._selected_anchor: int | None = None
        self._drag_origin: tuple[int, set[int], set[int]] | None = None
        self._draft_thresholds = None
        self._draft_keyframes: set[int] = set()
        self._draft_suppressed: set[int] = set()
        self._syncing_thresholds = False
        self._pending_preview = False
        self._last_preview_spec: dict[str, Any] | None = None

        self.setWindowTitle("MocapClean Workbench — 动捕废帧审计与稀疏关键姿态修复")
        self.resize(1500, 940)
        self._build_ui()
        self._setup_shortcuts()
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(140)
        self._preview_debounce.timeout.connect(self._launch_preview)
        self._preview_poll = QTimer(self)
        self._preview_poll.setInterval(45)
        self._preview_poll.timeout.connect(self._poll_preview)
        self._open_service(frame_count, seed)

    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #0b1118; color: #dbe6f3; }
            QGroupBox { border: 1px solid #26384c; border-radius: 8px; margin-top: 12px; padding-top: 10px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9fc4e8; }
            QPushButton { background: #1d3550; border: 1px solid #365a80; border-radius: 6px; padding: 7px 11px; }
            QPushButton:hover { background: #25476a; }
            QPushButton:disabled { color: #64748b; background: #131d29; border-color: #223044; }
            QPushButton#primary { background: #0f766e; border-color: #14b8a6; font-weight: 700; }
            QPushButton#primary:hover { background: #0d9488; }
            QComboBox, QDoubleSpinBox, QLabel#badge { background: #111a25; border: 1px solid #2a4058; border-radius: 5px; padding: 4px; }
            QTabWidget::pane { border: 1px solid #26384c; border-radius: 6px; }
            QTabBar::tab { background: #101924; padding: 7px 12px; border: 1px solid #22344a; }
            QTabBar::tab:selected { background: #1b2b3d; color: #8bd3ff; }
            QTableWidget { background: #0d151f; alternate-background-color: #111c28; gridline-color: #213044; }
            QHeaderView::section { background: #172535; color: #bcd5ec; padding: 5px; border: 0; }
            QTextEdit { background: #0d151f; border: 1px solid #24384d; border-radius: 6px; }
            """
        )
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("MocapClean Workbench")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color:#f1f7ff;")
        subtitle = QLabel("确定性废帧审计 · 稀疏关键姿态修复 · 增量失效闭包")
        subtitle.setStyleSheet("color:#8da3ba;")
        header.addWidget(title)
        header.addSpacing(14)
        header.addWidget(subtitle)
        header.addStretch(1)
        header.addWidget(QLabel("帧档"))
        self.tier_combo = QComboBox()
        for frames in FRAME_COUNTS:
            self.tier_combo.addItem(str(frames), frames)
        self.tier_combo.setCurrentIndex(self.tier_combo.findData(self.frame_count))
        header.addWidget(self.tier_combo)
        header.addWidget(QLabel(f"seed {self.seed}"))
        self.load_button = QPushButton("载入/生成")
        self.load_button.clicked.connect(self._load_selected_tier)
        header.addWidget(self.load_button)
        self.version_badge = QLabel("starting")
        self.version_badge.setObjectName("badge")
        header.addWidget(self.version_badge)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.preview_banner = QLabel("等待工作台载入…")
        self.preview_banner.setStyleSheet(
            "background:#12202e;border:1px solid #2f4d6c;border-radius:7px;padding:7px 10px;color:#a9c7e3;"
        )
        left_layout.addWidget(self.preview_banner)
        self.timeline = TimelineWidget()
        self.timeline.frameChanged.connect(self._on_frame_changed)
        self.timeline.anchorSelected.connect(self._on_anchor_selected)
        self.timeline.anchorMovePreview.connect(self._on_anchor_move_preview)
        self.timeline.anchorMoveFinished.connect(self._on_anchor_move_finished)
        self.timeline.addKeyframeRequested.connect(self._add_anchor_at)
        left_layout.addWidget(self.timeline, 0)
        self.tabs = QTabWidget()
        self.eval_table = QTableWidget()
        self.eval_table.setColumnCount(7)
        self.eval_table.setHorizontalHeaderLabels(
            ["帧", "关节", "类型", "角度误差°", "根误差 m", "可信", "容差"]
        )
        self.eval_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.eval_table.setAlternatingRowColors(True)
        self.eval_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(4)
        self.detection_table.setHorizontalHeaderLabels(["帧", "关节", "类型", "段"])
        self.detection_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.detection_table.setAlternatingRowColors(True)
        self.closure_text = QTextEdit()
        self.closure_text.setReadOnly(True)
        self.performance_text = QTextEdit()
        self.performance_text.setReadOnly(True)
        self.tabs.addTab(self.eval_table, "逐帧评估明细")
        self.tabs.addTab(self.detection_table, "检测清单")
        self.tabs.addTab(self.closure_text, "闭包 / 一致性")
        self.tabs.addTab(self.performance_text, "性能与恢复")
        left_layout.addWidget(self.tabs, 1)
        splitter.addWidget(left)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        threshold_group = QGroupBox("检测阈值（修改即生成候选预览）")
        threshold_form = QFormLayout(threshold_group)
        self.threshold_spins: dict[str, QDoubleSpinBox] = {}
        labels = {
            "joint_jump_deg": "关节跳变 (°)",
            "joint_jitter_deg": "二阶差分/抖震 (°)",
            "invalid_norm_min": "无效范数下限",
            "root_slip_m": "根滑移残差 (m)",
        }
        for key, label in labels.items():
            spin = QDoubleSpinBox()
            low, high = THRESHOLD_RANGES[key]
            spin.setRange(low, high)
            spin.setDecimals(4)
            spin.setSingleStep(0.1 if key in ("joint_jump_deg", "joint_jitter_deg") else 0.005)
            spin.valueChanged.connect(self._on_threshold_changed)
            self.threshold_spins[key] = spin
            threshold_form.addRow(label, spin)
        sidebar_layout.addWidget(threshold_group)

        key_group = QGroupBox("关键姿态")
        key_layout = QVBoxLayout(key_group)
        self.anchor_label = QLabel("当前帧 0 · 未选中锚点")
        self.anchor_label.setWordWrap(True)
        key_layout.addWidget(self.anchor_label)
        key_row = QHBoxLayout()
        self.add_button = QPushButton("添加当前帧")
        self.add_button.clicked.connect(lambda: self._add_anchor_at(self.timeline._current_frame))
        self.delete_button = QPushButton("删除所选锚点")
        self.delete_button.clicked.connect(self._delete_selected_anchor)
        key_row.addWidget(self.add_button)
        key_row.addWidget(self.delete_button)
        key_layout.addLayout(key_row)
        hint = QLabel("单击时间轴移动游标；双击空白处添加锚点；拖动菱形锚点可移动自动/用户关键帧。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#7f94aa;font-size:11px;")
        key_layout.addWidget(hint)
        sidebar_layout.addWidget(key_group)

        action_group = QGroupBox("候选与提交")
        action_layout = QVBoxLayout(action_group)
        self.preview_button = QPushButton("重新生成候选预览")
        self.preview_button.clicked.connect(self._request_preview_now)
        self.commit_button = QPushButton("确认提交候选版本")
        self.commit_button.setObjectName("primary")
        self.commit_button.setEnabled(False)
        self.commit_button.clicked.connect(self._commit_preview)
        self.export_button = QPushButton("导出审计文件")
        self.export_button.clicked.connect(self._export)
        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.commit_button)
        action_layout.addWidget(self.export_button)
        sidebar_layout.addWidget(action_group)

        stats_group = QGroupBox("审计概览")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("—")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        stats_layout.addWidget(self.stats_label)
        sidebar_layout.addWidget(stats_group)

        status_group = QGroupBox("运行状态")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("—")
        self.status_label.setWordWrap(True)
        status_layout.addWidget(self.status_label)
        sidebar_layout.addWidget(status_group)
        sidebar_layout.addStretch(1)
        splitter.addWidget(sidebar)
        splitter.setSizes([1120, 370])
        root.addWidget(splitter, 1)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._commit_preview)
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self._request_preview_now)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._export)
        QShortcut(QKeySequence("Delete"), self, activated=self._delete_selected_anchor)

    def _set_busy(self, busy: bool, text: str = "") -> None:
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()
        if text:
            self.status_label.setText(text)
        QApplication.processEvents()

    def _open_service(self, frame_count: int, seed: int) -> None:
        if self.service is not None:
            try:
                self.service.close()
            except Exception:
                pass
            self.service = None
        self._set_busy(True, f"正在载入 {frame_count} 帧数据并恢复最近已提交状态…")
        try:
            service = WorkbenchService.open(self.workspace, frame_count, seed)
            self.service = service
            self.frame_count = frame_count
            self.seed = seed
            self._draft_thresholds = service.snapshot.thresholds
            self._draft_keyframes = set(service.snapshot.keyframes)
            self._draft_suppressed = set(service.snapshot.suppressed_auto)
            self._selected_anchor = None
            self._display_snapshot = service.snapshot
            self._display_is_candidate = False
            self._preview_result = None
            self._sync_threshold_controls()
            self._refresh_display(service.snapshot, candidate=False)
            recovery = service.recovery
            self._refresh_performance()
            self.preview_banner.setText(
                ("已恢复最近提交版本" if service.restored else "已建立基线提交")
                + f" · {'上次正常退出' if recovery.previous_exit_clean else ('检测到强制终止' if recovery.previous_session_present else '首次会话')}"
                + f" · 孤儿回收 {recovery.orphan_blobs_removed + recovery.orphan_staging_files_removed} 项"
            )
            if service.restore_stale:
                self.preview_banner.setText(
                    "最近提交与新算法版本不兼容，已记录 stale 原因并建立新基线"
                    + f" · {service.restore_stale.get('code')}"
                )
        except BusinessError as exc:
            self.preview_banner.setText(f"载入失败 · {exc.code.value}: {exc.message}")
            self.status_label.setText(f"{exc.code.value}: {exc.message}")
            print(f"mocap-workbench load error: {exc}", file=sys.stderr)
        finally:
            self._set_busy(False)

    def _load_selected_tier(self) -> None:
        frame_count = int(self.tier_combo.currentData())
        self._open_service(frame_count, self.seed)

    def _sync_threshold_controls(self) -> None:
        if self._draft_thresholds is None:
            return
        self._syncing_thresholds = True
        try:
            for key, spin in self.threshold_spins.items():
                spin.setValue(float(getattr(self._draft_thresholds, key)))
        finally:
            self._syncing_thresholds = False

    def _refresh_display(self, snapshot: Snapshot, candidate: bool) -> None:
        self._display_snapshot = snapshot
        self._display_is_candidate = candidate
        self.timeline.set_snapshot(snapshot, candidate=candidate)
        self.version_badge.setText(("CANDIDATE " if candidate else "COMMITTED ") + snapshot.version.label)
        self._refresh_eval_table(snapshot)
        self._refresh_detection_table(snapshot)
        self._refresh_closure(snapshot)
        self._refresh_stats(snapshot)
        self.commit_button.setEnabled(candidate)

    def _refresh_eval_table(self, snapshot: Snapshot) -> None:
        details = list(snapshot.evaluation.details)
        limit = 1200
        self.eval_table.setRowCount(min(limit, len(details)))
        for row, detail in enumerate(details[:limit]):
            entries = [
                str(detail.frame),
                detail.joint,
                detail.defect_type,
                "" if detail.joint_angle_error_deg is None else f"{detail.joint_angle_error_deg:.6f}",
                "" if detail.root_error_m is None else f"{detail.root_error_m:.9f}",
                "是" if detail.trusted else "否",
                str(detail.tolerance),
            ]
            for column, text in enumerate(entries):
                item = QTableWidgetItem(text)
                if column == 5 and not detail.trusted:
                    item.setForeground(QColor(COLOR_JUMP))
                self.eval_table.setItem(row, column, item)

    def _refresh_detection_table(self, snapshot: Snapshot) -> None:
        entries = snapshot.detection.entries
        self.detection_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = [str(entry.frame), entry.joint, entry.defect_type, str(entry.segment_id)]
            for column, text in enumerate(values):
                self.detection_table.setItem(row, column, QTableWidgetItem(text))

    def _refresh_closure(self, snapshot: Snapshot) -> None:
        closure = snapshot.closure
        consistency = snapshot.consistency
        data = {
            "version": snapshot.version.to_dict(),
            "thresholds": snapshot.thresholds.to_dict(),
            "keyframes": sorted(snapshot.keyframes),
            "suppressed_auto": sorted(snapshot.suppressed_auto),
            "closure": closure.to_dict() if closure else None,
            "consistency": consistency.to_dict() if consistency else None,
            "evaluation_hash": snapshot.evaluation.evaluation_hash,
            "repair_hash": snapshot.repair.repair_hash,
        }
        import json

        self.closure_text.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))

    def _refresh_stats(self, snapshot: Snapshot) -> None:
        comparison = snapshot.detection.comparison
        summary = snapshot.evaluation.summary
        self.stats_label.setText(
            f"检测：{comparison['detected_count']} / 真值 {comparison['truth_count']}；"
            f"漏检 {comparison['missed_count']}，误检 {comparison['false_positive_count']}\n"
            f"修复：最大角误差 {summary['max_joint_angle_error_deg']:.6f}°；"
            f"最大根误差 {summary['max_root_error_m']:.9f} m\n"
            f"平滑改善 {summary['smoothness_improvement_ratio']:.4%}；"
            f"状态 {summary['publication_status']}\n"
            f"关键帧 {len(snapshot.keyframes)} 个用户锚点；抑制自动边界 {len(snapshot.suppressed_auto)} 个"
        )

    def _refresh_performance(self) -> None:
        if self.service is None:
            return
        state = self.service.recorder.state
        last = state.get("last") or {}
        timings = last.get("frame_timings_ms", {})
        top = sorted(timings.items(), key=lambda item: item[1], reverse=True)[:20]
        import json

        self.performance_text.setPlainText(
            json.dumps(
                {
                    "interaction_budget_ms": PREVIEW_BUDGET_MS.get(self.frame_count),
                    "preview_count": state.get("preview_count", 0),
                    "over_budget_count": state.get("over_budget_count", 0),
                    "max_elapsed_ms": state.get("max_elapsed_ms", 0),
                    "last": last,
                    "top_frame_timings_ms": top,
                    "recovery": self.service.recovery.__dict__ if self.service.recovery else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def _on_threshold_changed(self, value: float) -> None:
        if self._syncing_thresholds or self.service is None:
            return
        from .model import Thresholds

        values = {key: spin.value() for key, spin in self.threshold_spins.items()}
        try:
            self._draft_thresholds = Thresholds(**values)
        except TypeError:
            return
        self._schedule_preview()

    def _on_frame_changed(self, frame: int) -> None:
        self.timeline.set_current_frame(frame)
        selected = f"锚点 {self._selected_anchor}" if self._selected_anchor is not None else "未选中锚点"
        self.anchor_label.setText(f"当前帧 {frame} · {selected}")

    def _on_anchor_selected(self, frame: int) -> None:
        self._selected_anchor = frame
        self.timeline.set_selected_anchor(frame)
        self.anchor_label.setText(f"当前帧 {self.timeline._current_frame} · 锚点 {frame}")

    def _on_anchor_move_preview(self, old_frame: int, new_frame: int) -> None:
        if self.service is None or self._display_snapshot is None:
            return
        if self._drag_origin is None or self._drag_origin[0] != old_frame:
            self._drag_origin = (old_frame, set(self._draft_keyframes), set(self._draft_suppressed))
        origin_old, base_keyframes, base_suppressed = self._drag_origin
        auto_frames = self._current_auto_frames()
        keyframes = set(base_keyframes)
        suppressed = set(base_suppressed)
        keyframes.discard(origin_old)
        if origin_old in auto_frames:
            suppressed.add(origin_old)
        keyframes.add(new_frame)
        self._draft_keyframes = keyframes
        self._draft_suppressed = suppressed
        self._schedule_preview()

    def _on_anchor_move_finished(self, old_frame: int, new_frame: int) -> None:
        self._drag_origin = None
        self._on_anchor_move_preview(old_frame, new_frame)

    def _current_auto_frames(self) -> set[int]:
        snapshot = self.service.snapshot if self.service else self._display_snapshot
        if snapshot is None:
            return set()
        frames: set[int] = set()
        for anchors in snapshot.anchors.joint_auto:
            frames.update(anchors)
        frames.update(snapshot.anchors.root_auto)
        return frames

    def _add_anchor_at(self, frame: int) -> None:
        if self.service is None:
            return
        snapshot = self.service.snapshot
        if snapshot.detection.joint_bad[frame, :].any() or snapshot.detection.root_bad[frame]:
            self._show_error(
                BusinessError(
                    ErrorCode.E_KEYFRAME_INVALID,
                    "不能把缺陷帧设为关键姿态锚点",
                    {"frame": frame},
                )
            )
            return
        if frame in self.service.active_anchor_frames() or frame in self._draft_keyframes:
            self._show_error(
                BusinessError(ErrorCode.E_KEYFRAME_INVALID, "该帧已有锚点", {"frame": frame})
            )
            return
        self._draft_keyframes.add(frame)
        self._selected_anchor = frame
        self.timeline.set_selected_anchor(frame)
        self._schedule_preview()

    def _delete_selected_anchor(self) -> None:
        if self.service is None or self._selected_anchor is None:
            return
        frame = self._selected_anchor
        auto_frames = self._current_auto_frames()
        self._draft_keyframes.discard(frame)
        if frame in auto_frames:
            self._draft_suppressed.add(frame)
        self._selected_anchor = None
        self.timeline.set_selected_anchor(None)
        self._schedule_preview()

    def _request_preview_now(self) -> None:
        self._preview_debounce.stop()
        self._launch_preview()

    def _schedule_preview(self) -> None:
        if self.service is None:
            return
        self.preview_button.setEnabled(False)
        self.preview_banner.setText("正在计算候选预览… 已提交视图保持不变")
        self._preview_debounce.start()

    def _launch_preview(self) -> None:
        if self.service is None or self._draft_thresholds is None:
            return
        self._pending_preview = True
        self._last_preview_spec = {
            "thresholds": self._draft_thresholds,
            "keyframes": frozenset(self._draft_keyframes),
            "suppressed_auto": frozenset(self._draft_suppressed),
        }
        try:
            self.service.manager.submit_preview(
                self.service.bundle.defects,
                thresholds=self._draft_thresholds,
                keyframes=frozenset(self._draft_keyframes),
                suppressed_auto=frozenset(self._draft_suppressed),
            )
            self._preview_poll.start()
        except BusinessError as exc:
            self._pending_preview = False
            self._show_error(exc)

    def _poll_preview(self) -> None:
        if self.service is None:
            return
        result = self.service.manager.poll()
        if result is None:
            error = self.service.manager.last_error
            if error:
                self._pending_preview = False
                self._preview_poll.stop()
                self.preview_button.setEnabled(True)
                self.preview_banner.setText(f"候选失败 · {error.get('code')}: {error.get('message')}")
                if error.get("code") == ErrorCode.E_BUDGET_EXCEEDED.value:
                    QMessageBox.warning(self, "交互预算超限", str(error))
            return
        self._pending_preview = False
        self._preview_poll.stop()
        self.preview_button.setEnabled(True)
        self._preview_result = result
        self._refresh_display(result.candidate, candidate=True)
        consistency = result.candidate.consistency
        consistent = bool(consistency and consistency.consistent)
        budget_text = (
            f"候选预览 {result.elapsed_ms:.1f} ms / 预算 {result.budget_ms} ms"
            + (" · 超预算" if result.over_budget else " · 预算内")
        )
        self.preview_banner.setText(
            budget_text
            + f" · 闭包帧 {len({f for f, _ in result.candidate.closure.dirty_joint_coords}) if result.candidate.closure else 0}"
            + f" · 全量一致 {'是' if consistent else '否'} · 尚未提交"
        )
        self.preview_banner.setStyleSheet(
            "background:#3a2c10;border:1px solid #a87917;border-radius:7px;padding:7px 10px;color:#ffd98a;"
            if result.over_budget or not consistent
            else "background:#102c27;border:1px solid #168b7e;border-radius:7px;padding:7px 10px;color:#9af0df;"
        )
        self.status_label.setText(
            f"候选版本 {result.candidate.version.label}；已提交版本保持为 {self.service.snapshot.version.label}"
        )
        self._refresh_performance()
        if result.over_budget:
            self._show_error(
                BusinessError(
                    ErrorCode.E_BUDGET_EXCEEDED,
                    "候选预览超出声明交互预算，已保留候选但标记超预算",
                    {"elapsed_ms": result.elapsed_ms, "budget_ms": result.budget_ms},
                )
            )

    def _commit_preview(self) -> None:
        if self.service is None or self._preview_result is None:
            return
        candidate = self._preview_result.candidate
        if candidate.consistency is not None and not candidate.consistency.consistent:
            self._show_error(
                BusinessError(
                    ErrorCode.E_CLOSURE_MISMATCH,
                    "候选增量与全量重算不一致，禁止提交",
                    candidate.consistency.first_mismatch or {},
                )
            )
            return
        self._set_busy(True, "正在原子提交候选版本…")
        try:
            committed = self.service.manager.commit_preview(self.service.store, self.service.bundle.defects)
            self._draft_thresholds = committed.thresholds
            self._draft_keyframes = set(committed.keyframes)
            self._draft_suppressed = set(committed.suppressed_auto)
            self._preview_result = None
            self._selected_anchor = None
            self.timeline.set_selected_anchor(None)
            self._refresh_display(committed, candidate=False)
            self.preview_banner.setText(f"已提交 {committed.version.label} · 快照已整体替换")
            self.preview_banner.setStyleSheet(
                "background:#10243a;border:1px solid #2b6e9f;border-radius:7px;padding:7px 10px;color:#a9d9ff;"
            )
            self.status_label.setText(f"提交成功；账本内容哈希 {committed.content_hash[:20]}…")
            self._refresh_performance()
        except BusinessError as exc:
            self._show_error(exc)
        finally:
            self._set_busy(False)

    def _export(self) -> None:
        if self.service is None:
            return
        default = str(self.workspace / "outputs" / f"{self.frame_count}_{self.service.snapshot.version.label}")
        directory = QFileDialog.getExistingDirectory(self, "选择导出目录", default)
        if not directory:
            return
        try:
            paths = self.service.export(Path(directory))
            self.status_label.setText(f"已导出：{paths['summary']}")
        except Exception as exc:
            self._show_error(exc)

    def _show_error(self, exc: Exception) -> None:
        if isinstance(exc, BusinessError):
            QMessageBox.critical(
                self,
                f"业务错误 {exc.code.value}",
                f"{exc.code.value}\n\n{exc.message}\n\n{exc.context}",
            )
        else:
            QMessageBox.critical(self, "运行错误", f"{type(exc).__name__}: {exc}")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.service is not None:
            try:
                self.service.close()
            except Exception as exc:
                QMessageBox.warning(self, "关闭清理失败", str(exc))
        event.accept()


def run_gui(workspace: Path, frame_count: int, seed: int = DEFAULT_SEED) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MocapClean Workbench")
    app.setApplicationVersion("1.0.0")
    app.setStyle("Fusion")
    available = set(QFontDatabase.families())
    family = next(
        (name for name in ("Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "Segoe UI") if name in available),
        app.font().family(),
    )
    font = QFont(family, 9)
    app.setFont(font)
    window = WorkbenchWindow(Path(workspace), frame_count, seed)
    window.show()
    smoke_ms = os.environ.get("MOCAP_GUI_SMOKE_MS")
    if smoke_ms:
        QTimer.singleShot(int(smoke_ms), app.quit)
    return int(app.exec())
