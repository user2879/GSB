"""PySide6 workbench UI with immediate drag preview and committed-state guard.

The widget renders lanes from the current immutable snapshot only. A mouse
drag updates a :class:`DragPreview` candidate (drawn hatched, never written to
the snapshot); Enter commits a new snapshot, Escape cancels and restores the
pre-drag geometry. Rejected previews paint red and expose the business error
code through :pyattr:`last_error`, while the committed layout stays untouched.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from .controller import DragPreview, WorkbenchController
from .errors import ErrorRecord
from .layout import LayoutConfig
LANE_LABEL_W = 110
HEADER_H = 40
FRAME_SCALE = 0.5      # px per frame -> 2700 frames = 1350 px wide
LANE_PX = 64


class TimelineWidget(QtWidgets.QWidget):
    previewChanged = QtCore.Signal(object)

    def __init__(self, controller: WorkbenchController,
                 config: LayoutConfig | None = None):
        super().__init__()
        self.controller = controller
        self.config = config or LayoutConfig()
        self.total_frames = 2700
        self.setMinimumSize(int(LANE_LABEL_W + self.total_frames * FRAME_SCALE),
                            HEADER_H + self.config.total_lanes * LANE_PX + 60)
        self.setMouseTracking(True)
        self._drag_item_id: str | None = None
        self._preview: DragPreview | None = None
        self.last_error: ErrorRecord | None = None
        self._drag_mode: str | None = None  # 'lane' | 'start' | 'end'
        self._press_x = 0

    # ------------------------------------------------------------------
    def _lane_rect(self, lane_index: int) -> QtCore.QRect:
        return QtCore.QRect(LANE_LABEL_W, HEADER_H + lane_index * LANE_PX,
                            self.width() - LANE_LABEL_W - 10, LANE_PX - 4)

    def _frame_to_x(self, frame: int) -> int:
        return LANE_LABEL_W + int(frame * FRAME_SCALE)

    def _x_to_frame(self, x: int) -> int:
        return max(0, int((x - LANE_LABEL_W) / FRAME_SCALE))

    def _lane_at_y(self, y: int) -> int:
        return (y - HEADER_H) // LANE_PX

    def _cue_at(self, pos: QtCore.QPoint) -> tuple[str, str] | None:
        snapshot = self.controller.store.load_current()
        if snapshot is None:
            return None
        for cue in snapshot.layout.cues:
            rect = QtCore.QRect(
                self._frame_to_x(cue.item.start_frame),
                HEADER_H + cue.lane_index * LANE_PX + 4,
                max(8, int((cue.item.end_frame - cue.item.start_frame)
                           * FRAME_SCALE)),
                LANE_PX - 12,
            )
            if rect.contains(pos):
                edge = "start" if pos.x() - rect.left() <= 12 else (
                    "end" if rect.right() - pos.x() <= 12 else "lane")
                return cue.item.item_id, edge
        return None

    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        hit = self._cue_at(event.position().toPoint())
        if hit is None:
            return
        item_id, mode = hit
        self._drag_item_id = item_id
        self._drag_mode = mode
        self._press_x = event.position().x()
        self.controller.begin_drag(item_id)
        self.last_error = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_item_id is None:
            return
        snapshot = self.controller.store.load_current()
        cue = next(c for c in snapshot.layout.cues
                   if c.item.item_id == self._drag_item_id)
        item = cue.item
        target_lane = cue.lane_index
        new_start, new_end = item.start_frame, item.end_frame
        if self._drag_mode == "lane":
            target_lane = int(self._lane_at_y(event.position().y()))
        elif self._drag_mode == "start":
            new_start = min(new_end - 1, self._x_to_frame(event.position().x()))
        else:
            new_end = max(new_start + 1, self._x_to_frame(event.position().x()))
        self._preview = self.controller.preview_drag(
            self._drag_item_id, target_lane, new_start, new_end)
        self.last_error = self._preview.error
        self.previewChanged.emit(self._preview)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        # Release only ends the gesture; commit/cancel are explicit keys so
        # the candidate state stays observable within the interaction budget.
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._drag_item_id is None:
            return
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            self.commit_drag()
        elif event.key() == QtCore.Qt.Key_Escape:
            self.cancel_drag()

    def commit_drag(self) -> bool:
        if self._preview is None or not self._preview.accepted:
            return False
        _, snapshot = self.controller.commit_drag()
        self._reset_gesture()
        self.update()
        return snapshot is not None

    def cancel_drag(self) -> None:
        if self._drag_item_id is not None:
            self.controller.cancel_drag()
        self._reset_gesture()
        self.update()

    def undo(self) -> bool:
        versions = self.controller.store.history()
        if len(versions) < 2:
            return False
        target = versions[-2]
        self.controller.undo(target)
        self._reset_gesture()
        self.update()
        return True

    def _reset_gesture(self) -> None:
        self._drag_item_id = None
        self._preview = None
        self._drag_mode = None
        self.last_error = None

    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(24, 26, 31))
        snapshot = self.controller.store.load_current()
        self._paint_lanes(painter)
        if snapshot is None:
            painter.setPen(QtGui.QColor(220, 220, 220))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             "No committed snapshot")
            return
        for cue in snapshot.layout.cues:
            if cue.item.item_id == self._drag_item_id:
                continue
            self._paint_cue(painter, cue, preview=None)
        if self._preview is not None:
            self._paint_preview(painter, snapshot)

    def _paint_lanes(self, painter: QtGui.QPainter) -> None:
        font = QtGui.QFont("sans-serif", 9)
        painter.setFont(font)
        for lane in range(self.config.total_lanes):
            rect = self._lane_rect(lane)
            is_ad = lane < self.config.ad_lanes
            painter.fillRect(rect, QtGui.QColor(
                42, 38, 28) if is_ad else QtGui.QColor(32, 34, 40))
            painter.setPen(QtGui.QColor(150, 150, 150))
            painter.drawText(8, rect.top() + rect.height() // 2 + 4,
                             ("AD" if is_ad else "DLG") + f" {lane}")
        painter.setPen(QtGui.QColor(110, 110, 110))
        for second in range(0, 91, 10):
            x = self._frame_to_x(second * 30)
            painter.drawLine(x, HEADER_H - 12, x, self.height() - 20)
            painter.drawText(x + 2, HEADER_H - 16, f"{second}s")

    def _paint_cue(self, painter, cue, preview: DragPreview | None) -> None:
        lane = cue.lane_index if preview is None else preview.target_lane
        start = cue.item.start_frame if preview is None else \
            preview.candidate.start_frame
        end = cue.item.end_frame if preview is None else \
            preview.candidate.end_frame
        rect = QtCore.QRect(
            self._frame_to_x(start), HEADER_H + lane * LANE_PX + 4,
            max(8, int((end - start) * FRAME_SCALE)), LANE_PX - 12)
        is_ad = cue.item.is_ad
        base = QtGui.QColor(196, 166, 60) if is_ad else QtGui.QColor(96, 150,
                                                                      210)
        if preview is not None and not preview.accepted:
            base = QtGui.QColor(200, 70, 70)
        painter.setPen(QtGui.QPen(base, 2 if preview else 1))
        painter.setBrush(QtGui.QColor(base.red(), base.green(),
                                      base.blue(), 90 if preview else 180))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QtGui.QColor(235, 235, 235))
        painter.drawText(rect.adjusted(6, 2, -6, -2),
                         QtCore.Qt.AlignVCenter | QtCore.Qt.TextWordWrap,
                         cue.item.item_id + " " + cue.wrap.lines[0])

    def _paint_preview(self, painter, snapshot) -> None:
        cue = next(c for c in snapshot.layout.cues
                   if c.item.item_id == self._drag_item_id)
        self._paint_cue(painter, cue, self._preview)
        if self.last_error is not None:
            painter.setPen(QtGui.QColor(255, 110, 110))
            painter.drawText(
                LANE_LABEL_W + 8, self.height() - 8,
                f"{self.last_error.code} "
                f"(conflict={self.last_error.conflict_item_id})")
        else:
            painter.setPen(QtGui.QColor(120, 220, 140))
            painter.drawText(LANE_LABEL_W + 8, self.height() - 8,
                             "preview candidate - Enter commits, Esc cancels")


class WorkbenchWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: WorkbenchController):
        super().__init__()
        self.setWindowTitle("Subtitle Layout & Accessibility Workbench")
        self.timeline = TimelineWidget(controller)
        self.status = QtWidgets.QLabel("Ready")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.timeline)
        layout.addWidget(self.status)
        host = QtWidgets.QWidget()
        host.setLayout(layout)
        self.setCentralWidget(host)
        self.timeline.previewChanged.connect(self._on_preview)
        undo_btn = QtWidgets.QPushButton("Undo last commit")
        undo_btn.clicked.connect(self.timeline.undo)
        toolbar = QtWidgets.QToolBar()
        toolbar.addWidget(undo_btn)
        self.addToolBar(toolbar)

    def _on_preview(self, preview: DragPreview) -> None:
        if preview.error is None:
            self.status.setText(
                f"Preview {preview.item_id} -> lane {preview.target_lane} "
                f"[{preview.candidate.start_frame},{preview.candidate.end_frame})")
        else:
            self.status.setText(
                f"Rejected: {preview.error.code} - {preview.error.message}")
