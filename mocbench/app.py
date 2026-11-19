import json
import os
import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .engine import InterruptController, SolveEngine
from .errors import BenchError
from .parser import load_bundle
from .persistence import read_chunk

STATE_PLACEHOLDER = "PLACEHOLDER"
STATE_SOLVED = "SOLVED"


class SolveWorker(QThread):
    chunk_completed = Signal(int, list)
    failed = Signal(str, str)
    finished_ok = Signal(object)

    def __init__(self, bundle, state_dir):
        super().__init__()
        self.bundle = bundle
        self.state_dir = state_dir
        self.controller = InterruptController()
        self.engine = None

    def run(self):
        try:
            self.engine = SolveEngine(
                self.bundle,
                self.state_dir,
                controller=self.controller,
                progress=self._on_progress,
            )
            completed = self.engine.run()
            self.finished_ok.emit(completed)
        except BenchError as exc:
            self.failed.emit(exc.code, exc.message)
        except Exception as exc:
            self.failed.emit("E_SOLVE", str(exc))

    def _on_progress(self, event):
        if event.get("event") == "chunk_done":
            payload = read_chunk(self.state_dir, event["chunk"])
            self.chunk_completed.emit(event["chunk"], payload["frames"])

    def request_stop(self):
        import signal as _signal
        self.controller._handler(_signal.SIGINT, None)


class SkeletonCanvas(QWidget):
    def __init__(self, skeleton):
        super().__init__()
        self.skeleton = skeleton
        self.pose = None
        self.state = STATE_PLACEHOLDER
        self.setMinimumHeight(420)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("SkeletonCanvas{background-color:#14181f;}")

    def show_placeholder(self, frame_index):
        self.pose = None
        self.state = STATE_PLACEHOLDER
        self.frame_index = frame_index
        self.update()

    def show_pose(self, frame_index, world_positions, state=STATE_SOLVED):
        self.pose = {name: tuple(value) for name, value in world_positions.items()}
        self.state = state
        self.frame_index = frame_index
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        title = (
            f"{self.state} FRAME — {'PERSISTED POSE' if self.state == STATE_SOLVED else 'DETERMINISTIC PLACEHOLDER'}"
        )
        painter.setPen(QColor("#c8d3e0"))
        painter.drawText(18, 28, title)
        if self.pose is None:
            painter.setPen(QColor("#7d8aa0"))
            painter.drawText(18, self.height() // 2, f"frame {getattr(self, 'frame_index', 0)} not solved yet")
            return
        points = self._project()
        bone_pen = QPen(QColor("#4fd08b"), 3)
        painter.setPen(bone_pen)
        for name in self.skeleton.order:
            parent = self.skeleton.bones[name].parent
            if parent is None:
                continue
            painter.drawLine(*points[parent], *points[name])
        painter.setBrush(QBrush(QColor("#8fb8e8")))
        painter.setPen(QPen(QColor("#8fb8e8"), 2))
        for name, (x, y) in points.items():
            painter.drawEllipse(x - 4, y - 4, 8, 8)

    def _project(self):
        values = [self.pose[name] for name in self.skeleton.order]
        min_x = min(v[0] for v in values)
        max_x = max(v[0] for v in values)
        min_y = min(v[1] for v in values)
        max_y = max(v[1] for v in values)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        margin = 60
        width = self.width() - 2 * margin
        height = self.height() - 2 * margin
        scale = min(width / span_x, height / span_y)
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        projected = {}
        for name in self.skeleton.order:
            x, y, _z = self.pose[name]
            px = self.width() * 0.5 + (x - center_x) * scale
            py = self.height() * 0.5 - (y - center_y) * scale
            projected[name] = (int(px), int(py))
        return projected


class WorkbenchWindow(QWidget):
    def __init__(self, bundle, state_dir):
        super().__init__()
        self.bundle = bundle
        self.state_dir = state_dir
        self.skeleton = bundle.skeleton
        self.frame_count = bundle.curves.frame_count
        self.chunk_size = bundle.params.chunk_size
        self.solved_frames = {}
        self.frame_snapshot = {}
        self.chunk_status = {}
        self.setWindowTitle("MoCap Constraint Workbench")
        self._build_ui()
        self.worker = SolveWorker(bundle, state_dir)
        self.worker.chunk_completed.connect(self.on_chunk_completed)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.start()
        self.show_frame(0)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.canvas = SkeletonCanvas(self.skeleton)
        layout.addWidget(self.canvas)

        controls = QHBoxLayout()
        self.prev_button = QPushButton("Prev")
        self.next_button = QPushButton("Next")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.frame_count - 1)
        self.slider.valueChanged.connect(self.show_frame)
        self.status_label = QLabel()
        self.prev_button.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        self.next_button.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        controls.addWidget(self.prev_button)
        controls.addWidget(self.slider)
        controls.addWidget(self.next_button)
        controls.addWidget(self.status_label)
        layout.addLayout(controls)

        self.chunk_list = QListWidget()
        layout.addWidget(self.chunk_list)
        self._refresh_chunk_list()

    def chunk_of_frame(self, frame_index):
        return frame_index // self.chunk_size

    def show_frame(self, frame_index):
        frame_index = max(0, min(self.frame_count - 1, int(frame_index)))
        if frame_index in self.solved_frames:
            frame = self.solved_frames[frame_index]
            self.canvas.show_pose(frame_index, frame["world_positions"], STATE_SOLVED)
            self.frame_snapshot[frame_index] = json.dumps(frame, sort_keys=True)
            state = STATE_SOLVED
            iterations = frame["iterations_used"]
        else:
            self.canvas.show_placeholder(frame_index)
            state = STATE_PLACEHOLDER
            iterations = None
        self.status_label.setText(
            f"frame {frame_index}/{self.frame_count - 1} [{state}]"
            + (f" iters={iterations}" if iterations is not None else "")
        )
        if self.slider.value() != frame_index:
            self.slider.blockSignals(True)
            self.slider.setValue(frame_index)
            self.slider.blockSignals(False)

    def on_chunk_completed(self, chunk_index, frames):
        for frame in frames:
            frame_index = frame["frame"]
            self.solved_frames[frame_index] = frame
            self.chunk_status[chunk_index] = "SOLVED"
        self._refresh_chunk_list()
        current = self.slider.value()
        if current in self.solved_frames:
            self.show_frame(current)

    def on_finished(self, completed):
        for chunk_index in completed:
            self.chunk_status[chunk_index] = "SOLVED"
        self._refresh_chunk_list()
        self.show_frame(self.slider.value())

    def on_failed(self, code, message):
        self.status_label.setText(f"{code}: {message}")

    def _refresh_chunk_list(self):
        self.chunk_list.clear()
        for chunk_index, _start, ids in self._plans():
            status = self.chunk_status.get(chunk_index, "PENDING")
            label = f"chunk {chunk_index}: frames {ids[0]}-{ids[-1]} [{status}]"
            self.chunk_list.addItem(QListWidgetItem(label))

    def _plans(self):
        plans = []
        for start in range(0, self.frame_count, self.chunk_size):
            ids = list(range(start, min(start + self.chunk_size, self.frame_count)))
            plans.append((start // self.chunk_size, start, ids))
        return plans

    def export_ui_state(self, path):
        current = self.slider.value()
        display = {
            "frame": current,
            "state": self.canvas.state,
            "converged": self.solved_frames.get(current, {}).get("converged"),
            "iterations_used": self.solved_frames.get(current, {}).get("iterations_used"),
            "pose": self.solved_frames.get(current, {}).get("world_positions"),
        }
        payload = {
            "frame_count": self.frame_count,
            "current_frame": current,
            "cached_frames": sorted(self.solved_frames),
            "all_frames_immutable": self._immutable_check(),
            "display": display,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        return payload

    def _immutable_check(self):
        for frame_index, snapshot in self.frame_snapshot.items():
            if json.dumps(self.solved_frames[frame_index], sort_keys=True) != snapshot:
                return False
        return True

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(3000)
        event.accept()


def run_window(input_path, state_dir, screenshot_path=None, state_export=None):
    app = QApplication.instance() or QApplication(sys.argv)
    bundle = load_bundle(input_path)
    window = WorkbenchWindow(bundle, state_dir)
    window.resize(1100, 820)
    window.show()
    return app, window
