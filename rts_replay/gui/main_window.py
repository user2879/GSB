"""PySide6 interactive timeline replay window.

The GUI is a thin view over :class:`~rts_replay.player.ReplayPlayer`. A
:class:`QTimer` ticks at a fixed frame period; every timeout consumes one
player frame, so synthesis and screen refresh share the declared latency
budget. Jump actions rebuild from the nearest snapshot and only swap the view
after the whole target state is available, guaranteeing that no half-synthesized
state is ever displayed.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..events import load_events
from ..generator import PROFILES
from ..player import SPEED_MULTIPLIERS, ReplayPlayer
from ..rules import rules_from_dict
import json

FRAME_PERIOD_MS = 16
CURVE_COLORS = ("#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6")


class ResourceCurveWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(200)
        self.series: dict[str, list[int]] = {}
        self.caps: dict[str, int] = {}

    def update_series(self, player: ReplayPlayer) -> None:
        self.caps = {
            faction.faction: faction.resource_cap for faction in player.config.factions
        }
        self.series = {name: [] for name in player.config.faction_names()}
        for point in player.resource_history:
            for name in self.series:
                self.series[name].append(point.get(name, 0))
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        if not self.series or not any(self.series.values()):
            painter.end()
            return
        width = self.width()
        height = self.height()
        margin = 10
        max_cap = max(self.caps.values()) if self.caps else 1
        painter.drawText(4, 12, "resources per faction (cap line = rule cap)")
        for index, (name, values) in enumerate(sorted(self.series.items())):
            color = CURVE_COLORS[index % len(CURVE_COLORS)]
            pen = QPen(Qt.GlobalColor.black)
            pen.setColor(Qt.GlobalColor.black)
            painter.setPen(pen)
            painter.drawText(4, 28 + index * 14, f"{name} cap={self.caps.get(name)}")
            cap_y = height - margin - int(
                (height - 2 * margin) * self.caps.get(name, max_cap) / max(1, max_cap)
            )
            cap_pen = QPen(Qt.GlobalColor.lightGray)
            cap_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(cap_pen)
            painter.drawLine(0, cap_y, width, cap_y)
            line_pen = QPen()
            line_pen.setColor(Qt.GlobalColor.black)
            line_pen.setWidth(2)
            painter.setPen(line_pen)
            points = []
            for x_index, value in enumerate(values):
                x = margin + int(
                    (width - 2 * margin) * x_index / max(1, len(values) - 1)
                )
                y = height - margin - int(
                    (height - 2 * margin) * min(value, max_cap) / max(1, max_cap)
                )
                points.append((x, y))
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                painter.drawLine(x1, y1, x2, y2)
        painter.end()


class ReplayMainWindow(QMainWindow):
    def __init__(self, workspace_dir: str) -> None:
        super().__init__()
        self.workspace_dir = workspace_dir
        self.setWindowTitle("RTS Resource & Unit Replay Workbench")
        self.player: ReplayPlayer | None = None
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._on_frame)

        central = QWidget()
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.scenario_box = QComboBox()
        self.scenario_box.addItems([profile.name for profile in PROFILES])
        controls.addWidget(self.scenario_box)
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self._load_scenario)
        controls.addWidget(self.load_button)
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self._step_once)
        controls.addWidget(self.step_button)
        controls.addWidget(QLabel("Speed"))
        self.speed_box = QComboBox()
        self.speed_box.addItems([str(item) for item in SPEED_MULTIPLIERS])
        self.speed_box.currentTextChanged.connect(self._on_speed)
        controls.addWidget(self.speed_box)
        controls.addWidget(QLabel("Jump tick"))
        self.jump_spin = QSpinBox()
        self.jump_spin.setMinimum(0)
        controls.addWidget(self.jump_spin)
        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self._jump)
        controls.addWidget(self.jump_button)
        layout.addLayout(controls)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.sliderReleased.connect(self._jump_to_slider)
        layout.addWidget(self.slider)

        self.curve = ResourceCurveWidget()
        layout.addWidget(self.curve)
        self.status = QLabel("No scenario loaded")
        layout.addWidget(self.status)
        self.queue_label = QLabel("")
        self.queue_label.setWordWrap(True)
        layout.addWidget(self.queue_label)
        self.setCentralWidget(central)
        self.resize(900, 600)

    def _scenario_dir(self) -> str:
        return os.path.join(
            self.workspace_dir, "samples", self.scenario_box.currentText()
        )

    def _load_scenario(self) -> None:
        scenario_dir = self._scenario_dir()
        with open(os.path.join(scenario_dir, "rules.json"), encoding="utf-8") as handle:
            rules = rules_from_dict(json.loads(handle.read()))
        events = load_events(os.path.join(scenario_dir, "events.jsonl"))
        snapshot_dir = os.path.join(
            self.workspace_dir, "gui_snapshots", self.scenario_box.currentText()
        )
        self.player = ReplayPlayer(
            rules,
            events,
            snapshot_dir=snapshot_dir,
            budget_ms=FRAME_PERIOD_MS,
            base_ticks_per_frame=2,
        )
        self.slider.setMaximum(self.player.engine.last_tick)
        self.jump_spin.setMaximum(self.player.engine.last_tick)
        self.player.pause()
        self._refresh_view()

    def _toggle_play(self) -> None:
        if self.player is None:
            return
        if self.player.paused:
            self.player.resume()
            self.play_button.setText("Pause")
            self.timer.start(FRAME_PERIOD_MS)
        else:
            self.player.pause()
            self.play_button.setText("Play")
            self.timer.stop()

    def _step_once(self) -> None:
        if self.player is None:
            return
        self.player.pause()
        self.timer.stop()
        self.play_button.setText("Play")
        self.player.step_one()
        self._refresh_view()

    def _on_speed(self, text: str) -> None:
        if self.player is not None:
            self.player.set_speed(int(text))

    def _jump(self) -> None:
        if self.player is None:
            return
        self.player.jump_to(self.jump_spin.value())
        self._refresh_view()

    def _jump_to_slider(self) -> None:
        if self.player is None:
            return
        self.player.jump_to(self.slider.value())
        self._refresh_view()

    def _on_frame(self) -> None:
        if self.player is None:
            return
        self.player.run_frame()
        self._refresh_view()
        if self.player.finished:
            self.timer.stop()
            self.player.pause()
            self.play_button.setText("Play")

    def _refresh_view(self) -> None:
        assert self.player is not None
        self.slider.setValue(self.player.current_tick)
        self.curve.update_series(self.player)
        state = self.player.engine.state
        queue_text = []
        for name, faction in sorted(state.factions.items()):
            queue_text.append(
                f"{name}: res={faction.resources} idle={faction.idle_units} "
                f"deployed={faction.deployed_units} queued={len(faction.queue)} "
                f"active={faction.active_commands}"
            )
        self.queue_label.setText(" | ".join(queue_text))
        report = self.player.budget_report
        self.status.setText(
            f"tick={self.player.current_tick}/{self.player.engine.last_tick} "
            f"hash={state.state_hash()[:12]} speed={self.player.speed} "
            f"over_budget={report.over_budget_count} frames={len(report.frames)}"
        )


def run_gui(workspace_dir: str) -> int:
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = ReplayMainWindow(workspace_dir)
    window.show()
    return app.exec()
