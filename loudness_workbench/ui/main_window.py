"""PySide6 workbench window.

The UI is a thin, deterministic view over immutable snapshots. It never
mutates measurement state directly: playback, block-length changes, mode
switching and re-measurement are delegated to :class:`PlaybackEngine`, and
every label/table is refreshed exclusively from the latest committed
:class:`Snapshot`.

Set ``QT_QPA_PLATFORM=offscreen`` for headless demonstrations; no physical
audio device is opened because playback is driven by the scripted material
buffer.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..config import (
    MAX_BLOCK_MS,
    MIN_BLOCK_MS,
    WorkbenchConfig,
    ms_to_samples,
    validate_block_ms,
)
from ..engine import PlaybackEngine
from ..errors import WorkbenchError
from ..snapshot import Snapshot


class WorkbenchWindow(QtWidgets.QMainWindow):
    def __init__(self, engine: PlaybackEngine, signal: np.ndarray,
                 segment_count: int = 6) -> None:
        super().__init__()
        self.engine = engine
        self.signal = signal
        self.segment_count = segment_count
        self.setWindowTitle("Broadcast Loudness & Dynamic-Range Workbench")
        self.resize(960, 720)
        self._build_widgets()
        self._refresh_from_snapshot(engine.store.current)

    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        mode_row = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["realtime", "offline"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(QtWidgets.QLabel("Mode"))
        mode_row.addWidget(self.mode_combo)

        self.block_spin = QtWidgets.QSpinBox()
        self.block_spin.setRange(MIN_BLOCK_MS, MAX_BLOCK_MS)
        self.block_spin.setSingleStep(10)
        self.block_spin.setValue(self.engine.block_ms)
        self.block_spin.editingFinished.connect(self._on_block_changed)
        mode_row.addWidget(QtWidgets.QLabel("Block (ms)"))
        mode_row.addWidget(self.block_spin)
        layout.addLayout(mode_row)

        target_row = QtWidgets.QHBoxLayout()
        self.segment_spin = QtWidgets.QSpinBox()
        self.segment_spin.setRange(0, max(self.segment_count - 1, 0))
        self.target_spin = QtWidgets.QDoubleSpinBox()
        self.target_spin.setRange(-80.0, 0.0)
        self.target_spin.setDecimals(1)
        self.target_spin.setSingleStep(0.5)
        self.target_spin.setValue(-23.0)
        self.adjust_button = QtWidgets.QPushButton("Adjust segment target")
        self.adjust_button.clicked.connect(self._on_target_adjusted)
        target_row.addWidget(QtWidgets.QLabel("Segment"))
        target_row.addWidget(self.segment_spin)
        target_row.addWidget(QtWidgets.QLabel("Target LKFS"))
        target_row.addWidget(self.target_spin)
        target_row.addWidget(self.adjust_button)
        layout.addLayout(target_row)

        progress_row = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, self.signal.shape[0])
        self.play_button = QtWidgets.QPushButton("Play one block")
        self.play_button.clicked.connect(self.play_one_block)
        self.rewind_button = QtWidgets.QPushButton("Rewind")
        self.rewind_button.clicked.connect(self.rewind)
        self.remeasure_button = QtWidgets.QPushButton("Re-measure")
        self.remeasure_button.clicked.connect(self.remeasure)
        progress_row.addWidget(self.progress)
        progress_row.addWidget(self.play_button)
        progress_row.addWidget(self.rewind_button)
        progress_row.addWidget(self.remeasure_button)
        layout.addLayout(progress_row)

        self.loudness_label = QtWidgets.QLabel()
        self.range_label = QtWidgets.QLabel()
        self.peak_label = QtWidgets.QLabel()
        self.gain_label = QtWidgets.QLabel()
        for widget in (self.loudness_label, self.range_label,
                       self.peak_label, self.gain_label):
            widget.setStyleSheet("font-family: monospace; font-size: 14px;")
            layout.addWidget(widget)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["version", "mode", "coverage end (s)", "integrated LKFS", "reason"]
        )
        layout.addWidget(self.table)

        self.status = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status)

    # ------------------------------------------------------------------
    @QtCore.Slot()
    def play_one_block(self) -> None:
        position = self.engine._position
        if position >= self.signal.shape[0]:
            self.status.setText("End of material reached")
            return
        size = ms_to_samples(self.engine.block_ms, self.engine.config.sample_rate)
        nxt = min(size, self.signal.shape[0] - position)
        try:
            _, snapshot = self.engine.process_block(self.signal[position : position + nxt])
        except WorkbenchError as exc:
            self.status.setText(f"{exc.code}: {exc.detail}")
            return
        self._refresh_from_snapshot(snapshot)

    @QtCore.Slot()
    def rewind(self) -> None:
        # Rewind only moves the playback cursor; measurement state is preserved
        # to prove re-entry never resets the compressor / meter.
        self.engine._position = 0
        self.status.setText("Rewound (measurement state preserved)")
        self.progress.setValue(0)

    @QtCore.Slot()
    def remeasure(self) -> None:
        try:
            snapshot = self.engine.remeasure_from_snapshot()
        except WorkbenchError as exc:
            self.status.setText(f"{exc.code}: {exc.detail}")
            return
        self._refresh_from_snapshot(snapshot)

    def _on_mode_changed(self, text: str) -> None:
        try:
            snapshot = self.engine.set_mode(text)
        except WorkbenchError as exc:
            self.status.setText(f"{exc.code}: {exc.detail}")
            return
        self._refresh_from_snapshot(snapshot)

    def _on_block_changed(self) -> None:
        try:
            validate_block_ms(self.block_spin.value())
            self.engine.set_block_ms(self.block_spin.value())
        except WorkbenchError as exc:
            self.status.setText(f"{exc.code}: {exc.detail}")
            return
        self._refresh_from_snapshot(self.engine.store.current)

    def _on_target_adjusted(self) -> None:
        try:
            snapshot = self.engine.adjust_segment_target(
                self.segment_spin.value(), self.target_spin.value()
            )
        except WorkbenchError as exc:
            self.status.setText(f"{exc.code}: {exc.detail}")
            return
        self._refresh_from_snapshot(snapshot)

    # ------------------------------------------------------------------
    def _refresh_from_snapshot(self, snapshot: Snapshot | None) -> None:
        if snapshot is None:
            return
        payload = snapshot.payload
        self.progress.setValue(snapshot.end_sample)
        self.loudness_label.setText(
            f"Integrated loudness : {_fmt(payload['integrated_lkfs'])} LKFS"
        )
        self.range_label.setText(
            f"Loudness range      : {_fmt(payload['loudness_range_lu'])} LU"
        )
        self.peak_label.setText(
            f"True peak           : {_fmt(payload['true_peak_dbtp'])} dBTP"
        )
        self.gain_label.setText(
            f"Compressor gain     : {payload['compressor_gain_db']:.4f} dB"
        )
        self.status.setText(
            f"Snapshot v{snapshot.version} | {payload['reason']} | "
            f"coverage {snapshot.coverage_s[0]:.3f}-{snapshot.coverage_s[1]:.3f}s | "
            f"degraded frames {payload['budget']['frames_degraded']}"
        )
        history = self.engine.store.history()
        self.table.setRowCount(len(history))
        for row, item in enumerate(history):
            values = [
                str(item.version),
                item.mode,
                f"{item.coverage_s[1]:.3f}",
                _fmt(item.payload["integrated_lkfs"]),
                item.payload["reason"],
            ]
            for col, value in enumerate(values):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(value))


def _fmt(value: float) -> str:
    if value is None or np.isneginf(value) or np.isnan(value):
        return "-inf"
    return f"{value:.3f}"


def run_window(config: WorkbenchConfig | None = None) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ..synth import DEFAULT_PROGRAMME, generate_programme

    cfg = config or WorkbenchConfig()
    signal, _ = generate_programme(cfg.seed)
    engine = PlaybackEngine(cfg)
    window = WorkbenchWindow(engine, signal, len(DEFAULT_PROGRAMME))
    window.show()
    return app.exec()
