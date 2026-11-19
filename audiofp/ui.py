"""PySide6 workbench with an explicit portable fallback for headless hosts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import FingerprintConfig
from .errors import AudioWorkbenchError
from .library import Library

try:  # Imported lazily enough that the portable UI works on a bare host.
    from PySide6 import QtGui, QtWidgets
except Exception:  # pragma: no cover - exercised by portable-only hosts
    QtGui = QtWidgets = None


if QtWidgets is not None:

    class TimelineWidget(QtWidgets.QWidget):
        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self.results: list[dict[str, Any]] = []
            self.setMinimumHeight(170)

        def set_results(self, results: list[dict[str, Any]]) -> None:
            self.results = results
            self.update()

        def paintEvent(self, event: Any) -> None:
            painter = QtGui.QPainter(self)
            painter.fillRect(self.rect(), QtGui.QColor("#111820"))
            width = max(1, self.width() - 24)
            top = 26
            painter.setPen(QtGui.QColor("#8090a0"))
            painter.drawText(10, 18, "Aligned candidate timeline (start / duration)")
            for index, result in enumerate(self.results[:8]):
                y = top + index * 17
                painter.setPen(QtGui.QColor("#566573"))
                painter.drawLine(12, y + 7, 12 + width, y + 7)
                duration = max(0.1, float(result.get("alignment_duration_seconds", 0.1)))
                start = max(0.0, float(result.get("alignment_start_seconds", 0.0)))
                maximum = max(start + duration, 1.0)
                for candidate in self.results:
                    maximum = max(
                        maximum,
                        float(candidate.get("alignment_start_seconds", 0.0))
                        + float(candidate.get("alignment_duration_seconds", 0.0)),
                    )
                x = 12 + int(width * (start / maximum))
                bar_width = max(3, int(width * (duration / maximum)))
                color = QtGui.QColor("#55c2a5") if index == 0 else QtGui.QColor("#3b7ea1")
                painter.fillRect(x, y, min(bar_width, 12 + width - x), 10, color)
                painter.setPen(QtGui.QColor("#d8e1e8"))
                painter.drawText(
                    16,
                    y - 2,
                    f"#{result.get('rank', index + 1)} {result.get('display_label', '')} "
                    f"score={float(result.get('score', 0.0)):.3f} start={start:.2f}s",
                )

    class WorkbenchWindow(QtWidgets.QMainWindow):
        def __init__(self, index_path: str, config: FingerprintConfig) -> None:
            super().__init__()
            self.index_path = index_path
            self.config = config
            self.results: list[dict[str, Any]] = []
            self.setWindowTitle("Audio Fingerprint Similar-Segment Workbench")
            self.resize(1120, 760)

            central = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(central)
            self.setCentralWidget(central)
            title = QtWidgets.QLabel("曲库查重 · 音频指纹与相似片段检索")
            title.setStyleSheet("font-size: 20px; font-weight: 600; padding: 4px;")
            layout.addWidget(title)

            controls = QtWidgets.QHBoxLayout()
            self.query_edit = QtWidgets.QLineEdit()
            self.query_edit.setPlaceholderText("选择本地查询片段 (.wav/.ogg/.mp3/.flac)")
            browse = QtWidgets.QPushButton("浏览")
            browse.clicked.connect(self.browse_query)
            self.top_k = QtWidgets.QSpinBox()
            self.top_k.setRange(1, 50)
            self.top_k.setValue(10)
            search = QtWidgets.QPushButton("检索")
            search.clicked.connect(self.search)
            controls.addWidget(QtWidgets.QLabel("查询片段"))
            controls.addWidget(self.query_edit, 1)
            controls.addWidget(browse)
            controls.addWidget(QtWidgets.QLabel("Top K"))
            controls.addWidget(self.top_k)
            controls.addWidget(search)
            layout.addLayout(controls)

            self.table = QtWidgets.QTableWidget(0, 6)
            self.table.setHorizontalHeaderLabels(
                ["Rank", "Logical track", "Score", "Align start (s)", "Duration (s)", "Formats"]
            )
            self.table.horizontalHeader().setStretchLastSection(True)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.itemSelectionChanged.connect(self.explain_selected)
            layout.addWidget(self.table, 3)

            self.timeline = TimelineWidget()
            layout.addWidget(self.timeline, 2)
            self.explanation = QtWidgets.QPlainTextEdit()
            self.explanation.setReadOnly(True)
            self.explanation.setMinimumHeight(120)
            layout.addWidget(self.explanation, 2)
            self.status = QtWidgets.QLabel("Ready · offline local index")
            layout.addWidget(self.status)

            if not Path(self.index_path).is_file():
                self.status.setText(
                    f"Index not found: {self.index_path}. Run `python -m audiofp demo` first."
                )

        def browse_query(self) -> None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select query audio",
                str(Path.home()),
                "Audio files (*.wav *.ogg *.mp3 *.flac *.opus);;All files (*)",
            )
            if path:
                self.query_edit.setText(path)

        def search(self) -> None:
            query_path = self.query_edit.text().strip()
            if not query_path:
                self.status.setText("Please select a query fragment first.")
                return
            try:
                with Library(self.index_path, self.config) as library:
                    payload = library.query_file(query_path, top_k=self.top_k.value())
            except AudioWorkbenchError as exc:
                self._show_error(exc)
                return
            except Exception as exc:  # GUI boundary: keep the window alive.
                self.status.setText(f"Search failed: {exc}")
                return
            self.results = payload["results"]
            self.table.setRowCount(len(self.results))
            for row, result in enumerate(self.results):
                values = [
                    str(result["rank"]),
                    result["display_label"],
                    f"{result['score']:.6f}",
                    f"{result['alignment_start_seconds']:.6f}",
                    f"{result['alignment_duration_seconds']:.6f}",
                    ", ".join(source["format_signature"] for source in result["formats"]),
                ]
                for column, value in enumerate(values):
                    self.table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            self.timeline.set_results(self.results)
            self.status.setText(
                f"Query: {Path(query_path).name} · {len(self.results)} logical candidates · "
                f"variant search={payload.get('variant_search', {}).get('triggered', False)}"
            )
            if self.results:
                self.table.selectRow(0)
                self.explain_result(self.results[0])

        def explain_selected(self) -> None:
            rows = self.table.selectionModel().selectedRows()
            if rows and rows[0].row() < len(self.results):
                self.explain_result(self.results[rows[0].row()])

        def explain_result(self, result: dict[str, Any]) -> None:
            formats = "\n".join(
                f"  - {source['format_signature']}  {source['relative_path']}  "
                f"sha256={source['content_sha256'][:12]}…"
                for source in result["formats"]
            )
            transform = result.get("transform")
            text = (
                f"{result['display_label']} ({result['group_id']})\n"
                f"score={result['score']:.6f}  votes={result['votes']}  "
                f"aligned_start={result['alignment_start_seconds']:.6f}s  "
                f"duration={result['alignment_duration_seconds']:.6f}s\n"
                f"score formula: {result['score_explanation']['formula']}\n"
                f"variant hypothesis: {transform or 'none (exact query space)'}\n"
                f"top vote bins: {json.dumps(result['vote_histogram'][:6], ensure_ascii=False)}\n"
                f"format sources:\n{formats}\n"
            )
            self.explanation.setPlainText(text)

        def _show_error(self, exc: AudioWorkbenchError) -> None:
            self.status.setText(f"{exc.code}: {exc.message}")
            self.explanation.setPlainText(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))


def _portable_notice(reason: str) -> None:
    print(
        "[PORTABLE] 图形后端不可用或未请求窗口，已进入便携终端模式；"
        f"原因: {reason}",
        file=sys.stderr,
    )
    print(
        "PORTABLE_MODE: use `list`, `query <path> [top_k]`, `help`, `quit`.",
        file=sys.stderr,
    )


def portable_session(
    index_path: str,
    config: FingerprintConfig,
    *,
    reason: str,
    commands: list[str] | None = None,
) -> int:
    _portable_notice(reason)
    library: Library | None = None
    try:
        library = Library(index_path, config)
        groups = library.store.list_groups()
        print(
            json.dumps(
                {
                    "mode": "portable",
                    "index": Path(index_path).name,
                    "groups": len(groups),
                    "sources": sum(len(group["sources"]) for group in groups),
                    "tracks": [group["display_label"] for group in groups],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except Exception as exc:
        print(json.dumps({"mode": "portable", "index_error": str(exc)}, ensure_ascii=False))

    def execute(command: str) -> bool:
        nonlocal library
        parts = command.strip().split()
        if not parts:
            return True
        if parts[0] in {"quit", "exit"}:
            return False
        if parts[0] == "help":
            print("list | query <audio-path> [top_k] | quit")
            return True
        if parts[0] == "list":
            if library is None:
                print("index unavailable")
            else:
                for group in library.store.list_groups():
                    formats = ", ".join(
                        source["format_signature"] for source in group["sources"]
                    )
                    print(f"{group['group_id']}  {group['display_label']}  [{formats}]")
            return True
        if parts[0] == "query":
            if len(parts) < 2 or library is None:
                print("usage: query <audio-path> [top_k]")
                return True
            top_k = int(parts[2]) if len(parts) > 2 else 10
            try:
                payload = library.query_file(parts[1], top_k=top_k)
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            except Exception as exc:
                if isinstance(exc, AudioWorkbenchError):
                    print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(f"query failed: {exc}")
            return True
        print(f"unknown command: {parts[0]}; type `help`")
        return True

    try:
        if commands is not None:
            for command in commands:
                if not execute(command):
                    break
        elif sys.stdin.isatty():
            while True:
                try:
                    command = input("audiofp> ")
                except EOFError:
                    break
                if not execute(command):
                    break
        else:
            print("Non-interactive stdin: portable session initialized; exiting.")
    finally:
        if library is not None:
            library.close()
    return 0


def main(
    index_path: str,
    config: FingerprintConfig,
    *,
    portable: bool = False,
    smoke: bool = False,
) -> int:
    if portable:
        return portable_session(index_path, config, reason="--portable requested")
    if QtWidgets is None:
        return portable_session(
            index_path,
            config,
            reason="PySide6 import failed",
        )
    has_display = (
        bool(os.environ.get("DISPLAY"))
        or bool(os.environ.get("WAYLAND_DISPLAY"))
        or sys.platform.startswith("win")
        or sys.platform == "darwin"
    )
    if smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    elif not has_display:
        return portable_session(
            index_path,
            config,
            reason="no DISPLAY/WAYLAND_DISPLAY on this host",
        )
    try:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        window = WorkbenchWindow(index_path, config)
        if smoke:
            window.show()
            app.processEvents()
            print(
                json.dumps(
                    {
                        "ok": True,
                        "mode": "qt-offscreen",
                        "platform": app.platformName(),
                        "index_exists": Path(index_path).is_file(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            window.close()
            app.processEvents()
            return 0
        window.show()
        return int(app.exec())
    except Exception as exc:
        if smoke:
            print(json.dumps({"ok": False, "mode": "qt-offscreen", "error": str(exc)}))
            return 1
        return portable_session(
            index_path,
            config,
            reason=f"Qt initialization failed: {exc}",
        )
