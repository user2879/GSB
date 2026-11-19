"""PyQt5 工作台界面：展示响度曲线、逐窗口报告与预算/驻留使用情况。

界面只读取演示产出的确定性 JSON，不参与任何测量数学，因此线程数与 GUI
状态不会影响报告内容。无显示环境时可用 QT_QPA_PLATFORM=offscreen 启动自检。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class LoudnessCurve(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._windows: List[Dict[str, Any]] = []
        self.setMinimumHeight(260)

    def set_windows(self, windows: List[Dict[str, Any]]) -> None:
        self._windows = windows
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor("#101418"))
        if not self._windows:
            painter.setPen(QColor("#8899aa"))
            painter.drawText(self.rect(), Qt.AlignCenter, "加载运行报告后显示响度曲线")
            return

        vals = [win["short_lufs"] for win in self._windows if win["short_lufs"] is not None]
        lo, hi = min(vals), max(vals)
        pad = 12.0
        span = max(hi - lo, 1.0)

        def x(i: int) -> float:
            return pad + (w - 2 * pad) * i / max(len(self._windows) - 1, 1)

        def y(v: float) -> float:
            return h - pad - (h - 2 * pad) * (v - lo) / span

        painter.setPen(QPen(QColor("#334455"), 1))
        for gy in range(4):
            yy = pad + (h - 2 * pad) * gy / 3
            painter.drawLine(int(pad), int(yy), int(w - pad), int(yy))

        pen = QPen(QColor("#4fc3f7"), 2)
        painter.setPen(pen)
        prev = None
        for i, win in enumerate(self._windows):
            if win["short_lufs"] is None:
                prev = None
                continue
            pt = (x(i), y(win["short_lufs"]))
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(pt[0]), int(pt[1]))
            prev = pt

        pen_tp = QPen(QColor("#ffb74d"), 2, Qt.DashLine)
        painter.setPen(pen_tp)
        prev = None
        tpv = [win["true_peak_db"] for win in self._windows if win["true_peak_db"] is not None]
        tlo, thi = min(tpv), max(tpv)
        tspan = max(thi - tlo, 1.0)

        def ty(v: float) -> float:
            return h - pad - (h - 2 * pad) * (v - tlo) / tspan

        for i, win in enumerate(self._windows):
            if win["true_peak_db"] is None:
                prev = None
                continue
            pt = (x(i), ty(win["true_peak_db"]))
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(pt[0]), int(pt[1]))
            prev = pt

        painter.setPen(QColor("#ffffff"))
        painter.drawText(8, 18, f"短时响度 {lo:.1f}~{hi:.1f} LUFS（蓝）  真峰值 {tlo:.1f}~{thi:.1f} dBTP（橙虚线）")


class MainWindow(QMainWindow):
    def __init__(self, default_report: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("响度与动态范围并行测量工作台")
        self.resize(1180, 760)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        bar = QHBoxLayout()
        self.btn_open = QPushButton("打开运行报告…")
        self.btn_open.clicked.connect(self.choose_report)
        self.lbl_file = QLabel("未加载")
        bar.addWidget(self.btn_open)
        bar.addWidget(self.lbl_file, 1)
        layout.addLayout(bar)

        self.curve = LoudnessCurve()
        layout.addWidget(self.curve, stretch=3)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["任务", "起点样本", "短时LUFS", "门控LUFS", "真峰值dBTP", "命中", "重算"]
        )
        layout.addWidget(self.table, stretch=3)

        self.budget = QTextEdit()
        self.budget.setReadOnly(True)
        self.budget.setMaximumHeight(150)
        layout.addWidget(self.budget, stretch=1)

        if default_report and Path(default_report).exists():
            self.load_report(default_report)

    def choose_report(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择运行报告", "", "JSON 报告 (*.json)"
        )
        if path:
            self.load_report(path)

    def load_report(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merged = data["merged"]
        windows = merged["windows"]
        self.lbl_file.setText(path)
        self.curve.set_windows(windows)

        self.table.setRowCount(len(windows))
        for row, win in enumerate(windows):
            vals = [
                win["task_key"],
                str(win["start_sample"]),
                _fmt(win["short_lufs"]),
                _fmt(win["gated_rel_lufs"]),
                _fmt(win["true_peak_db"]),
                str(win["cache_hits"]),
                str(win["recomputes"]),
            ]
            for col, value in enumerate(vals):
                self.table.setItem(row, col, QTableWidgetItem(value))

        stats = data["cache_stats"]
        whole = merged["whole_program"]
        comp = merged.get("reference_comparison", {})
        lines = [
            f"工作线程: 请求 {data['requested_workers']} -> 实际 {data['effective_workers']}",
            f"驻留统计: STORE {stats['stores']}  HIT {stats['hits']}  RECOMPUTE {stats['recomputes']}  EVICT {stats['evictions']}  DENIED {stats['denied_buffers']}",
            f"整段门控响度: {_fmt(whole['rel_gated_lufs'])} LUFS  真峰值: {_fmt(whole['true_peak_db'])} dBTP  LRA: {_fmt(whole['lra_lu'])} LU",
            f"降级动作: {json.dumps(data['degradation'], ensure_ascii=False)}",
            f"参考一致性: {'PASS' if comp.get('all_pass') else 'N/A'}",
        ]
        self.budget.setPlainText("\n".join(lines))


def _fmt(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.3f}"


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    default = argv[1] if len(argv) > 1 else "workspace/demo/runs/S1_unrestricted_4w.json"
    app = QApplication(argv[:1])
    window = MainWindow(default)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
