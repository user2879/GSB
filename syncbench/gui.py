"""PySide6 诊断视图（可选）。

视图只从 SnapshotStore 取不可变快照渲染；渲染前用 render_guard 校验版本，
版本过期抛 E_SNAPSHOT_STALE，不直接读取任何可变计算状态。
无显示环境时设置 QT_QPA_PLATFORM=offscreen 仍可构建与自检。
"""
from __future__ import annotations

import os

from .config import default_config
from .engine import Engine


def launch_gui(ws_dir: str) -> int:
    # 延迟导入，保证无 PySide6 的纯计算/测试场景不被 GUI 依赖阻塞
    try:
        from PySide6.QtGui import QColor, QPainter, QPen
        from PySide6.QtWidgets import (
            QApplication,
            QLabel,
            QMainWindow,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:  # pragma: no cover
        print("PySide6 未安装，无法启动 GUI；CLI demo 可独立运行。")
        return 3

    if not os.environ.get("QT_QPA_PLATFORM") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QApplication.instance() or QApplication([])
    engine = Engine(ws_dir, default_config())
    if engine.store.load_node_state() is None:
        engine.initial_build()
    else:
        engine.restore_from_store()
    snap = engine.snapshots.render_guard(engine.snapshots.current.version)

    class TimelineCanvas(QWidget):
        def __init__(self, snapshot) -> None:
            super().__init__()
            self.snapshot = snapshot
            self.setMinimumSize(720, 360)

        def paintEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(18, 20, 26))
            colors = [QColor(96, 180, 255), QColor(255, 160, 96),
                      QColor(120, 220, 140), QColor(220, 120, 220)]
            n_ch = 4
            row_h = self.height() / (n_ch + 1)
            for tile in self.snapshot.tiles:
                y = int(row_h * tile.channel) + 12
                pen = QPen(colors[tile.channel % 4])
                painter.setPen(pen)
                label = f"ch{tile.channel} {tile.kind}[{tile.mark}]"
                painter.drawText(8, y, label)
                if not tile.values:
                    continue
                step = max(1, self.width() // max(1, len(tile.values)))
                for i, value in enumerate(tile.values):
                    x = 8 + i * step
                    h = int(min(row_h - 24, abs(float(value)) * 4 + 4))
                    painter.drawLine(x, y + 4, x, y + 4 + h)
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(8, self.height() - 8,
                             f"snapshot v{snap.version} "
                             f"hash={snap.content_hash[:12]} "
                             f"degraded={snap.degraded}")

    window = QMainWindow()
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"多通道时钟同步诊断工作台  snapshot v{snap.version}"))
    layout.addWidget(TimelineCanvas(snap))
    window.setCentralWidget(central)
    window.setWindowTitle("SyncBench Diagnostic View")
    window.resize(760, 420)
    window.show()
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        # 无显示环境：渲染一帧即自检退出
        canvas = window.centralWidget().findChild(QWidget)
        pixmap = canvas.grab()
        assert not pixmap.isNull()
        return 0
    return app.exec()
