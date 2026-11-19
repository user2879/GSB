"""Optional PyQt5 viewer for immutable, versioned diagnostic snapshots."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .snapshot import SnapshotStore


def launch_viewer(workspace: str | Path, *, offscreen: bool = False) -> int:
    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget
    except Exception as exc:  # pragma: no cover - depends on local Qt deployment
        raise RuntimeError(f"PyQt5 is unavailable: {exc}") from exc

    store = SnapshotStore(Path(workspace))
    snapshot = store.load_current()
    view_dir = store.views / snapshot["version"]
    if not (view_dir / "manifest.json").is_file():
        from .render import render_committed_snapshot

        info = render_committed_snapshot(store)
        view_dir = Path(info["view_dir"])

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle(
        f"Spatial acoustic diagnostics | immutable snapshot {snapshot['version']}"
    )
    central = QWidget()
    layout = QVBoxLayout(central)
    header = QLabel(
        f"Snapshot `{snapshot['version']}` | content {snapshot['integrity']['content_sha256'][:20]}...\n"
        "Only verified snapshot payloads are rendered. Low-confidence directions are explicit."
    )
    header.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(header)
    tabs = QTabWidget()
    for tier_name in snapshot["tiers"]:
        image_path = view_dir / f"{tier_name}.png"
        label = QLabel()
        if image_path.is_file():
            pixmap = QPixmap(str(image_path))
            label.setPixmap(pixmap.scaledToWidth(1200, Qt.SmoothTransformation))
            label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        else:
            label.setText(f"Missing rendered view: {image_path}")
        tabs.addTab(label, tier_name)
    layout.addWidget(tabs)
    window.setCentralWidget(central)
    window.resize(1280, 900)
    window.show()
    if offscreen:
        app.processEvents()
        export_path = store.views / snapshot["version"] / "qt-offscreen-window.png"
        window.grab().save(str(export_path), "PNG")
        return 0
    return int(app.exec_())

