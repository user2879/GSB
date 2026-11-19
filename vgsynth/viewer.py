from __future__ import annotations

import json
import sys
from pathlib import Path


def show_run(run_dir: Path) -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QMainWindow, QSplitter

    run_dir = Path(run_dir)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    app = QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("VGSynth deterministic run viewer")
    splitter = QSplitter(Qt.Orientation.Horizontal)
    case_list = QListWidget()
    for case in report.get("cases", []):
        case_list.addItem(f"{case['case_id']}  [{case['status']}]")
    preview = QLabel("Select a case")
    preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
    preview.setMinimumSize(640, 480)
    splitter.addWidget(case_list)
    splitter.addWidget(preview)
    splitter.setStretchFactor(1, 1)
    window.setCentralWidget(splitter)

    def select_case(row: int) -> None:
        case = report["cases"][row]
        image_path = run_dir / "cases" / case["case_id"] / case["composite"]["final_bgra_file"].replace(".bgra", ".png")
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            preview.setText(f"No preview: {image_path}")
        else:
            preview.setPixmap(pixmap.scaled(768, 576, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
            verification = case.get("verification", {})
            preview.setToolTip(f"verification={verification.get('status')} code={verification.get('code')}")

    case_list.currentRowChanged.connect(select_case)
    if case_list.count():
        case_list.setCurrentRow(0)
    window.resize(1000, 680)
    window.show()
    return app.exec()
