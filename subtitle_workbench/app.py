"""Desktop entry point: opens the interactive workbench window."""
from __future__ import annotations

import sys

from PySide6 import QtWidgets

from .controller import WorkbenchController
from .gui import WorkbenchWindow
from .layout import solve_layout
from .sample import generate_sample
from .snapshot import SnapshotStore


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    store = SnapshotStore("snapshots")
    if store.load_current() is None:
        project = generate_sample()
        store.commit(project.items, solve_layout(project.items))
    window = WorkbenchWindow(WorkbenchController(store))
    window.resize(1500, 780)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
