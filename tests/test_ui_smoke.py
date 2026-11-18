from __future__ import annotations

import pytest

from graygate.service import WorkbenchService


def test_tk_workbench_renders_stable_rows(sample_spec, tmp_path):
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    db_path = tmp_path / "ui.db"
    service = WorkbenchService(sample_spec, db_path)
    try:
        for index in range(12):
            service.ingest(
                f"ui-client-{index % 4}",
                "alpha:ui" if index % 2 == 0 else "beta:ui",
                [((index + offset) % 11) / 11.0 for offset in range(8)],
            )
        snapshot = service.rebuild_now()
        assert service.check_consistency().ok
    finally:
        service.close()

    from graygate.ui import WorkbenchApp

    app = WorkbenchApp(root, str(sample_spec.source_path), str(db_path))
    try:
        root.update_idletasks()
        root.update()
        assert root.title() == "Gray Snapshot Rebuild Workbench"
        assert len(app.stats_tree.get_children()) == sample_spec.stage_count
        assert len(app.snapshot_tree.get_children()) == 1
        assert len(app.rebuild_tree.get_children()) == 1
        assert f"snapshot={snapshot.snapshot_id}" in app.summary_var.get()
        assert "authoritative_check=" in app.summary_var.get()
    finally:
        app.close()

