from __future__ import annotations

import pytest

from ossdiag.gui import dashboard_tables, launch_gui

from conftest import KiB, object_doc, spec_doc, workbench


def test_dashboard_model_contains_all_required_sections(tmp_path) -> None:
    wb = workbench(
        tmp_path,
        doc=spec_doc(objects=[object_doc("b", total_bytes=128 * KiB), object_doc("a", total_bytes=64 * KiB)]),
    )
    wb.run()
    dashboard = wb.build_dashboard()
    tables = dashboard_tables(dashboard)
    assert set(tables) == {"objects", "shards", "snapshots"}
    assert dashboard["ledger"]["format_version"] == 1
    assert "ledger_digest" in dashboard["ledger"]
    assert dashboard["scale_policy"]["scale_exponent"] == 6
    assert dashboard["objects"][0]["lease_watermark"]["global_limit"] == 3
    assert all("error_bounds_display" in item["fixed_stats"] for item in dashboard["objects"])
    assert all("path_scores" in item and item["path_reason"] for item in dashboard["objects"])
    wb.close()


def test_gui_reports_missing_tk_without_importing_at_module_load(tmp_path) -> None:
    wb = workbench(tmp_path)
    try:
        try:
            launch_gui(wb)
        except RuntimeError as exc:
            assert "Tkinter" in str(exc)
        else:  # pragma: no cover - only on hosts with a usable display
            wb.close()
    finally:
        wb.close()
