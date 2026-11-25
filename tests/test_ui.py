"""无头 Tk 仪表盘冒烟测试（通过 SLO_DISPLAY/Xvfb）。"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("SLO_DISPLAY") and not os.environ.get("DISPLAY"),
    reason="无可用 X 显示（需先启动 scripts/start-xvfb.sh）",
)


def test_dashboard_builds_and_renders(demo_workspace, tmp_path):
    from tests.test_ledger_recovery import _run

    ledger_path = tmp_path / "ui.db"
    _run(demo_workspace, ledger_path)
    from slo_burn.ui import BurnDashboard

    app = BurnDashboard(str(demo_workspace), str(ledger_path))
    app.update_idletasks()
    app.update()
    widgets = app.winfo_children()
    assert widgets
    app.destroy()
