from __future__ import annotations

import os
from pathlib import Path

import pytest

from leaseaudit.demo import run_demo


@pytest.mark.skipif(
    os.environ.get("RUN_DEMO_INTEGRATION") != "1",
    reason="set RUN_DEMO_INTEGRATION=1 to run the real 5.6 second loopback demo",
)
def test_full_loopback_demo_and_snapshot_restore(tmp_path: Path):
    summary = run_demo(tmp_path, base_port=55901)
    assert summary["overall_code"] == "OK"
    assert summary["business_codes"]["cold_start"] == "OK"
    assert summary["business_codes"]["expiry_reclaim"] == "OK"
    assert summary["business_codes"]["quota_reclaim"] == "OK"
    assert summary["business_codes"]["audit"] == "OK"
    assert summary["business_codes"]["snapshot_replace_restore"] == "OK"
    assert summary["retained_plus_global_equals_capacity"] is True
    assert summary["final_state"]["total_active"] <= summary["final_state"]["cluster_capacity"]
