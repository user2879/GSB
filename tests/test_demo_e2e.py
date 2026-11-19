"""要求 8：固定序列演示端到端、两次逐字节一致、崩溃恢复一致。"""
from __future__ import annotations

import os

from syncbench.demo import run_sequence, run_with_crash_recovery, write_reports
from syncbench.hashing import sha256_hex


def test_demo_sequence_report_deterministic(tmp_path):
    ws_a = str(tmp_path / "a")
    ws_b = str(tmp_path / "b")
    report_a = run_sequence(ws_a)
    report_b = run_sequence(ws_b)
    pa, _ = write_reports(ws_a, report_a)
    pb, _ = write_reports(ws_b, report_b)
    assert sha256_hex(open(pa, "rb").read()) == sha256_hex(open(pb, "rb").read())


def test_demo_contains_all_expected_error_codes(tmp_path):
    report = run_sequence(str(tmp_path / "ws"))
    all_codes = set()
    for codes in report["all_error_codes"].values():
        all_codes.update(codes)
    # 高漂移 + 超长空洞两类业务码必须在固定序列中出现
    assert "E_DRIFT_EXCEED" in all_codes
    assert "E_GAP_TOO_LONG" in all_codes


def test_crash_recovery_continues_consistently(tmp_path):
    ws = str(tmp_path / "crash")
    report = run_with_crash_recovery(ws)
    assert report["crash_exit_code"] == 137
    assert report["recovery"]["forced_termination_detected"] is True
    assert report["recovery"]["orphans_reclaimed"]
    # 恢复后继续增量成功并正常受控关闭
    assert report["shutdown"]["snapshot_version"] >= 1


def test_gui_offscreen_self_check(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from syncbench.gui import launch_gui
    rc = launch_gui(str(tmp_path / "gui_ws"))
    assert rc == 0
