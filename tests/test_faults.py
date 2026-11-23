from __future__ import annotations

from pathlib import Path

from leaseaudit.faults import run_fault_injection


def test_three_required_fault_classes(tmp_path: Path):
    expected = {
        "scene_corrupt": ("E_SCENE", "sources[0].address"),
        "quota_overflow": ("E_QUOTA", "sources[1].reserve_quota"),
        "ledger_digest": ("E_SNAPSHOT", "$.events[1].event_hash"),
    }
    for kind, (code, field_path) in expected.items():
        result = run_fault_injection(tmp_path / kind, kind)
        assert result["status"] == "EXPECTED_FAILURE"
        assert result["business_code"] == code
        assert result["field_path"] == field_path
        assert result["source_code_modified"] is False
        assert result["partially_written_ledger"] is False
        log_text = (tmp_path / kind / "structured.jsonl").read_text(encoding="utf-8")
        assert code in log_text
        assert field_path in log_text
