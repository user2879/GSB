from __future__ import annotations

from candidate_diagnostics.injections import run_injection_plan
from candidate_diagnostics.stats_receipts import (
    adjudicate_receipts,
    load_three_receipt_formats,
)


def test_three_receipt_formats_normalize_and_adjudicate(root, config):
    receipts = load_three_receipt_formats(
        root / "samples/stats_receipts_v1_legacy.txt",
        root / "samples/stats_receipts_v2_nested.json",
        root / "samples/stats_receipts_v3_columnar.csv",
    )
    view = adjudicate_receipts(receipts)
    assert view["receipt_count"] == 13
    req1_search = next(
        item for item in view["authoritative"]
        if item["request_id"] == "req-001" and item["channel"] == "search"
    )
    assert req1_search["receipt_version"] == 3
    decision = next(
        item for item in view["decisions"]
        if item["request_id"] == "req-001" and item["channel"] == "search"
    )
    assert "version 3 takes precedence" in decision["decision_basis"]
    req3_feed = next(
        item for item in view["decisions"]
        if item["request_id"] == "req-003" and item["channel"] == "feed"
    )
    assert "version 3 takes precedence" in req3_feed["decision_basis"]

    same_version = adjudicate_receipts([
        {
            "request_id": "r", "channel": "c", "population": 2, "quota": 1,
            "selected": 1, "unselected": 1, "ratio_permille": 500, "scheme": "s",
            "receipt_version": 2, "receipt_time": "2026-11-21T09:00:00+08:00",
            "source_format": "a", "source_file": "a",
        },
        {
            "request_id": "r", "channel": "c", "population": 2, "quota": 1,
            "selected": 1, "unselected": 1, "ratio_permille": 500, "scheme": "s",
            "receipt_version": 2, "receipt_time": "2026-11-21T09:30:00+08:00",
            "source_format": "b", "source_file": "b",
        },
    ])
    assert "latest receipt_time" in same_version["decisions"][0]["decision_basis"]
    assert same_version["authoritative"][0]["receipt_time"].endswith("09:30:00+08:00")


def test_all_injections_hit_expected_business_codes(root, config):
    report = run_injection_plan(root / "samples/injections.json", config)
    assert report["total"] == 6
    assert report["all_hit"] is True
    by_id = {item["injection_id"]: item for item in report["results"]}
    assert "E_SCORE_NON_FINITE" in by_id["INJ-NAN-SCORE-001"]["observed_codes"]
    assert "E_CANDIDATE_ID_DUPLICATE" in by_id["INJ-DUP-CAND-002"]["observed_codes"]
    assert by_id["INJ-RATIO-003"]["hit"] is True
    assert "E_SELECT_CODE_UNKNOWN" in by_id["INJ-CODE-004"]["observed_codes"]
    assert "E_FIELD_TRUNCATED" in by_id["INJ-WIDTH-005"]["observed_codes"]
    assert "E_CAND_COUNT_OVER" in by_id["INJ-COUNT-006"]["observed_codes"]
