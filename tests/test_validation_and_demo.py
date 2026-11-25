from __future__ import annotations

import json

from candidate_diagnostics.demo import run_demo
from candidate_diagnostics.loader import load_requests_jsonl


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_duplicate_request_id_is_failure_not_batch_abort(tmp_path, config):
    path = tmp_path / "requests.jsonl"
    row = {"request_id": "r1", "candidates": [
        {"candidate_id": 1, "raw_score": 1.0, "channel": "feed"}
    ]}
    write_jsonl(path, [row, row])
    requests, failures = load_requests_jsonl(path, config)
    assert list(requests) == ["r1"]
    assert any(f.code == "E_REQUEST_ID_DUPLICATE" for f in failures)


def test_invalid_candidate_rows_are_collected(tmp_path, config):
    path = tmp_path / "invalid.jsonl"
    write_jsonl(path, [{
        "request_id": "r-bad",
        "candidates": [
            {"candidate_id": -1, "raw_score": 1.0, "channel": "feed"},
            {"candidate_id": 2, "raw_score": float("nan"), "channel": "feed"},
            {"candidate_id": 3, "raw_score": 1.0, "aux_feature": 1.5, "channel": "missing"},
        ],
    }])
    _, failures = load_requests_jsonl(path, config)
    codes = {f.code for f in failures}
    assert "E_CANDIDATE_ID_INVALID" in codes
    assert "E_SCORE_NON_FINITE" in codes
    assert "E_AUX_NON_INTEGER" in codes
    assert "E_CHANNEL_UNKNOWN" in codes


def test_demo_report_end_to_end(root):
    report = run_demo(root)
    assert report["injections"]["all_hit"] is True
    assert report["incremental"]["final_matches_blank_rebuild"] is True
    assert report["concurrency"]["business_fields_match_serial"] is True
    assert report["format_roundtrip"]["mismatch_count"] == 0
    assert report["sample_parse"]["generated_v1_failures"] == []
    assert report["sample_parse"]["generated_v2_failures"] == []
