from __future__ import annotations

from rerank_lab.engine import run_experiment


def test_injection_hits_are_isolated_per_request(tmp_path):
    report = run_experiment("samples/injection", tmp_path / "out")
    assert [item["injection_id"] for item in report["injection_hits"]] == [
        "INJ-001",
        "INJ-002",
        "INJ-003",
    ]
    assert [(item["request_id"], item["code"]) for item in report["failures"]] == [
        ("r-002", "E_INPUT_INVALID"),
        ("r-004", "E_REQUEST_BUDGET_EXHAUSTED"),
        ("r-003", "E_INPUT_INVALID"),
    ]
    assert [request["request_id"] for request in report["requests"]] == ["r-001"]
    assert report["summary"]["completed_requests"] == 1
    assert report["summary"]["failed_requests"] == 3
    assert report["summary"]["eligible_requests"] == 1
    assert report["summary"]["progress"]["basis_points"] == 10000
    assert report["consistency"]["fresh_cold_match"] is True
