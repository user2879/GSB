from __future__ import annotations

import json

from rerank_lab.loader import load_sample_dir
from tests.conftest import patch_config


def test_invalid_candidates_are_reported_and_valid_rows_sorted(sample_copy):
    sample = sample_copy("default")
    path = sample / "candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.extend(
        [
            {
                "request_id": "",
                "item_id": "bad-empty-request",
                "recall_scores": {"bm25": 1},
                "dense_features": [0, 0, 0, 0],
                "sparse_features": {},
            },
            dict(rows[0], item_id="c-001"),
            {
                "request_id": "r-001",
                "item_id": "bad-negative-sparse",
                "recall_scores": {"bm25": 1},
                "dense_features": [0, 0, 0, 0],
                "sparse_features": {"brand": -1},
            },
            {
                "request_id": "r-001",
                "item_id": "bad-nan",
                "recall_scores": {"bm25": 1},
                "dense_features": [float("nan"), 0, 0, 0],
                "sparse_features": {},
            },
        ]
    )
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    loaded = load_sample_dir(sample)
    codes = [failure.code for failure in loaded.failures]
    assert codes.count("E_INPUT_INVALID") >= 4
    assert loaded.candidates == sorted(
        loaded.candidates, key=lambda item: (item.request_id, item.item_id)
    )
    assert ("r-001", "c-001") in {
        (candidate.request_id, candidate.item_id) for candidate in loaded.candidates
    }


def test_weight_dimension_and_budget_codes(sample_copy):
    sample = sample_copy("default")
    patch_config(sample, feature_weights=[1, 2])
    loaded = load_sample_dir(sample)
    assert loaded.config is None
    assert any(failure.code == "E_INPUT_INVALID" for failure in loaded.failures)

    sample = sample_copy("budget")
    patch_config(sample, concurrency=33)
    loaded = load_sample_dir(sample)
    assert loaded.config is None
    assert any(failure.code == "E_BUDGET_INVALID" for failure in loaded.failures)


def test_request_without_valid_candidates_is_excluded_from_progress(sample_copy, tmp_path):
    from rerank_lab.engine import run_experiment
    from tests.conftest import write_commands

    sample = sample_copy("default")
    path = sample / "candidates.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["request_id"] != "r-001"
    ]
    rows.append(
        {
            "request_id": "r-001",
            "item_id": "bad",
            "recall_scores": {"bm25": 1},
            "dense_features": ["not-a-number", 0, 0, 0],
            "sparse_features": {},
        }
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    write_commands(sample, [])
    report = run_experiment(sample, tmp_path / "out", enable_snapshots=False)
    assert report["summary"]["total_requests"] == 4
    assert report["summary"]["failed_requests"] == 1
    assert report["summary"]["eligible_requests"] == 3
    assert report["summary"]["completed_requests"] == 3
    assert report["summary"]["progress"]["basis_points"] == 10000
