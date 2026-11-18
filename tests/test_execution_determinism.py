from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from columnar_pipeline.model import RunConfig
from columnar_pipeline.runner import replay_snapshot, run_pipeline


def _reference_result(path: Path) -> dict[str, list[Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, int]] = {}
    for table in sorted(document["tables"], key=lambda item: item["name"]):
        columns = {column["name"]: column for column in table["columns"]}
        for row in range(table["row_count"]):
            channel = columns["channel"]["values"][row]
            amount_is_null = bool(columns["amount"]["null_bitmap"][row])
            amount = None if amount_is_null else columns["amount"]["values"][row]
            if channel != "online":
                continue
            if amount is None or amount < 100 or amount > 900000:
                continue
            region = columns["region"]["values"][row]
            metric = groups.setdefault(region, {"row_count": 0, "amount_sum": 0, "nonnull": 0})
            metric["row_count"] += 1
            metric["amount_sum"] += amount
            metric["nonnull"] += 1
    result: dict[str, list[Any]] = {}
    for region in sorted(groups):
        metric = groups[region]
        result[region] = [
            metric["row_count"],
            metric["amount_sum"],
            metric["amount_sum"] // metric["nonnull"] if metric["nonnull"] else None,
        ]
    return result


def _run(
    data_path: Path,
    config: RunConfig,
    root: Path,
    name: str,
) -> dict[str, Any]:
    return run_pipeline(
        data_path,
        config,
        out_dir=root / name,
        ledger_path=root / "ledger.sqlite",
        snapshot_dir=root / "snapshots",
        log_path=root / "pipeline.jsonl",
    )


def test_reference_aggregation_and_parallelism_independence(
    tmp_path: Path,
    small_data: Path,
) -> None:
    configs = [
        RunConfig(
            parallelism=1,
            memory_budget_bytes=900,
            budget_version=21,
            split_threshold=1000,
            split_parts=2,
            row_group_size=16,
            ledger_batch_size=2,
        ),
        RunConfig(
            parallelism=4,
            memory_budget_bytes=2048,
            budget_version=22,
            split_threshold=1,
            split_parts=4,
            row_group_size=16,
            ledger_batch_size=3,
        ),
        RunConfig(
            parallelism=3,
            memory_budget_bytes=1200,
            budget_version=23,
            split_threshold=4,
            split_parts=2,
            row_group_size=16,
            ledger_batch_size=1,
        ),
    ]
    summaries = [
        _run(small_data, config, tmp_path, f"run_{index}")
        for index, config in enumerate(configs)
    ]
    result_bytes = [
        (tmp_path / f"run_{index}" / "final_result.json").read_bytes()
        for index in range(len(configs))
    ]
    assert result_bytes[0] == result_bytes[1] == result_bytes[2]
    assert len({summary["result_digest"] for summary in summaries}) == 1

    result = json.loads(result_bytes[0])
    assert result["query_id"] == summaries[0]["query_id"]
    expected = _reference_result(small_data)
    actual = {
        group["key"][0]: [
            group["metrics"]["row_count"],
            group["metrics"]["amount_sum"],
            group["metrics"]["amount_avg"],
        ]
        for group in result["groups"]
    }
    assert actual == expected
    assert result["group_by"] == ["region"]
    assert [group["key"][0] for group in result["groups"]] == sorted(expected)
    assert result["precision"]["avg_rounding"] == "floor"
    assert result["precision"]["avg_error_upper_bound_scaled_units"] == 1
    assert result["precision"]["sum_error_scaled_units"] == 0


def test_empty_table_and_zero_task_replay(tmp_path: Path) -> None:
    data_path = tmp_path / "empty.json"
    data_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "tables": [
                    {
                        "name": "empty_sales",
                        "version": 1,
                        "row_count": 0,
                        "columns": [
                            {"name": "region", "type": "string", "null_bitmap": [], "values": []},
                            {
                                "name": "amount",
                                "type": "fixed",
                                "scale": 2,
                                "null_bitmap": [],
                                "values": [],
                            },
                        ],
                    }
                ],
                "query": {
                    "from": "empty_sales",
                    "group_by": "region",
                    "aggregates": [
                        {"name": "row_count", "op": "count", "column": "*"},
                        {"name": "amount_sum", "op": "sum", "column": "amount"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    config = RunConfig(
        parallelism=1,
        memory_budget_bytes=1024,
        budget_version=1,
        split_threshold=2,
        split_parts=2,
        row_group_size=16,
    )
    summary = run_pipeline(
        data_path,
        config,
        out_dir=tmp_path / "empty_run",
        ledger_path=tmp_path / "empty_ledger.sqlite",
        snapshot_dir=tmp_path / "empty_snapshots",
        log_path=tmp_path / "empty.jsonl",
    )
    assert summary["task_count"] == 0
    result = json.loads((tmp_path / "empty_run" / "final_result.json").read_text(encoding="utf-8"))
    assert result["groups"] == []
    replay = replay_snapshot(
        summary["snapshot_path"],
        data_path=data_path,
        ledger_path=tmp_path / "empty_ledger.sqlite",
        out_dir=tmp_path / "empty_replay",
        log_path=tmp_path / "empty.jsonl",
    )
    assert replay["replayed_tasks"] == 0
    assert replay["result_digest"] == summary["result_digest"]
