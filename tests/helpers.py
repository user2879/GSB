from __future__ import annotations

import json
from pathlib import Path

from vecflow.model import Workspace, parse_data_payload, parse_tasks_payload, workspace_state_from_parts


def data_payload(row_count: int = 8) -> dict:
    x = [1 + index for index in range(row_count)]
    divisor = [2.0 if index % 3 else 0.0 for index in range(row_count)]
    flag = [index % 2 == 0 for index in range(row_count)]
    null_x = [0] * row_count
    if row_count > 2:
        null_x[2] = 1
    return {
        "format_version": 1,
        "blocks": [
            {
                "id": "b",
                "row_count": row_count,
                "columns": [
                    {"name": "x", "type": "integer", "values": x, "null_bitmap": null_x},
                    {"name": "d", "type": "float", "values": divisor, "null_bitmap": [0] * row_count},
                    {"name": "flag", "type": "boolean", "values": flag, "null_bitmap": [0] * row_count},
                    {"name": "y", "type": "integer", "values": [index for index in range(row_count)], "null_bitmap": [0] * row_count},
                ],
            }
        ],
    }


def tasks_payload(
    *,
    row_count: int = 8,
    budget: int = 4,
    threads: int = 3,
    quota: int = 2,
    delay_ms: int = 0,
    expression: str = "flag ? x / d : 0.0",
    aggregation: str = "concat",
    shards: list[dict] | None = None,
) -> dict:
    if shards is None:
        shards = [
            {"start": 0, "end": min(3, row_count), "resources": 1},
            {"start": min(3, row_count), "end": min(6, row_count), "resources": 2},
            {"start": min(6, row_count), "end": row_count, "resources": 1},
        ]
    return {
        "format_version": 1,
        "parameter_version": 1,
        "global_budget": budget,
        "threads": threads,
        "tasks": [
            {
                "id": "t",
                "block": "b",
                "expression": expression,
                "quota": quota,
                "aggregation": aggregation,
                "delay_ms": delay_ms,
                "shards": shards,
            }
        ],
    }


def make_state(*, budget: int = 4, threads: int = 3, quota: int = 2, delay_ms: int = 0, expression: str = "flag ? x / d : 0.0", aggregation: str = "concat"):
    data = parse_data_payload(data_payload())
    tasks = parse_tasks_payload(
        tasks_payload(
            budget=budget,
            threads=threads,
            quota=quota,
            delay_ms=delay_ms,
            expression=expression,
            aggregation=aggregation,
        )
    )
    return workspace_state_from_parts(data, tasks)


def make_workspace(tmp_path: Path, **kwargs) -> Workspace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = make_state(**kwargs)
    data_file = tmp_path / "data.json"
    tasks_file = tmp_path / "tasks.json"
    data_file.write_text(json.dumps(state.blocks.to_dict()), encoding="utf-8")
    payload = state.tasks.to_dict()
    payload["parameter_version"] = state.parameter_version
    payload["global_budget"] = state.global_budget
    payload["threads"] = state.threads
    tasks_file.write_text(json.dumps(payload), encoding="utf-8")
    return Workspace.load(data_file, tasks_file)
