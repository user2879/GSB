from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quota_app.codec import parse_document
from quota_app.events import load_events
from quota_app.models import ConfigDocument


def document_raw(revision: int = 3) -> dict[str, Any]:
    return {
        "format_version": 3,
        "revision": revision,
        "name": "test-config",
        "volumes": [
            {
                "id": "root",
                "capacity_bytes": 1000,
                "ops_per_sec": 2,
                "burst_capacity": 2,
                "limit_dimension": "operations",
                "derivation": None,
            },
            {
                "id": "child",
                "capacity_bytes": 700,
                "ops_per_sec": 5,
                "burst_capacity": 400,
                "limit_dimension": "bytes",
                "derivation": {"upstream_id": "root", "factor": 1, "offset": 100},
            },
            {
                "id": "archive",
                "capacity_bytes": 50,
                "ops_per_sec": 0,
                "burst_capacity": 0,
                "limit_dimension": "bytes",
                "derivation": None,
            },
            {
                "id": "mirror",
                "capacity_bytes": 5000,
                "ops_per_sec": 1,
                "burst_capacity": 1,
                "limit_dimension": "operations",
                "derivation": {"upstream_id": "root", "factor": 1, "offset": 0},
            },
        ],
    }


def document(revision: int = 3) -> ConfigDocument:
    return parse_document(document_raw(revision))


def event_records() -> list[dict[str, Any]]:
    return [
        {"seq": 1, "timestamp_ms": 0, "volume_id": "root", "operation_count": 1, "write_bytes": 100},
        {"seq": 2, "timestamp_ms": 0, "volume_id": "root", "operation_count": 1, "write_bytes": 200},
        {"seq": 3, "timestamp_ms": 100, "volume_id": "root", "operation_count": 1, "write_bytes": 100},
        {"seq": 4, "timestamp_ms": 100, "volume_id": "child", "operation_count": 1, "write_bytes": 300},
        {"seq": 5, "timestamp_ms": 200, "volume_id": "child", "operation_count": 1, "write_bytes": 450},
        {"seq": 6, "timestamp_ms": 1000, "volume_id": "child", "operation_count": 1, "write_bytes": 100},
        {"seq": 7, "timestamp_ms": 1000, "volume_id": "root", "operation_count": 1, "write_bytes": 100},
        {"seq": 8, "timestamp_ms": 1500, "volume_id": "archive", "operation_count": 1, "write_bytes": 60},
        {"seq": 9, "timestamp_ms": 2000, "volume_id": "mirror", "operation_count": 1, "write_bytes": 100},
        {"seq": 10, "timestamp_ms": 2000, "volume_id": "mirror", "operation_count": 1, "write_bytes": 0},
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_events(path: Path, records: list[dict[str, Any]] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in (records if records is not None else event_records())
        ),
        encoding="utf-8",
    )
    return path


def load_test_events(path: Path):
    return load_events(write_events(path))


def make_legacy_workspace(root: Path, *, checkpoint: bool = True) -> Path:
    configs = root / "configs"
    write_json(
        configs / "config-v1.json",
        {
            "format_version": 1,
            "revision": 1,
            "name": "v1",
            "volumes": [
                {"id": "root", "capacity_bytes": 1000},
                {"id": "child", "capacity_bytes": 700},
            ],
        },
    )
    write_json(
        configs / "config-v2.json",
        {
            "format_version": 2,
            "revision": 2,
            "name": "v2",
            "volumes": [
                {"id": "root", "capacity_bytes": 1000, "ops_per_sec": 2, "burst_capacity": 2},
                {"id": "child", "capacity_bytes": 700, "ops_per_sec": 5, "burst_capacity": 400},
            ],
        },
    )
    write_json(configs / "config-v3.json", document_raw(3))
    if checkpoint:
        write_json(
            configs / "upgrade_checkpoint.json",
            {
                "checkpoint_version": 1,
                "target_schema_version": 3,
                "entries": {},
                "updated_at": "2026-01-01T00:00:00.000Z",
            },
        )
    return configs
