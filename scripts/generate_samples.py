#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_FIXED_TIME = "2026-01-01T00:00:00.000Z"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _remove_readonly(function: Any, path: str, exc_info: Any) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        function(path)
    except OSError:
        pass


def volume_v1(volume_id: str, capacity: int) -> dict[str, Any]:
    return {"id": volume_id, "capacity_bytes": capacity}


def volume_v2(
    volume_id: str,
    capacity: int,
    ops_per_sec: int,
    burst_capacity: int,
) -> dict[str, Any]:
    return {
        "id": volume_id,
        "capacity_bytes": capacity,
        "ops_per_sec": ops_per_sec,
        "burst_capacity": burst_capacity,
    }


def volume_v3(
    volume_id: str,
    capacity: int,
    ops_per_sec: int,
    burst_capacity: int,
    limit_dimension: str,
    derivation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": volume_id,
        "capacity_bytes": capacity,
        "ops_per_sec": ops_per_sec,
        "burst_capacity": burst_capacity,
        "limit_dimension": limit_dimension,
        "derivation": derivation,
    }


def generate(root: Path) -> None:
    config_dir = root / "configs"
    events_dir = root / "events"
    metadata_dir = root / "metadata"
    logs_dir = root / "logs"
    for directory in (config_dir, events_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # The sample generator owns the config fixture directory.  Remove only
    # generated JSON/backup artifacts so a rerun returns to the pristine three
    # historical versions instead of retaining a previously edited snapshot.
    backups = config_dir / "backups"
    if backups.exists():
        shutil.rmtree(backups, onerror=_remove_readonly)
    for path in config_dir.glob("*.json"):
        if path.name != "upgrade_checkpoint.json":
            path.unlink()

    versions: list[dict[str, Any]] = [
        {
            "format_version": 1,
            "revision": 1,
            "name": "sample-v1-capacity-only",
            "volumes": [
                volume_v1("root", 1000),
                volume_v1("child", 700),
                volume_v1("archive", 50),
            ],
        },
        {
            "format_version": 2,
            "revision": 2,
            "name": "sample-v2-rate-fields",
            "volumes": [
                volume_v2("root", 1000, 2, 2),
                volume_v2("child", 700, 5, 400),
                volume_v2("archive", 50, 0, 0),
            ],
        },
        {
            "format_version": 3,
            "revision": 3,
            "name": "sample-v3-derivation",
            "metadata": {"generated_by": "scripts/generate_samples.py"},
            "volumes": [
                volume_v3("root", 1000, 2, 2, "operations"),
                volume_v3(
                    "child",
                    700,
                    5,
                    400,
                    "bytes",
                    {"upstream_id": "root", "factor": 1, "offset": 100},
                ),
                volume_v3("archive", 50, 0, 0, "bytes"),
                volume_v3(
                    "mirror",
                    5000,
                    1,
                    1,
                    "operations",
                    {"upstream_id": "root", "factor": 1, "offset": 0},
                ),
            ],
        },
    ]
    for index, document in enumerate(versions, start=1):
        write_json(config_dir / f"config-v{index}.json", document)

    checkpoint = {
        "checkpoint_version": 1,
        "target_schema_version": 3,
        "entries": {},
        "updated_at": CHECKPOINT_FIXED_TIME,
    }
    write_json(config_dir / "upgrade_checkpoint.json", checkpoint)

    derivation_metadata = {
        "schema_version": 3,
        "arbitration_rule": "effective_capacity = min(explicit_capacity, derived_capacity)",
        "derived_capacity_rule": "upstream_effective_capacity * factor + offset",
        "relationships": [
            {
                "volume_id": "child",
                "upstream_id": "root",
                "factor": 1,
                "offset": 100,
                "explicit_capacity_bytes": 700,
                "expected_derived_capacity_bytes": 1100,
                "expected_arbitration": "explicit",
            },
            {
                "volume_id": "mirror",
                "upstream_id": "root",
                "factor": 1,
                "offset": 0,
                "explicit_capacity_bytes": 5000,
                "expected_derived_capacity_bytes": 1000,
                "expected_arbitration": "derived",
            },
        ],
    }
    write_json(metadata_dir / "derivations.json", derivation_metadata)

    events = [
        {
            "seq": 1,
            "timestamp_ms": 0,
            "volume_id": "root",
            "operation_count": 1,
            "write_bytes": 100,
        },
        {
            "seq": 2,
            "timestamp_ms": 0,
            "volume_id": "root",
            "operation_count": 1,
            "write_bytes": 200,
        },
        {
            "seq": 3,
            "timestamp_ms": 100,
            "volume_id": "root",
            "operation_count": 1,
            "write_bytes": 100,
        },
        {
            "seq": 4,
            "timestamp_ms": 100,
            "volume_id": "child",
            "operation_count": 1,
            "write_bytes": 300,
        },
        {
            "seq": 5,
            "timestamp_ms": 200,
            "volume_id": "child",
            "operation_count": 1,
            "write_bytes": 450,
        },
        {
            "seq": 6,
            "timestamp_ms": 1000,
            "volume_id": "child",
            "operation_count": 1,
            "write_bytes": 100,
        },
        {
            "seq": 7,
            "timestamp_ms": 1000,
            "volume_id": "root",
            "operation_count": 1,
            "write_bytes": 100,
        },
        {
            "seq": 8,
            "timestamp_ms": 1500,
            "volume_id": "archive",
            "operation_count": 1,
            "write_bytes": 60,
        },
        {
            "seq": 9,
            "timestamp_ms": 2000,
            "volume_id": "mirror",
            "operation_count": 1,
            "write_bytes": 100,
        },
        {
            "seq": 10,
            "timestamp_ms": 2000,
            "volume_id": "mirror",
            "operation_count": 1,
            "write_bytes": 0,
        },
    ]
    events_path = events_dir / "access-events.ndjson"
    events_path.write_text(
        "".join(
            json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for event in events
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic local quota samples")
    parser.add_argument("--output", default=str(ROOT / "data"))
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if any(part.lower().startswith(("http:", "https:", "ftp:")) for part in target.parts):
        raise SystemExit("output must be a local filesystem path")
    generate(target)
    print(json.dumps({"code": "OK", "output": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
