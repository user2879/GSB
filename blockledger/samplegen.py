from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .spec import normalize_spec
from .engine import replay_spec
from .spec import expand_intervals


def build_sample_spec() -> dict[str, Any]:
    volumes = [
        {
            "id": "vol-alpha",
            "block_size": 512,
            "block_count": 12,
            "seed": 2026112701,
            "baseline_version": 1,
        },
        {
            "id": "vol-beta",
            "block_size": 512,
            "block_count": 10,
            "seed": 2026112702,
            "baseline_version": 1,
        },
        {
            "id": "vol-gamma",
            "block_size": 1024,
            "block_count": 9,
            "seed": 2026112703,
            "baseline_version": 1,
        },
    ]
    snapshot_defs = [
        {
            "id": "snap-alpha-10",
            "volume_id": "vol-alpha",
            "version": 10,
            "intervals": [[0, 8]],
        },
        {
            "id": "snap-alpha-11",
            "volume_id": "vol-alpha",
            "version": 11,
            "intervals": [[4, 10]],
        },
        {
            "id": "snap-beta-20",
            "volume_id": "vol-beta",
            "version": 20,
            "intervals": [[0, 7]],
        },
        {
            "id": "snap-gamma-30",
            "volume_id": "vol-gamma",
            "version": 30,
            "intervals": [[1, 8]],
        },
    ]
    events = [
        {"seq": 1, "event_id": "evt-001", "type": "write", "volume_id": "vol-alpha", "intervals": [[1, 4]], "seed": 1101},
        {"seq": 2, "event_id": "evt-002", "type": "snapshot_create", "snapshot_id": "snap-alpha-10"},
        {"seq": 3, "event_id": "evt-003", "type": "write", "volume_id": "vol-alpha", "intervals": [[2, 4]], "seed": 1102},
        {"seq": 4, "event_id": "evt-004", "type": "write", "volume_id": "vol-alpha", "intervals": [[5, 7]], "seed": 1103},
        {"seq": 5, "event_id": "evt-005", "type": "snapshot_create", "snapshot_id": "snap-alpha-11"},
        {"seq": 6, "event_id": "evt-006", "type": "write", "volume_id": "vol-beta", "intervals": [[0, 3]], "seed": 2201},
        {"seq": 7, "event_id": "evt-007", "type": "snapshot_create", "snapshot_id": "snap-beta-20"},
        {"seq": 8, "event_id": "evt-008", "type": "write", "volume_id": "vol-gamma", "intervals": [[2, 5]], "seed": 3301},
        {"seq": 9, "event_id": "evt-009", "type": "snapshot_create", "snapshot_id": "snap-gamma-30"},
        {"seq": 10, "event_id": "evt-010", "type": "rollback", "rollback_id": "rb-alpha-10", "snapshot_id": "snap-alpha-10"},
        {"seq": 11, "event_id": "evt-011", "type": "rollback", "rollback_id": "rb-alpha-11", "snapshot_id": "snap-alpha-11"},
        {"seq": 12, "event_id": "evt-012", "type": "rollback", "rollback_id": "rb-beta-20", "snapshot_id": "snap-beta-20"},
        {"seq": 13, "event_id": "evt-013", "type": "rollback", "rollback_id": "rb-gamma-30", "snapshot_id": "snap-gamma-30"},
        {"seq": 14, "event_id": "evt-014", "type": "snapshot_delete", "snapshot_id": "snap-alpha-10"},
    ]
    budget = {
        "per_volume_rollback_concurrency": 1,
        "global_rollback_concurrency": 2,
        "max_blocks_per_rollback": 4,
        "verification_sample_permille": 600,
        "verification_timeout_ms": 1000,
    }

    snapshots: list[dict[str, Any]] = []
    for snapshot in snapshot_defs:
        block_indices = expand_intervals(snapshot["intervals"])
        snapshots.append(
            {
                **snapshot,
                "digests": {str(index): "0" * 64 for index in block_indices},
            }
        )
    raw = {"format_version": 1, "volumes": volumes, "snapshots": snapshots, "events": events, "budget": budget}
    spec = normalize_spec(raw)
    replay = replay_spec(spec, verify_declared_digests=False)
    for snapshot in spec["snapshots"]:
        captured = replay.snapshots[snapshot["id"]]
        snapshot["digests"] = {
            str(index): captured.digests[index] for index in sorted(captured.digests)
        }
    return normalize_spec(spec)


def sample_changes() -> dict[str, dict[str, Any]]:
    return {
        "change-write.json": {
            "change_id": "write-alpha-seq4-shift-into-snap11",
            "write": {
                "seq": 4,
                "event_id": "evt-004",
                "type": "write",
                "volume_id": "vol-alpha",
                "intervals": [[4, 6]],
                "seed": 1103,
            },
        },
        "change-delete.json": {
            "change_id": "delete-snap-alpha-11-after-rollback",
            "delete_snapshot": "snap-alpha-11",
        },
    }


def write_samples(root: str | Path = "samples") -> list[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    spec = build_sample_spec()
    written: list[Path] = []

    files = {
        "spec.json": spec,
        "volumes.json": {"format_version": 1, "volumes": spec["volumes"]},
        "snapshots.json": {"snapshots": spec["snapshots"]},
        "events.json": {"events": spec["events"]},
        "budget.json": {"budget": spec["budget"]},
    }
    for name, payload in files.items():
        path = root / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)

    for name, payload in sample_changes().items():
        path = root / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    for path in write_samples():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
