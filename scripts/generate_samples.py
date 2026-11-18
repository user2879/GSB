"""Generate deterministic local NDJSON samples and anomaly fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SMALL_BASE_MS = 1_700_000_000_000
LARGE_BASE_MS = 1_710_000_000_000


def _record(
    request_id: str,
    timestamp_ms: int,
    duration_ms: int,
    service_id: str,
    frames: Iterable[str],
    weight: int,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "timestamp_ms": timestamp_ms,
        "duration_ms": duration_ms,
        "service_id": service_id,
        "frames": list(frames),
        "weight": weight,
    }


def small_records() -> list[dict[str, Any]]:
    specs = [
        (120, "checkout", ["api.handle", "db.query", "sql.execute"], 2),
        (120, "checkout", ["api.handle", "db.query", "sql.execute"], 1),
        (25, "checkout", ["api.handle", "cache.get"], 4),
        (200, "checkout", ["api.handle", "db.query", "parser.parse"], 1),
        (500, "checkout", ["api.handle", "billing.charge", "http.call"], 2),
        (90, "checkout", ["api.handle", "billing.charge"], 1),
        (15, "worker", ["worker.run", "queue.poll"], 3),
        (40, "worker", ["worker.run", "queue.poll", "redis.get"], 2),
        (35, "checkout", ["api.handle", "auth.verify"], 2),
        (80, "checkout", ["api.handle", "db.query", "sql.execute"], 3),
        (480, "checkout", ["api.handle", "billing.charge", "http.call"], 1),
        (30, "checkout", ["api.handle", "cache.get"], 2),
    ]
    return [
        _record(
            f"req-small-{index:04d}",
            SMALL_BASE_MS + index * 1_000,
            duration,
            service,
            frames,
            weight,
        )
        for index, (duration, service, frames, weight) in enumerate(specs, start=1)
    ]


def large_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    moment = LARGE_BASE_MS

    def append(duration: int, service: str, frames: Iterable[str], weight: int) -> None:
        nonlocal moment
        records.append(
            _record(
                f"req-large-{len(records) + 1:05d}",
                moment,
                duration,
                service,
                frames,
                weight,
            )
        )
        moment += 10

    skew_paths = [
        ["gw.route", "checkout.charge", "risk.score", "ledger.write"],
        ["gw.route", "checkout.charge", "risk.score", "ledger.write"],
        ["gw.route", "checkout.charge", "coupon.apply", "ledger.write"],
        ["gw.route", "checkout.charge", "risk.score", "inventory.reserve"],
    ]
    for index in range(3000):
        append(900, "checkout", skew_paths[index % len(skew_paths)], 20)
    for index in range(100):
        append(
            50 + index,
            "worker",
            ["api.handle", "worker.work", f"worker.step{index:03d}"],
            40,
        )
    for index in range(20):
        append(
            70 + index,
            "db",
            ["api.handle", "db.query", f"sql.execute{index:03d}"],
            20,
        )
    deep_frames = [f"d{index:02d}.m{index:02d}" for index in range(63)]
    append(250, "deep", deep_frames, 60)
    for _ in range(10):
        append(140, "worker", ["worker.run", "queue.poll", "redis.get"], 30)
    return records


def zero_weight_records() -> list[dict[str, Any]]:
    return [
        _record("req-anomaly-ok", 1_720_000_000_000, 10, "checkout", ["api.handle", "db.query"], 1),
        _record("req-anomaly-zero", 1_720_000_000_010, 10, "checkout", ["api.handle", "db.query"], 0),
    ]


def depth_exceeded_records() -> list[dict[str, Any]]:
    frames = [f"s{index:02d}.m{index:02d}" for index in range(65)]
    return [
        _record("req-depth-ok", 1_720_000_001_000, 10, "deep", ["api.handle", "db.query"], 1),
        _record("req-depth-over", 1_720_000_001_010, 10, "deep", frames, 1),
    ]


def write_ndjson(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir = output_dir / "anomalies"
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "small": output_dir / "small.ndjson",
        "large": output_dir / "large.ndjson",
        "zero_weight": anomaly_dir / "zero_weight.ndjson",
        "depth_exceeded": anomaly_dir / "depth_exceeded.ndjson",
        "snapshot_missing_version": anomaly_dir / "snapshot_missing_version.json",
    }
    write_ndjson(paths["small"], small_records())
    write_ndjson(paths["large"], large_records())
    write_ndjson(paths["zero_weight"], zero_weight_records())
    write_ndjson(paths["depth_exceeded"], depth_exceeded_records())
    paths["snapshot_missing_version"].write_text(
        json.dumps(
            {
                "snapshot_schema_version": "slowprof.snapshot.v1",
                "interval": {},
                "sample": {},
                "aggregation": {},
                "folded_result": {},
                "hotspot_ranking": [],
                "error_annotation": {},
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    manifest = {
        "small_base_ms": SMALL_BASE_MS,
        "large_base_ms": LARGE_BASE_MS,
        "suggested_small_interval": {
            "start_ms": SMALL_BASE_MS,
            "end_ms": SMALL_BASE_MS + 20_000,
            "interval_version": "obs-v1",
        },
        "suggested_changed_interval": {
            "start_ms": SMALL_BASE_MS + 4_000,
            "end_ms": SMALL_BASE_MS + 20_000,
            "interval_version": "obs-narrow-v2",
        },
        "suggested_large_interval": {
            "start_ms": LARGE_BASE_MS,
            "end_ms": LARGE_BASE_MS + 1_000_000,
            "interval_version": "obs-v1",
        },
        "files": {
            name: {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("samples"))
    args = parser.parse_args()
    manifest = generate(args.output_dir)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
