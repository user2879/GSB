"""Explicit Python API walkthrough for loading, folding, ranking and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slowprof import (
    Interval,
    ProfilePolicy,
    aggregate_file,
    build_ranking,
    canonical_core_bytes,
    load_samples,
    profile_file,
    replay_snapshot,
    restore_snapshot,
    save_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="samples/small.ndjson")
    parser.add_argument("--runtime-dir", default="runtime")
    args = parser.parse_args()

    policy = ProfilePolicy(top_k=5)
    interval = Interval(1_700_000_000_000, 1_700_000_020_000, "obs-v1")

    samples, metadata = load_samples(args.samples, policy, interval)
    full = aggregate_file(args.samples, policy, interval, force_path="full")
    streaming = aggregate_file(args.samples, policy, interval, force_path="streaming")
    full_report = build_ranking(full, policy)
    streaming_report = build_ranking(streaming, policy)
    byte_equal = canonical_core_bytes(full, full_report) == canonical_core_bytes(
        streaming, streaming_report
    )

    session = profile_file(args.samples, policy, interval)
    snapshot_path = Path(args.runtime_dir) / "sessions" / "api-usage.json"
    save_snapshot(session, snapshot_path)
    restored = restore_snapshot(snapshot_path, session)
    replayed = replay_snapshot(snapshot_path, args.samples)

    output = {
        "loaded_rows": len(samples),
        "sample_sha256": metadata.file_sha256,
        "aggregation_path": session.aggregation.selection.path,
        "threshold_basis": session.aggregation.selection.to_dict(),
        "peak_memory_estimate_bytes": session.aggregation.memory_estimate[
            "estimated_peak_bytes"
        ],
        "byte_equal_full_and_streaming": byte_equal,
        "top_hotspots": [
            {
                "rank": item["rank"],
                "frame": item["frame"],
                "self_time_ms": item["self_time_ms"],
                "self_ratio_fixed": item["self_ratio_fixed"],
                "error_bound_dominated": item[
                    "quantization_order_error_bound_dominated"
                ],
            }
            for item in session.ranking_report["ranking"]
        ],
        "restore_strategy": restored.strategy,
        "replay_status": replayed.report["status"],
        "snapshot": str(snapshot_path),
    }
    print(json.dumps(output, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

