"""JSON diagnostic report assembly."""

import json


def build_report(result, config, injection_hits=None, recovery_version=None):
    report = {
        "status": "ok" if not result.get("failures") else "completed_with_errors",
        "recovery_version": recovery_version,
        "snapshot_version": (
            result.get("snapshot_manifest", {}).get("global_version")
        ),
        "command_cursor": result.get("snapshot_manifest", {}).get(
            "command_cursor"
        ),
        "config_hash": config.hash_key(),
        "worker_threads": config.data["worker_threads"],
        "worker_high_water_mark": result.get("worker_high_water_mark"),
        "stage_timings_ms": result.get("stage_timings_ms", {}),
        "total_pipeline_ms": result.get("total_pipeline_ms"),
        "n_files": len(result.get("files", [])),
        "n_intervals": len(result.get("intervals", [])),
        "n_clusters": len(result.get("clusters", [])),
        "clusters": result.get("clusters", []),
        "attribution": result.get("attribution", []),
        "reused_products": result.get("reused_products", []),
        "recomputed_products": result.get("recomputed_products", []),
        "injection_hits": injection_hits or [],
        "failures": result.get("failures", []),
        "snapshot_events": result.get("snapshot_events", []),
        "snapshot_manifest": result.get("snapshot_manifest"),
    }
    return report


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, ensure_ascii=False)
    return path


def canonical_result(result):
    """Project a result to the fields compared for full/incremental parity."""
    return {
        "clusters": result.get("clusters", []),
        "attribution": [
            {
                key: row[key]
                for key in (
                    "interval_fingerprint",
                    "forced_cluster_id",
                    "nearest_cluster_id",
                    "attribution_distance",
                )
            }
            for row in result.get("attribution", [])
        ],
        "intervals": [
            {
                key: interval[key]
                for key in (
                    "start_frame",
                    "end_frame",
                    "start_ms",
                    "end_ms",
                    "overlap",
                    "mean_voiceprint",
                )
            }
            for interval in result.get("intervals", [])
        ],
        "interval_meta": result.get("interval_meta", []),
    }
