from __future__ import annotations

from typing import Any

from .canonical import sha256_json


def split_service_analysis(service_analysis: dict[str, Any]) -> dict[str, Any]:
    summary_policies: dict[str, Any] = {}
    error_policies: dict[str, Any] = {}
    buckets: list[dict[str, Any]] = []
    attribution: list[dict[str, Any]] = []

    for policy_id, result in sorted(service_analysis["policies"].items()):
        summary_policies[policy_id] = {
            "policy_id": result["policy_id"],
            "policy_version": result["policy_version"],
            "strategy_version": result["strategy_version"],
            "policy": result["policy"],
            "sample_summary": result["sample_summary"],
            "result_digest": result["result_digest"],
        }
        error_policies[policy_id] = {
            "policy_version": result["policy_version"],
            "fixed_error": result["fixed_error"],
            "quantiles": result["quantiles"],
            "full_mean_us": result["full_mean_us"],
            "estimated_mean_us": result["estimated_mean_us"],
            "estimated_mean_fixed_us": result["estimated_mean_fixed_us"],
            "mean_error_bound_us": result["mean_error_bound_us"],
        }
        for bucket in result["bucket_stats"]:
            buckets.append({"policy_id": policy_id, **bucket})
        for bucket in result["attribution"]["buckets"]:
            attribution.append({"policy_id": policy_id, "scope": "bucket", **bucket})
        attribution.append(
            {
                "policy_id": policy_id,
                "scope": "total",
                **result["attribution"]["totals"],
            }
        )

    summary_payload = {
        "service_id": service_analysis["service_id"],
        "service_version": service_analysis["service_version"],
        "strategy_version": service_analysis["strategy_version"],
        "description": service_analysis["description"],
        "sample_stream_digest": service_analysis["sample_stream_digest"],
        "stream_summary": service_analysis["stream_summary"],
        "service_result_digest": service_analysis["service_result_digest"],
        "policies": summary_policies,
    }
    error_payload = {
        "service_id": service_analysis["service_id"],
        "policies": error_policies,
    }
    return {
        "summary": summary_payload,
        "error_bounds": error_payload,
        "buckets": sorted(
            buckets,
            key=lambda item: (item["policy_id"], int(item["bucket_index"])),
        ),
        "attribution": sorted(
            attribution,
            key=lambda item: (
                item["policy_id"],
                0 if item["scope"] == "total" else 1,
                int(item.get("bucket_index", -1)),
            ),
        ),
    }


def batch_content_digest(
    run_id: str,
    ordinal: int,
    service_id: str,
    split: dict[str, Any],
    flips: list[dict[str, Any]],
) -> str:
    return sha256_json(
        {
            "run_id": run_id,
            "ordinal": ordinal,
            "service_id": service_id,
            "summary": split["summary"],
            "error_bounds": split["error_bounds"],
            "buckets": split["buckets"],
            "attribution": split["attribution"],
            "flips": sorted(flips, key=lambda item: item["flip_key"]),
        }
    )
