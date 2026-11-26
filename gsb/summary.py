from __future__ import annotations

from typing import Any

from .canonical import fingerprint
from .fixedpoint import quantize_ratio


SUMMARY_ALGORITHM_VERSION = "ratio-summary-v1"


def build_summary(
    *,
    candidates: list[str],
    sampling_artifact: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    config: dict[str, Any],
    dedup_fingerprint: str,
) -> dict[str, Any]:
    width = config["fixed_point"]["decimal_width"]
    tier = config["fixed_point"]["quantization_tier"]
    scale = 10**width

    candidate_counts = {name: 0 for name in config["domain_names"]}
    candidate_tokens = {name: 0 for name in config["domain_names"]}
    for sample_id in candidates:
        domain = samples[sample_id]["domain"]
        candidate_counts[domain] += 1
        candidate_tokens[domain] += samples[sample_id]["term_count_total"]

    sampled_counts = {name: 0 for name in config["domain_names"]}
    sampled_tokens = {name: 0 for name in config["domain_names"]}
    for selection in sampling_artifact["selections"]:
        domain = selection["domain"]
        sampled_counts[domain] += 1
        sampled_tokens[domain] += selection["effective_token_count"]

    total_candidates = len(candidates)
    total_sampled = len(sampling_artifact["selections"])
    domains: list[dict[str, Any]] = []
    for domain in config["domain_names"]:
        sampled_ratio_fp = quantize_ratio(sampled_counts[domain], total_sampled, width, tier)
        post_dedup_share_fp = quantize_ratio(candidate_counts[domain], total_candidates, width, tier) if total_candidates else 0
        target_fp = config["domain_map"][domain]["target_ratio_fp"]
        deviation_fp = sampled_ratio_fp - target_fp
        post_dedup_deviation_fp = post_dedup_share_fp - target_fp
        direction = "equal" if deviation_fp == 0 else ("above_target" if deviation_fp > 0 else "below_target")
        domains.append(
            {
                "domain": domain,
                "sampled_count": sampled_counts[domain],
                "sampled_ratio_fp": sampled_ratio_fp,
                "post_dedup_count": candidate_counts[domain],
                "post_dedup_share_fp": post_dedup_share_fp,
                "post_dedup_effective_token_total": candidate_tokens[domain],
                "sampled_effective_token_total": sampled_tokens[domain],
                "target_ratio_fp": target_fp,
                "deviation_fp": deviation_fp,
                "deviation_direction": direction,
                "post_dedup_deviation_fp": post_dedup_deviation_fp,
                "post_dedup_deviation_direction": (
                    "equal"
                    if post_dedup_deviation_fp == 0
                    else ("above_target" if post_dedup_deviation_fp > 0 else "below_target")
                ),
            }
        )

    artifact = {
        "algorithm": SUMMARY_ALGORITHM_VERSION,
        "total_candidates": total_candidates,
        "total_sampled": total_sampled,
        "scale_fp": scale,
        "domains": domains,
        "upstream_sampling_fingerprint": sampling_artifact["artifact_fingerprint"],
        "upstream_dedup_fingerprint": dedup_fingerprint,
        "cache_key": fingerprint(
            {
                "stage": SUMMARY_ALGORITHM_VERSION,
                "sampling_fingerprint": sampling_artifact["artifact_fingerprint"],
                "dedup_fingerprint": dedup_fingerprint,
                "targets_fp": {
                    name: config["domain_map"][name]["target_ratio_fp"] for name in config["domain_names"]
                },
                "fixed_point": config["fixed_point"],
            }
        ),
    }
    artifact["artifact_fingerprint"] = fingerprint({k: v for k, v in artifact.items() if k != "artifact_fingerprint"})
    return artifact
