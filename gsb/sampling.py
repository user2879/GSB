from __future__ import annotations

from typing import Any

from .canonical import fingerprint
from .errors import PipelineError
from .fixedpoint import quantize_ratio


SAMPLING_ALGORITHM_VERSION = "weighted-order-v1"


def sampling_scope(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "algorithm": SAMPLING_ALGORITHM_VERSION,
        "sample_count": config["sampling"]["sample_count"],
        "seed": config["sampling"]["seed"],
        "domain_weights_fp": {
            name: config["domain_map"][name]["sampling_weight_fp"] for name in config["domain_names"]
        },
        "fixed_point": config["fixed_point"],
    }


def sampling_cache_key(survivors: list[str], dedup_fingerprint: str, config: dict[str, Any]) -> str:
    return fingerprint(
        {
            "stage": SAMPLING_ALGORITHM_VERSION,
            "survivor_ids": survivors,
            "upstream_dedup_fingerprint": dedup_fingerprint,
            "sampling_config_fingerprint": fingerprint(sampling_scope(config)),
        }
    )


def weighted_sample(
    survivors: list[str],
    samples: dict[str, dict[str, Any]],
    config: dict[str, Any],
    *,
    dedup_fingerprint: str,
) -> dict[str, Any]:
    count_required = config["sampling"]["sample_count"]
    if len(survivors) < count_required:
        raise PipelineError(
            "E_SAMPLE_POOL_EXHAUSTED",
            "not enough clean deduplicated samples for configured sample_count",
            field="sampling.sample_count",
            details={"required": count_required, "available": len(survivors)},
        )

    queues: dict[str, list[str]] = {name: [] for name in config["domain_names"]}
    for sample_id in survivors:
        queues[samples[sample_id]["domain"]].append(sample_id)
    for domain in queues:
        queues[domain].sort()

    selected: list[dict[str, Any]] = []
    allocated = {name: 0 for name in config["domain_names"]}

    for global_rank in range(count_required):
        best_domain: str | None = None
        best_sample_id: str | None = None
        for domain in config["domain_names"]:
            if not queues[domain]:
                continue
            candidate_sample_id = queues[domain][0]
            if best_domain is None:
                best_domain = domain
                best_sample_id = candidate_sample_id
                continue

            weight = config["domain_map"][domain]["sampling_weight_fp"]
            best_weight = config["domain_map"][best_domain]["sampling_weight_fp"]
            denominator = allocated[domain] + 1
            best_denominator = allocated[best_domain] + 1
            left = weight * best_denominator
            right = best_weight * denominator
            if left > right or (left == right and candidate_sample_id < best_sample_id):
                best_domain = domain
                best_sample_id = candidate_sample_id

        if best_domain is None or best_sample_id is None:
            raise PipelineError(
                "E_SAMPLE_POOL_EXHAUSTED",
                "weighted sampler ran out of domain candidates",
                field="sampling.sample_count",
            )

        sample_id = queues[best_domain].pop(0)
        allocated[best_domain] += 1
        selected.append(
            {
                "global_rank": global_rank,
                "sample_id": sample_id,
                "domain": best_domain,
                "allocation_number": allocated[best_domain],
                "domain_weight_fp": config["domain_map"][best_domain]["sampling_weight_fp"],
                "tie_break_key": sample_id,
                "effective_token_count": samples[sample_id]["term_count_total"],
            }
        )

    artifact = {
        "algorithm": SAMPLING_ALGORITHM_VERSION,
        "global_order": [item["sample_id"] for item in selected],
        "selections": selected,
        "domain_allocation": allocated,
        "cache_key": sampling_cache_key(survivors, dedup_fingerprint, config),
        "upstream_dedup_fingerprint": dedup_fingerprint,
    }
    artifact["artifact_fingerprint"] = fingerprint(_hashable(artifact))
    return artifact


def _hashable(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "artifact_fingerprint"}
