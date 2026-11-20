from __future__ import annotations

from collections import defaultdict
from typing import Any

from .analysis import combination_audit_row
from .model import ExperimentConfig, SampleSpec


def _combo_label(combo: dict[str, Any]) -> str:
    block = combo["block_shape"]
    return f"BM{block['BM']}-BN{block['BN']}-BK{block['BK']}/{combo['loop_order']}"


def _max_error(combo: dict[str, Any]) -> float:
    return float(combo["error_summary"]["max_absolute_error"])


def build_sample_comparison(
    sample: SampleSpec,
    combinations: list[dict[str, Any]],
    config: ExperimentConfig,
) -> dict[str, Any]:
    audit_rows = [combination_audit_row(combo) for combo in combinations]
    if not combinations:
        return {
            "sample": sample.name,
            "shapes": sample.shape_dict(),
            "reference_policy": "float64_serial_k_ascending",
            "audit_table": audit_rows,
            "conclusions": {
                "block_size_effect": "No successful block-size combination was available.",
                "loop_order_effect": "No successful loop-order combination was available.",
                "dominant_error_source": "No successful combination was available.",
                "overall": "All combinations failed; inspect the failure list.",
            },
        }

    by_loop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for combo in combinations:
        by_loop[str(combo["loop_order"])].append(combo)
        block_key = _combo_label(combo).split("/")[0]
        by_block[block_key].append(combo)

    block_sensitivity: list[dict[str, Any]] = []
    for order, group in sorted(by_loop.items()):
        if len(group) < 2:
            continue
        errors = [_max_error(item) for item in group]
        block_sensitivity.append(
            {
                "loop_order": order,
                "min_max_absolute_error": min(errors),
                "max_max_absolute_error": max(errors),
                "delta": max(errors) - min(errors),
            }
        )

    loop_sensitivity: list[dict[str, Any]] = []
    for block, group in sorted(by_block.items()):
        if len(group) < 2:
            continue
        errors = [_max_error(item) for item in group]
        digests = sorted({item["kernel"]["output_sha256"] for item in group})
        loop_sensitivity.append(
            {
                "block_label": block,
                "min_max_absolute_error": min(errors),
                "max_max_absolute_error": max(errors),
                "delta": max(errors) - min(errors),
                "distinct_output_digests": len(digests),
            }
        )

    best = min(
        combinations,
        key=lambda item: (
            _max_error(item),
            item["error_summary"]["exceed_count"],
            _combo_label(item),
        ),
    )
    worst = max(
        combinations,
        key=lambda item: (
            _max_error(item),
            item["error_summary"]["exceed_count"],
            _combo_label(item),
        ),
    )
    block_delta = max((item["delta"] for item in block_sensitivity), default=0.0)
    loop_delta = max((item["delta"] for item in loop_sensitivity), default=0.0)
    distinct_loop_outputs = max(
        (item["distinct_output_digests"] for item in loop_sensitivity), default=1
    )

    block_conclusion = (
        f"Across fixed loop orders, the block-shape spread in max |error| is {block_delta:.3e}; "
        f"best={_combo_label(best)} and worst={_combo_label(worst)}. "
        "Different BK values change K-chunk local sums, the number of chunk merges, and the merge "
        "association depth, which is the observed block-size rounding mechanism."
    )
    if distinct_loop_outputs == 1:
        loop_conclusion = (
            f"Loop-order spread in max |error| is {loop_delta:.3e}, and all loop orders produced "
            "bit-identical C for each fixed block shape. With the configured BK, either only one "
            "K chunk exists or the three merge topologies happen to round identically for this "
            "sample; the per-element merge schedule is still recorded in the audit metadata."
        )
    else:
        loop_conclusion = (
            f"Loop-order spread in max |error| is {loop_delta:.3e}, with "
            f"{distinct_loop_outputs} distinct output digests for at least one block shape. "
            "The loop order controls both intra-chunk ijk/ikj/jik traversal and the K-chunk "
            "merge topology (forward sequential, reverse sequential, balanced tree); inspect "
            "per-element accumulation order ids for the affected combinations."
        )

    band_shares: dict[str, list[float]] = defaultdict(list)
    for combo in combinations:
        for row in combo["layer_tables"]["K"]:
            band_shares[str(row["layer_key"])].append(float(row["attribution_share"]))
    mean_band_shares = {
        key: sum(values) / len(values) for key, values in sorted(band_shares.items())
    }
    dominant_band = (
        max(mean_band_shares, key=lambda key: (mean_band_shares[key], key))
        if mean_band_shares
        else "n/a"
    )
    dominant_band_share = mean_band_shares.get(dominant_band, 0.0)
    stage_counts: dict[str, int] = defaultdict(int)
    for combo in combinations:
        for element in combo["exceeded_elements"]:
            stage = "unknown"
            stages = element.get("propagation_stages", [])
            if stages:
                stage = str(stages[0].get("stage", "k_chunk"))
            stage_counts[stage] += 1
    dominant_stage = (
        max(stage_counts, key=lambda key: (stage_counts[key], key)) if stage_counts else "n/a"
    )
    attribution_conclusion = (
        f"Mean K-band attribution share is led by {dominant_band} "
        f"({dominant_band_share:.1%}); the most frequent first propagation stage among exceeded "
        f"elements is {dominant_stage}. This ties excess error to localized chunk accumulation "
        "before alpha/beta finalization."
    )
    overall = (
        f"{sample.name}: {len(combinations)} successful combinations. {block_conclusion} "
        f"{loop_conclusion} {attribution_conclusion}"
    )
    return {
        "sample": sample.name,
        "shapes": sample.shape_dict(),
        "reference_policy": "float64_serial_k_ascending",
        "audit_table": audit_rows,
        "block_sensitivity": block_sensitivity,
        "loop_sensitivity": loop_sensitivity,
        "mean_k_band_attribution_share": mean_band_shares,
        "exceeded_first_stage_counts": dict(sorted(stage_counts.items())),
        "conclusions": {
            "block_size_effect": block_conclusion,
            "loop_order_effect": loop_conclusion,
            "dominant_error_source": attribution_conclusion,
            "overall": overall,
        },
    }
