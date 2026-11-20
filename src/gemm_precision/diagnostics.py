from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_DIAGNOSTIC_MISMATCH, BusinessError
from .model import BlockShape, LoopOrder
from .reduction import MERGE_SCHEDULES, merge_rounding_for_element
from .util import payload_sha256


@dataclass(frozen=True)
class ElementTrace:
    main_value: float
    exact_float64_product_sum: float
    product_accumulation_rounding: float
    chunk_merge_rounding: float
    scale_rounding: float
    total_rounding_vs_reference: float
    k_band_abs_rounding: tuple[float, ...]
    dominant_k_band_index: int
    nonzero_term_count: int
    nonzero_term_positions: tuple[int, ...]
    contributors: tuple[dict[str, Any], ...]
    propagation_stages: tuple[dict[str, Any], ...]
    propagation_conclusion: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "main_value": self.main_value,
            "exact_float64_product_sum": self.exact_float64_product_sum,
            "product_accumulation_rounding": self.product_accumulation_rounding,
            "chunk_merge_rounding": self.chunk_merge_rounding,
            "scale_rounding": self.scale_rounding,
            "total_rounding_vs_reference": self.total_rounding_vs_reference,
            "k_band_abs_rounding": list(self.k_band_abs_rounding),
            "dominant_k_band_index": self.dominant_k_band_index,
            "nonzero_term_count": self.nonzero_term_count,
            "contributors": [dict(item) for item in self.contributors],
            "propagation_stages": [dict(item) for item in self.propagation_stages],
            "propagation_conclusion": self.propagation_conclusion,
        }


def _chunk_ranges(k_size: int, bk: int) -> list[tuple[int, int]]:
    return [(start, min(start + bk, k_size)) for start in range(0, k_size, bk)]


def _band_for_k(k: int, k_bands: list[dict[str, int]]) -> dict[str, int]:
    for band in k_bands:
        if int(band["start"]) <= k < int(band["end"]):
            return band
    raise IndexError(f"k={k} outside K bands")


def trace_element(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    *,
    row: int,
    col: int,
    block_shape: BlockShape,
    loop_order: LoopOrder,
    alpha: float,
    beta: float,
    reference_value: float,
    actual_value: float,
    k_bands: list[dict[str, int]],
    top_k: int,
    collect_details: bool,
) -> ElementTrace:
    """Replay the exact kernel chain and localize float32 rounding by K band."""
    f32 = np.float32
    k_size = A.shape[1]
    alpha32 = f32(alpha)
    beta32 = f32(beta)
    beta_nonzero = float(beta) != 0.0
    c_value = float(C[row, col])

    partial_values: list[np.float32] = []
    exact_product_sum = 0.0
    chunk_merge_rounding = 0.0
    band_abs = [0.0 for _ in k_bands]
    nonzero_count = 0
    product_contributors: list[dict[str, Any]] = []
    local_stage_records: list[dict[str, Any]] = []
    merge_stage_records: list[dict[str, Any]] = []
    position = 0

    for chunk_index, (k_start, k_end) in enumerate(_chunk_ranges(k_size, block_shape.bk)):
        partial = f32(0.0)
        exact_chunk = 0.0
        for k in range(k_start, k_end):
            a_value = A[row, k]
            b_value = B[k, col]
            product_float64 = float(a_value) * float(b_value)
            exact_chunk += product_float64
            exact_product_sum += product_float64

            before = float(partial)
            product_float32 = f32(a_value * b_value)
            after = f32(partial + product_float32)
            local_rounding = float(after) - (before + product_float64)
            partial = after

            band = _band_for_k(k, k_bands)
            band_abs[int(band["index"])] += abs(local_rounding)

            is_nonzero_term = float(a_value) != 0.0 and float(b_value) != 0.0
            if is_nonzero_term:
                nonzero_count += 1
                if collect_details:
                    product_contributors.append(
                        {
                            "accumulation_position": position,
                            "step_number": position + 1,
                            "k": k,
                            "a_index": [row, k],
                            "b_index": [k, col],
                            "a_value": float(a_value),
                            "b_value": float(b_value),
                            "product_float32": float(product_float32),
                            "product_float64": product_float64,
                            "local_rounding_contribution": local_rounding,
                            "absolute_rounding_contribution": abs(local_rounding),
                            "k_band": band["index"],
                        }
                    )
            position += 1

        partial_values.append(partial)
        if collect_details:
            chunk_band = _band_for_k(max(k_start, k_end - 1), k_bands)
            local_stage_records.append(
                {
                    "chunk_index": chunk_index,
                    "k_range": [k_start, k_end],
                    "band_index": int(chunk_band["index"]),
                    "partial_float32": float(partial),
                    "partial_float64": exact_chunk,
                    "local_chunk_rounding": float(partial) - exact_chunk,
                }
            )

    row_sum, merge_events = merge_rounding_for_element(partial_values, loop_order)
    for event in merge_events:
        merge_rounding = float(event["local_merge_rounding"])
        chunk_merge_rounding += merge_rounding
        involved_chunks = tuple(event["left_chunks"]) + tuple(event["right_chunks"])
        representative_chunk = max(involved_chunks)
        representative_k = min(k_size - 1, (representative_chunk + 1) * block_shape.bk - 1)
        representative_band = _band_for_k(representative_k, k_bands)
        band_abs[int(representative_band["index"])] += abs(merge_rounding)
        if collect_details:
            stage_record = dict(event)
            stage_record["merge_k_band_index"] = int(representative_band["index"])
            merge_stage_records.append(stage_record)

    scaled = f32(alpha32 * row_sum)
    if beta_nonzero:
        final_value = f32(scaled + f32(beta32 * f32(c_value)))
        float64_scale_ideal = float(alpha) * float(row_sum) + float(beta) * c_value
    else:
        final_value = scaled
        float64_scale_ideal = float(alpha) * float(row_sum)

    if f32(final_value) != f32(actual_value):
        raise BusinessError(
            code=E_DIAGNOSTIC_MISMATCH,
            stage="error_analysis_trace_replay",
            message="Trace replay does not reproduce the kernel output element.",
            shape={"A": list(A.shape), "B": list(B.shape), "C": list(C.shape)},
            details={
                "row": row,
                "col": col,
                "replayed_value": float(final_value),
                "kernel_value": float(actual_value),
                "loop_order": loop_order,
                "block_shape": block_shape.as_dict(),
            },
        )

    product_rounding = float(row_sum) - exact_product_sum
    scale_rounding = float(final_value) - float64_scale_ideal
    total_rounding = float(final_value) - float(reference_value)
    dominant_band = 0
    if band_abs:
        dominant_band = max(range(len(band_abs)), key=lambda index: (band_abs[index], -index))

    contributors: tuple[dict[str, Any], ...] = ()
    stages: list[dict[str, Any]] = []
    if collect_details:
        total_abs_contrib = sum(
            item["absolute_rounding_contribution"] for item in product_contributors
        )
        for item in product_contributors:
            item["contribution_share_of_abs_rounding"] = (
                item["absolute_rounding_contribution"] / total_abs_contrib
                if total_abs_contrib > 0.0
                else 0.0
            )
        product_contributors.sort(
            key=lambda item: (
                -item["absolute_rounding_contribution"],
                item["accumulation_position"],
            )
        )
        contributors = tuple(dict(item) for item in product_contributors[:top_k])
        worst_chunks = sorted(
            local_stage_records,
            key=lambda item: (-abs(float(item["local_chunk_rounding"])), int(item["chunk_index"])),
        )[:top_k]
        worst_merges = sorted(
            merge_stage_records,
            key=lambda item: (
                -abs(float(item["local_merge_rounding"])),
                int(item["merge_level"]),
                tuple(item["left_chunks"]),
                tuple(item["right_chunks"]),
            ),
        )[:top_k]
        stages.append(
            {
                "stage": "k_chunk_local_accumulation",
                "chunk_count": len(partial_values),
                "local_rounding_l1": sum(
                    abs(float(item["local_chunk_rounding"])) for item in local_stage_records
                ),
                "worst_chunks": [dict(item) for item in worst_chunks],
                "all_chunk_payload_sha256": payload_sha256(local_stage_records),
            }
        )
        stages.append(
            {
                "stage": "chunk_merge",
                "merge_schedule": MERGE_SCHEDULES[loop_order],
                "merge_count": len(merge_stage_records),
                "merge_rounding_l1": sum(
                    abs(float(item["local_merge_rounding"])) for item in merge_stage_records
                ),
                "worst_merges": [dict(item) for item in worst_merges],
                "all_merge_payload_sha256": payload_sha256(merge_stage_records),
            }
        )
        scale_adjustment = (
            float(alpha) * exact_product_sum + (float(beta) * c_value if beta_nonzero else 0.0)
        )
        stages.append(
            {
                "stage": "alpha_beta_finalize",
                "alpha": float(alpha),
                "beta": float(beta),
                "pre_scale_float32_sum": float(row_sum),
                "float64_reference_from_same_sum": float(scale_adjustment),
                "final_float32_value": float(final_value),
                "scale_rounding_vs_float64": scale_rounding,
            }
        )
        stages.append(
            {
                "stage": "total_vs_reference",
                "reference_value": float(reference_value),
                "main_value": float(final_value),
                "total_error": total_rounding,
                "product_accumulation_rounding": product_rounding,
                "chunk_merge_rounding": chunk_merge_rounding,
                "scale_rounding": scale_rounding,
                "unclassified_float64_reference_residual": total_rounding
                - product_rounding
                - scale_rounding,
            }
        )

    band_label = (
        f"k[{k_bands[dominant_band]['start']}:{k_bands[dominant_band]['end']}]"
        if k_bands
        else "k[none]"
    )
    band_total = sum(band_abs)
    band_share = band_abs[dominant_band] / band_total if band_total > 0.0 else 0.0
    stage_parts = {
        "product_accumulation": abs(product_rounding),
        "chunk_merge": abs(chunk_merge_rounding),
        "alpha_beta_scale": abs(scale_rounding),
    }
    dominant_stage = max(stage_parts, key=lambda key: (stage_parts[key], key))
    top_hint = ""
    if contributors:
        top = contributors[0]
        top_hint = (
            f"; largest local term is k={top['k']} at accumulation position "
            f"{top['accumulation_position']} (|rounding|={top['absolute_rounding_contribution']:.3e})"
        )
    conclusion = (
        f"dominant K band {band_label} carries {band_share:.1%} of local |rounding|; "
        f"dominant stage is {dominant_stage}{top_hint}."
    )

    return ElementTrace(
        main_value=float(final_value),
        exact_float64_product_sum=exact_product_sum,
        product_accumulation_rounding=product_rounding,
        chunk_merge_rounding=chunk_merge_rounding,
        scale_rounding=scale_rounding,
        total_rounding_vs_reference=total_rounding,
        k_band_abs_rounding=tuple(band_abs),
        dominant_k_band_index=dominant_band,
        nonzero_term_count=nonzero_count,
        nonzero_term_positions=tuple(
            item["accumulation_position"] for item in product_contributors
        ),
        contributors=contributors,
        propagation_stages=tuple(stages),
        propagation_conclusion=conclusion,
    )
