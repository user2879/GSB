from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .diagnostics import trace_element
from .errors import E_SHAPE_MISMATCH, BusinessError
from .kernel import KernelResult
from .model import AccuracyTolerance, ExperimentConfig, SampleSpec
from .reference import ReferenceResult
from .util import (
    RELATIVE_ERROR_DENOMINATOR_FLOOR,
    build_index_bands,
    payload_sha256,
)
from .validation import prepare_bmm_arrays, prepare_gemm_arrays


@dataclass
class _LayerAccumulator:
    axis: str
    key: str
    start: int | None
    end: int | None
    definition: str
    element_count: int = 0
    exceed_count: int = 0
    abs_error_sum: float = 0.0
    max_relative_error: float = 0.0
    max_absolute_error: float = 0.0
    attribution_numerator: float = 0.0

    def add(self, abs_error: float, rel_error: float, attribution: float, exceeded: bool) -> None:
        self.element_count += 1
        self.exceed_count += int(exceeded)
        self.abs_error_sum += abs_error
        self.attribution_numerator += attribution
        self.max_relative_error = max(self.max_relative_error, rel_error)
        self.max_absolute_error = max(self.max_absolute_error, abs_error)

    def as_dict(self, attribution_denominator: float) -> dict[str, Any]:
        exceed_ratio = self.exceed_count / self.element_count if self.element_count else 0.0
        attribution_share = (
            self.attribution_numerator / attribution_denominator
            if attribution_denominator > 0.0
            else 0.0
        )
        return {
            "axis": self.axis,
            "layer_key": self.key,
            "start": self.start,
            "end": self.end,
            "definition": self.definition,
            "element_count": self.element_count,
            "exceed_count": self.exceed_count,
            "exceed_ratio": exceed_ratio,
            "max_relative_error": self.max_relative_error,
            "max_absolute_error": self.max_absolute_error,
            "abs_error_sum": self.abs_error_sum,
            "attribution_share": attribution_share,
            "conclusion": _layer_conclusion(self.key, exceed_ratio, self.max_relative_error, attribution_share),
        }


def _layer_conclusion(key: str, exceed_ratio: float, max_rel: float, share: float) -> str:
    if exceed_ratio > 0.0:
        return (
            f"{key} exceeds tolerance on {exceed_ratio:.1%} of elements; "
            f"max relative error is {max_rel:.3e} and attribution share is {share:.1%}."
        )
    return (
        f"{key} has no tolerance exceedances; max relative error is {max_rel:.3e} "
        f"and attribution share is {share:.1%}."
    )


def _round_label(axis: str, band: dict[str, int]) -> str:
    return f"{axis}[{band['start']}:{band['end']}]"


def _metadata_index(result: KernelResult) -> dict[tuple[int | None, int, int], dict[str, Any]]:
    index: dict[tuple[int | None, int, int], dict[str, Any]] = {}
    for item in result.element_metadata:
        key = (item["batch_index"], int(item["row"]), int(item["col"]))
        if key in index:
            raise BusinessError(
                code=E_SHAPE_MISMATCH,
                stage="error_analysis_metadata",
                message="Duplicate per-element metadata entry.",
                details={"element": list(key)},
            )
        index[key] = item
    return index


def _dominant_layer(layers: Iterable[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = list(layers)
    if not candidates:
        return None
    return max(candidates, key=lambda item: (float(item[metric]), -int(item.get("start") or 0)))


def _axis_conclusion(axis: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_rel = _dominant_layer(rows, "max_relative_error")
    by_exceed = _dominant_layer(rows, "exceed_ratio")
    by_share = _dominant_layer(rows, "attribution_share")
    if by_exceed is not None and float(by_exceed["exceed_ratio"]) > 0.0:
        conclusion = (
            f"{axis} sensitivity is led by {by_exceed['layer_key']} on exceedance ratio "
            f"({by_exceed['exceed_ratio']:.1%}); max-relative leader is "
            f"{by_rel['layer_key'] if by_rel else 'n/a'}; attribution leader is "
            f"{by_share['layer_key'] if by_share else 'n/a'}."
        )
    else:
        conclusion = (
            f"{axis} has no exceedances in this combination; max-relative leader is "
            f"{by_rel['layer_key'] if by_rel else 'n/a'} and attribution leader is "
            f"{by_share['layer_key'] if by_share else 'n/a'}."
        )
    return {
        "axis": axis,
        "dominant_by_max_relative_error": by_rel["layer_key"] if by_rel else None,
        "dominant_by_exceed_ratio": by_exceed["layer_key"] if by_exceed else None,
        "dominant_by_attribution_share": by_share["layer_key"] if by_share else None,
        "conclusion": conclusion,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _prepare_for_mode(
    mode: str,
    A: Any,
    B: Any,
    C: Any,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if mode == "gemm":
        return prepare_gemm_arrays(A, B, C, alpha, beta)
    if mode == "bmm":
        return prepare_bmm_arrays(A, B, C, alpha, beta)
    raise ValueError(f"Unsupported mode: {mode}")


def analyse_combination(
    A: Any,
    B: Any,
    C: Any,
    main: KernelResult,
    reference: ReferenceResult,
    tolerance: AccuracyTolerance,
    config: ExperimentConfig,
    sample: SampleSpec,
) -> dict[str, Any]:
    tolerance.validate(stage="error_analysis_tolerance_validation")
    if main.mode != reference.mode:
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="error_analysis_mode_validation",
            message="Main path and reference mode differ.",
            details={"main_mode": main.mode, "reference_mode": reference.mode},
        )
    if main.c.shape != reference.c.shape:
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="error_analysis_shape_validation",
            message="Main and reference output shapes differ.",
            shape={"main": list(main.c.shape), "reference": list(reference.c.shape)},
        )

    A32, B32, C32, alpha_value, beta_value = _prepare_for_mode(
        main.mode, A, B, C, sample.alpha, sample.beta
    )
    if main.input_sha256 != reference.input_sha256:
        raise BusinessError(
            code=E_SHAPE_MISMATCH,
            stage="error_analysis_input_validation",
            message="Main and reference were not computed from the same inputs.",
        )

    metadata = _metadata_index(main)
    M, N = (int(main.c.shape[0]), int(main.c.shape[1])) if main.mode == "gemm" else (
        int(main.c.shape[1]),
        int(main.c.shape[2]),
    )
    batch_size = 1 if main.mode == "gemm" else int(main.c.shape[0])
    K = int(A32.shape[1]) if main.mode == "gemm" else int(A32.shape[2])

    m_bands = build_index_bands(M, config.layer_count)
    n_bands = build_index_bands(N, config.layer_count)
    k_bands = build_index_bands(K, config.layer_count)
    batch_bands = build_index_bands(batch_size, config.layer_count) if batch_size > 1 else [
        {"index": 0, "start": 0, "end": 1}
    ]

    layer_map: dict[str, dict[str, _LayerAccumulator]] = {
        "M": {
            _round_label("m", band): _LayerAccumulator(
                "M", _round_label("m", band), band["start"], band["end"], "contiguous_M_band"
            )
            for band in m_bands
        },
        "N": {
            _round_label("n", band): _LayerAccumulator(
                "N", _round_label("n", band), band["start"], band["end"], "contiguous_N_band"
            )
            for band in n_bands
        },
        "K": {
            _round_label("k", band): _LayerAccumulator(
                "K",
                _round_label("k", band),
                band["start"],
                band["end"],
                "dominant_local_rounding_K_band",
            )
            for band in k_bands
        },
        "BATCH": {
            f"batch[{band['start']}:{band['end']}]": _LayerAccumulator(
                "BATCH",
                f"batch[{band['start']}:{band['end']}]",
                band["start"],
                band["end"],
                "batch_index_band",
            )
            for band in batch_bands
        },
    }

    abs_errors: list[float] = []
    total_abs_error = 0.0
    total_k_rounding = 0.0
    k_band_attribution_totals = [0.0 for _ in k_bands]
    exceed_details: list[dict[str, Any]] = []
    nonzero_counts: list[int] = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    exceed_count = 0

    for batch_index in range(batch_size):
        public_batch = batch_index if main.mode == "bmm" else None
        A2 = A32[batch_index] if main.mode == "bmm" else A32
        B2 = B32[batch_index] if main.mode == "bmm" else B32
        C2 = C32[batch_index] if main.mode == "bmm" else C32
        main2 = main.c[batch_index] if main.mode == "bmm" else main.c
        ref2 = reference.c[batch_index] if main.mode == "bmm" else reference.c
        for row in range(M):
            for col in range(N):
                main_value = float(main2[row, col])
                reference_value = float(ref2[row, col])
                abs_error = abs(main_value - reference_value)
                denominator = max(abs(reference_value), RELATIVE_ERROR_DENOMINATOR_FLOOR)
                relative_error = abs_error / denominator
                threshold = tolerance.threshold(reference_value)
                exceeded = abs_error > threshold
                meta = metadata.get((public_batch, row, col))
                if meta is None:
                    raise BusinessError(
                        code=E_SHAPE_MISMATCH,
                        stage="error_analysis_metadata",
                        message="Missing per-element accumulation metadata.",
                        details={"batch_index": public_batch, "row": row, "col": col},
                    )

                trace = trace_element(
                    A2,
                    B2,
                    C2,
                    row=row,
                    col=col,
                    block_shape=main.block_shape,
                    loop_order=main.loop_order,
                    alpha=alpha_value,
                    beta=beta_value,
                    reference_value=reference_value,
                    actual_value=main_value,
                    k_bands=k_bands,
                    top_k=config.top_k_contributors,
                    collect_details=exceeded,
                )
                band_values = list(trace.k_band_abs_rounding)
                for index, value in enumerate(band_values):
                    k_band_attribution_totals[index] += value
                total_k_rounding += sum(band_values)

                m_band = next(band for band in m_bands if band["start"] <= row < band["end"])
                n_band = next(band for band in n_bands if band["start"] <= col < band["end"])
                batch_band = next(
                    band for band in batch_bands if band["start"] <= batch_index < band["end"]
                )
                k_band = k_bands[trace.dominant_k_band_index]
                layer_map["M"][_round_label("m", m_band)].add(
                    abs_error, relative_error, abs_error, exceeded
                )
                layer_map["N"][_round_label("n", n_band)].add(
                    abs_error, relative_error, abs_error, exceeded
                )
                layer_map["K"][_round_label("k", k_band)].add(
                    abs_error, relative_error, sum(band_values), exceeded
                )
                batch_key = f"batch[{batch_band['start']}:{batch_band['end']}]"
                layer_map["BATCH"][batch_key].add(
                    abs_error, relative_error, abs_error, exceeded
                )

                abs_errors.append(abs_error)
                total_abs_error += abs_error
                nonzero_counts.append(trace.nonzero_term_count)
                max_abs_error = max(max_abs_error, abs_error)
                max_rel_error = max(max_rel_error, relative_error)
                if exceeded:
                    exceed_count += 1
                    exceed_details.append(
                        {
                            "batch_index": public_batch,
                            "row": row,
                            "col": col,
                            "main_value": main_value,
                            "reference_value": reference_value,
                            "absolute_error": abs_error,
                            "relative_error": relative_error,
                            "relative_error_denominator_floor": RELATIVE_ERROR_DENOMINATOR_FLOOR,
                            "tolerance_threshold": threshold,
                            "loop_order": meta["loop_order"],
                            "stable_task_key": meta["stable_task_key"],
                            "accumulation_order_id": meta["accumulation_order_id"],
                            "chain_length": meta["chain_length"],
                            "k_chunk_count": meta["k_chunk_count"],
                            "rounding_step_count": meta["rounding_step_count"],
                            "nonzero_term_count": trace.nonzero_term_count,
                            "dominant_k_band": _round_label("k", k_band),
                            "rounding_by_k_band": [
                                {
                                    "layer_key": _round_label("k", band),
                                    "start": int(band["start"]),
                                    "end": int(band["end"]),
                                    "abs_rounding_contribution": band_values[band["index"]],
                                }
                                for band in k_bands
                            ],
                            "product_accumulation_rounding": trace.product_accumulation_rounding,
                            "chunk_merge_rounding": trace.chunk_merge_rounding,
                            "scale_rounding": trace.scale_rounding,
                            "top_contributors": [dict(item) for item in trace.contributors],
                            "propagation_stages": [dict(item) for item in trace.propagation_stages],
                            "propagation_conclusion": trace.propagation_conclusion,
                        }
                    )

    exceed_details.sort(
        key=lambda item: (
            -1 if item["batch_index"] is None else int(item["batch_index"]),
            int(item["row"]),
            int(item["col"]),
        )
    )
    k_band_total = sum(k_band_attribution_totals)
    layers_by_axis: dict[str, list[dict[str, Any]]] = {}
    layer_exceedances: list[dict[str, Any]] = []
    for axis, layer_dict in layer_map.items():
        rows: list[dict[str, Any]] = []
        for key, accumulator in layer_dict.items():
            if axis == "K":
                band_index = next(
                    band["index"]
                    for band in k_bands
                    if _round_label("k", band) == key
                )
                denominator = k_band_attribution_totals[band_index]
                numerator_override = k_band_attribution_totals[band_index]
                item = accumulator.as_dict(1.0)
                item["attribution_share"] = (
                    numerator_override / k_band_total if k_band_total > 0.0 else 0.0
                )
                item["conclusion"] = _layer_conclusion(
                    key,
                    float(item["exceed_ratio"]),
                    float(item["max_relative_error"]),
                    float(item["attribution_share"]),
                )
            else:
                item = accumulator.as_dict(total_abs_error)
            rows.append(item)
            if float(item["exceed_ratio"]) > float(config.layer_exceed_limit):
                layer_exceedances.append(
                    {
                        "axis": axis,
                        "layer_key": item["layer_key"],
                        "exceed_ratio": item["exceed_ratio"],
                        "exceed_count": item["exceed_count"],
                        "element_count": item["element_count"],
                        "configured_limit": float(config.layer_exceed_limit),
                        "block_shape": main.block_shape.as_dict(),
                        "loop_order": main.loop_order,
                    }
                )
        layers_by_axis[axis] = rows

    axis_conclusions = [
        _axis_conclusion(axis, layers_by_axis[axis])
        for axis in ("M", "N", "K", "BATCH")
    ]
    exceed_details.sort(key=lambda item: (
        -1 if item["batch_index"] is None else int(item["batch_index"]),
        int(item["row"]),
        int(item["col"]),
    ))
    layer_exceedances.sort(
        key=lambda item: (
            item["axis"],
            -float(item["exceed_ratio"]),
            item["layer_key"],
            item["loop_order"],
            item["block_shape"]["BM"],
            item["block_shape"]["BN"],
            item["block_shape"]["BK"],
        )
    )

    chain_length_histogram: dict[str, int] = {}
    rounding_step_histogram: dict[str, int] = {}
    nonzero_histogram: dict[str, int] = {}
    for meta in main.element_metadata:
        chain_key = str(meta["chain_length"])
        rounding_key = str(meta["rounding_step_count"])
        chain_length_histogram[chain_key] = chain_length_histogram.get(chain_key, 0) + 1
        rounding_step_histogram[rounding_key] = rounding_step_histogram.get(rounding_key, 0) + 1
    for count in nonzero_counts:
        key = str(count)
        nonzero_histogram[key] = nonzero_histogram.get(key, 0) + 1

    error_summary = {
        "element_count": len(abs_errors),
        "max_absolute_error": max_abs_error,
        "max_relative_error": max_rel_error,
        "mean_absolute_error": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(abs_errors)))) if abs_errors else 0.0,
        "p95_absolute_error": _percentile(abs_errors, 95.0),
        "exceed_count": exceed_count,
        "exceed_ratio": exceed_count / len(abs_errors) if abs_errors else 0.0,
        "tolerance": tolerance.as_dict(),
        "relative_error_denominator_floor": RELATIVE_ERROR_DENOMINATOR_FLOOR,
    }

    deterministic_payload = {
        "sample": sample.name,
        "sample_category": sample.category,
        "shapes": sample.shape_dict(),
        "alpha": float(alpha_value),
        "beta": float(beta_value),
        "block_shape": main.block_shape.as_dict(),
        "loop_order": main.loop_order,
        "kernel": main.summary(),
        "reference": reference.summary(),
        "error_summary": error_summary,
        "chain_length_distribution": {
            "mathematical_chain_length": dict(sorted(chain_length_histogram.items(), key=lambda kv: int(kv[0]))),
            "rounding_step_count": dict(sorted(rounding_step_histogram.items(), key=lambda kv: int(kv[0]))),
            "nonzero_term_count": dict(sorted(nonzero_histogram.items(), key=lambda kv: int(kv[0]))),
        },
        "k_bands": [dict(band) for band in k_bands],
        "layer_tables": layers_by_axis,
        "layer_conclusions": axis_conclusions,
        "layer_exceedances": layer_exceedances,
        "exceeded_elements": exceed_details,
        "element_trace_identifiers": [dict(item) for item in main.element_metadata],
        "tasks": [task.as_dict() for task in main.tasks],
        "total_abs_error": total_abs_error,
        "total_abs_k_band_rounding": total_k_rounding,
    }
    combination = dict(deterministic_payload)
    combination["execution_max_workers"] = int(main.max_workers)
    combination["audit_digest"] = payload_sha256(deterministic_payload)
    return combination


def combination_audit_row(combination: dict[str, Any]) -> dict[str, Any]:
    summary = combination["error_summary"]
    return {
        "sample": combination["sample"],
        "block_shape": combination["block_shape"],
        "loop_order": combination["loop_order"],
        "max_absolute_error": summary["max_absolute_error"],
        "max_relative_error": summary["max_relative_error"],
        "mean_absolute_error": summary["mean_absolute_error"],
        "p95_absolute_error": summary["p95_absolute_error"],
        "exceed_count": summary["exceed_count"],
        "exceed_ratio": summary["exceed_ratio"],
        "rounding_step_count": combination["chain_length_distribution"][
            "rounding_step_count"
        ],
        "layer_exceedance_count": len(combination["layer_exceedances"]),
        "audit_digest": combination["audit_digest"],
    }
