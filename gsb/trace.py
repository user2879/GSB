"""Fixed-point staged error tracing and recognition-flip attribution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from gsb.config import BudgetConfig, QuantizationConfig, RunConfig
from gsb.fonts import TemplateLibrary
from gsb.models import DatasetState
from gsb.pipeline import PipelineResult, run_pipeline
from gsb.util import error_measure, first_exceeding_stage, round_half_up

STAGE_KEYPOINT = "keypoint_coordinate_quantization"
STAGE_MATCH = "template_match_score_rounding"
STAGE_CONFIDENCE = "confidence_composition"
STAGE_READING = "reading_order_adjudication"


def _by_keypoint(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for sample in report.get("samples", []):
        for recognition in sample.get("recognitions", []):
            output[recognition["keypoint_id"]] = {
                "sample_id": sample["sample_id"],
                "recognition": recognition,
            }
    return output


def _keypoints_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for sample in report.get("samples", []):
        for keypoint in sample.get("keypoints", []):
            output[keypoint["keypoint_id"]] = {
                **keypoint,
                "sample_id": sample["sample_id"],
            }
    return output


def _reading_positions(report: dict[str, Any]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for sample in report.get("samples", []):
        for entry in sample.get("reading_order", []):
            positions[entry["keypoint_id"]] = entry["position"]
    return positions


def _reference_run(
    dataset: DatasetState,
    *,
    config: RunConfig,
    templates: TemplateLibrary,
    reference_bits: int,
) -> PipelineResult:
    reference_config = replace(
        config,
        quantization=QuantizationConfig(
            level=f"REF_Q{reference_bits}",
            fixed_bits=reference_bits,
        ),
        budget=BudgetConfig(max_template_points=10**12),
    )
    return run_pipeline(dataset, config=reference_config, templates=templates)


def _stage_error(
    reference: int,
    observed: int,
    *,
    stage: str,
    note: str,
) -> dict[str, Any]:
    measurement = error_measure(reference, observed, truth_zero=(reference == 0))
    return {
        "stage": stage,
        "reference": reference,
        "observed": observed,
        "error": measurement["absolute_error"],
        "error_branch": measurement["branch"],
        "relative_error_ppm": measurement["relative_error_ppm"],
        "truth_zero": reference == 0,
        "note": note,
    }


def trace_reference_errors(
    dataset: DatasetState,
    *,
    config: RunConfig,
    templates: TemplateLibrary,
    base_result: PipelineResult,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_bits = config.fixed_bits + 1
    reference = _reference_run(
        dataset,
        config=config,
        templates=templates,
        reference_bits=reference_bits,
    )
    base_recognition = _by_keypoint(base_result.report)
    ref_recognition = _by_keypoint(reference.report)
    base_keypoints = _keypoints_by_id(base_result.report)
    ref_keypoints = _keypoints_by_id(reference.report)
    base_read = _reading_positions(base_result.report)
    ref_read = _reading_positions(reference.report)
    tolerance = config.thresholds.tolerance_fixed << (reference_bits - config.fixed_bits)
    records: list[dict[str, Any]] = []
    stage_counts = {
        STAGE_KEYPOINT: 0,
        STAGE_MATCH: 0,
        STAGE_CONFIDENCE: 0,
        STAGE_READING: 0,
        "none": 0,
    }
    for keypoint_id in sorted(base_keypoints):
        base_kp = base_keypoints[keypoint_id]
        ref_kp = ref_keypoints.get(keypoint_id)
        base_rec = base_recognition.get(keypoint_id, {}).get("recognition", {})
        ref_rec = ref_recognition.get(keypoint_id, {}).get("recognition", {})
        if ref_kp is None or not ref_rec:
            continue
        observed_x = int(base_kp["x"]) << reference_bits
        observed_y = int(base_kp["y"]) << reference_bits
        ref_x = int(ref_kp["raw_fixed"][0])
        ref_y = int(ref_kp["raw_fixed"][1])
        keypoint_error = max(abs(ref_x - observed_x), abs(ref_y - observed_y))
        coordinate_stage = _stage_error(
            ref_x,
            observed_x,
            stage=STAGE_KEYPOINT,
            note="x fixed-point coordinates; y error reported separately below",
        )
        coordinate_stage["y_reference"] = ref_y
        coordinate_stage["y_observed"] = observed_y
        coordinate_stage["y_error"] = abs(ref_y - observed_y)
        base_score = base_rec.get("score")
        ref_score = ref_rec.get("score")
        if isinstance(base_score, int) and isinstance(ref_score, int):
            observed_score = base_score << (reference_bits - config.fixed_bits)
        else:
            observed_score = 0
            ref_score = 0
        match_stage = _stage_error(
            ref_score,
            observed_score,
            stage=STAGE_MATCH,
            note="template fusion score converted to reference fixed scale",
        )
        base_confidence = int(base_rec.get("confidence", 0) or 0)
        ref_confidence = int(ref_rec.get("confidence", 0) or 0)
        confidence_stage = _stage_error(
            ref_confidence,
            base_confidence,
            stage=STAGE_CONFIDENCE,
            note="confidence ppm difference converted to reference fixed fraction",
        )
        confidence_stage["raw_ppm_error"] = abs(ref_confidence - base_confidence)
        confidence_stage["error"] = round_half_up(
            abs(ref_confidence - base_confidence) << reference_bits,
            1_000_000,
        )
        base_position = base_read.get(keypoint_id, 0)
        ref_position = ref_read.get(keypoint_id, 0)
        reading_stage = _stage_error(
            ref_position,
            base_position,
            stage=STAGE_READING,
            note="reading position gap scaled by one reference fixed unit per position",
        )
        if ref_position == 0:
            reading_stage["error_branch"] = "absolute"
            reading_stage["truth_zero"] = True
        reading_stage["error"] = abs(ref_position - base_position) * (1 << reference_bits)
        stages = [coordinate_stage, match_stage, confidence_stage, reading_stage]
        first = first_exceeding_stage(
            (
                (STAGE_KEYPOINT, keypoint_error),
                (STAGE_MATCH, match_stage["error"]),
                (STAGE_CONFIDENCE, confidence_stage["error"]),
                (STAGE_READING, reading_stage["error"]),
            ),
            tolerance,
        )
        stage_counts[first or "none"] += 1
        records.append(
            {
                "keypoint_id": keypoint_id,
                "sample_id": base_kp.get("sample_id"),
                "base_char": base_rec.get("char"),
                "reference_char": ref_rec.get("char"),
                "recognition_equal": base_rec.get("char") == ref_rec.get("char"),
                "tolerance_reference": tolerance,
                "stages": stages,
                "first_exceeding_stage": first,
            }
        )
    return records, {
        "reference_bits": reference_bits,
        "reference_quantization": "high_precision_unquantized_reference",
        "tolerance_reference": tolerance,
        "first_exceeding_counts": stage_counts,
    }


def _score_margin(recognition: dict[str, Any]) -> int:
    scores = sorted(
        (
            int(item.get("score", 0))
            for item in recognition.get("template_scores", [])
        ),
        reverse=True,
    )
    if len(scores) < 2:
        return 0
    return scores[0] - scores[1]


def attribute_flips(
    base_result: PipelineResult,
    alternate_results: dict[str, PipelineResult],
) -> list[dict[str, Any]]:
    base = _by_keypoint(base_result.report)
    base_keypoints = _keypoints_by_id(base_result.report)
    base_read = _reading_positions(base_result.report)
    flips: list[dict[str, Any]] = []
    for level in sorted(alternate_results):
        alternate = _by_keypoint(alternate_results[level].report)
        alternate_keypoints = _keypoints_by_id(alternate_results[level].report)
        alternate_read = _reading_positions(alternate_results[level].report)
        for keypoint_id in sorted(base):
            base_rec = base[keypoint_id]["recognition"]
            alt_rec = alternate.get(keypoint_id, {}).get("recognition")
            if alt_rec is None or base_rec.get("char") == alt_rec.get("char"):
                continue
            base_position = base_read.get(keypoint_id)
            alt_position = alternate_read.get(keypoint_id)
            base_kp = base_keypoints[keypoint_id]
            alt_kp = alternate_keypoints[keypoint_id]
            if base_position != alt_position:
                source = "reading_order_adjudication"
                evidence = "reading positions differ"
            elif base_rec.get("block_bbox") != alt_rec.get("block_bbox"):
                source = "keypoint_coordinate_quantization"
                evidence = "quantized keypoint changes the recognition block"
            elif base_kp.get("quantized_fixed") != alt_kp.get("quantized_fixed"):
                source = "template_match_score_rounding"
                evidence = "same block geometry; score quantization changes ranking"
            else:
                source = "template_match_score_rounding"
                evidence = "score rounding changes the top template"
            flips.append(
                {
                    "keypoint_id": keypoint_id,
                    "sample_id": base[keypoint_id]["sample_id"],
                    "base_level": base_rec.get("quantization_level"),
                    "alternate_level": level,
                    "base_char": base_rec.get("char"),
                    "alternate_char": alt_rec.get("char"),
                    "source": source,
                    "evidence": evidence,
                    "base_confidence": base_rec.get("confidence"),
                    "alternate_confidence": alt_rec.get("confidence"),
                    "base_score_margin": _score_margin(base_rec),
                    "alternate_score_margin": _score_margin(alt_rec),
                    "base_reading_position": base_position,
                    "alternate_reading_position": alt_position,
                    "base_block_bbox": base_rec.get("block_bbox"),
                    "alternate_block_bbox": alt_rec.get("block_bbox"),
                    "keypoint_quantized_fixed": {
                        "base": base_kp.get("quantized_fixed"),
                        "alternate": alt_kp.get("quantized_fixed"),
                    },
                }
            )
    flips.sort(key=lambda item: (item["sample_id"], item["keypoint_id"], item["alternate_level"]))
    return flips


def analyze_attribution(
    dataset: DatasetState,
    *,
    config: RunConfig,
    templates: TemplateLibrary,
    base_result: PipelineResult,
    alternate_levels: Iterable[str] = ("Q4", "Q6", "Q10"),
) -> dict[str, Any]:
    stage_records, reference_summary = trace_reference_errors(
        dataset,
        config=config,
        templates=templates,
        base_result=base_result,
    )
    alternate_results: dict[str, PipelineResult] = {}
    for level in alternate_levels:
        if level == config.quantization.level:
            continue
        alternate_config = replace(
            config.with_quantization(level),
            budget=BudgetConfig(max_template_points=10**12),
        )
        alternate_results[level] = run_pipeline(
            dataset,
            config=alternate_config,
            templates=templates,
        )
    flips = attribute_flips(base_result, alternate_results)
    return {
        "reference": reference_summary,
        "per_character": stage_records,
        "alternate_levels": sorted(alternate_results),
        "flips": flips,
        "flip_count": len(flips),
    }
