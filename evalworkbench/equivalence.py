"""Field-by-field cross-carrier equivalence with explicit uncertainty classes."""

from __future__ import annotations

from typing import Any

from .errors import BusinessError
from .formats import CarrierModel
from .util import hash_object, numeric_equal


CARRIER_ORDER = ("project_json", "metric_binary", "legacy_json")
PAIR_ORDER = (
    ("project_json", "metric_binary"),
    ("project_json", "legacy_json"),
    ("metric_binary", "legacy_json"),
)


def _frame_map(model: CarrierModel) -> dict[int, dict[str, Any]]:
    return {int(frame["frame_index"]): frame for frame in model.declared_frames}


def _compare_value(value_a: Any, value_b: Any, path: str, tolerance: float) -> str | None:
    if isinstance(value_a, (int, float)) and not isinstance(value_a, bool) and isinstance(value_b, (int, float)) and not isinstance(value_b, bool):
        return None if numeric_equal(value_a, value_b, tolerance) else path
    if type(value_a) is not type(value_b):
        return path
    if isinstance(value_a, dict):
        keys = sorted(set(value_a) | set(value_b))
        for key in keys:
            if key not in value_a or key not in value_b:
                return f"{path}.{key}"
            mismatch = _compare_value(value_a[key], value_b[key], f"{path}.{key}", tolerance)
            if mismatch:
                return mismatch
        return None
    if isinstance(value_a, list):
        if len(value_a) != len(value_b):
            return f"{path}.length"
        for index, (item_a, item_b) in enumerate(zip(value_a, value_b, strict=True)):
            mismatch = _compare_value(item_a, item_b, f"{path}[{index}]", tolerance)
            if mismatch:
                return mismatch
        return None
    return None if value_a == value_b else path


def _compare_models(left: CarrierModel, right: CarrierModel, tolerance: float = 1e-6) -> str | None:
    fields = (
        ("content", left.content, right.content),
        ("encoding_params", left.encoding_params, right.encoding_params),
        ("shots", left.shots, right.shots),
        ("metric_policy", left.metric_policy, right.metric_policy),
    )
    for path, value_a, value_b in fields:
        mismatch = _compare_value(value_a, value_b, path, tolerance)
        if mismatch:
            return mismatch
    left_frames = _frame_map(left)
    right_frames = _frame_map(right)
    if set(left_frames) != set(right_frames):
        return "frames.index_set"
    for frame_index in sorted(left_frames):
        frame_fields = ("frame_index", "shot_id", "timestamp_ms", "psnr_db", "ssim", "chroma_psnr_db", "flags")
        for field in frame_fields:
            value_a = left_frames[frame_index].get(field)
            value_b = right_frames[frame_index].get(field)
            mismatch = _compare_value(value_a, value_b, f"frames[{frame_index}].{field}", tolerance)
            if mismatch:
                return mismatch
    left_segments = {int(item["shot_id"]): item for item in left.segments}
    right_segments = {int(item["shot_id"]): item for item in right.segments}
    if set(left_segments) != set(right_segments):
        return "segments.shot_set"
    for shot_id in sorted(left_segments):
        for field in ("start_frame", "end_frame", "frame_count", "valid_frame_count", "complete", "psnr_db", "ssim", "policy_hash"):
            mismatch = _compare_value(
                left_segments[shot_id].get(field),
                right_segments[shot_id].get(field),
                f"segments[{shot_id}].{field}",
                tolerance,
            )
            if mismatch:
                return mismatch
    return None


def compare_carriers(
    models: dict[str, CarrierModel],
    errors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    content_hash = None
    for name in CARRIER_ORDER:
        if name in models:
            content_hash = models[name].content.get("content_hash")
            break
    for left_name, right_name in PAIR_ORDER:
        left_error = errors.get(left_name)
        right_error = errors.get(right_name)
        pair: dict[str, Any] = {
            "left": left_name,
            "right": right_name,
            "status": "equivalent",
            "first_mismatch": None,
            "semantic_uncertainty": [],
            "conditions": {},
        }
        if left_error or right_error:
            pair["status"] = "invalid_carrier"
            pair["first_mismatch"] = f"carrier.{left_name if left_error else right_name}.parse"
            pair["errors"] = {
                side: error
                for side, error in ((left_name, left_error), (right_name, right_error))
                if error is not None
            }
        else:
            left = models[left_name]
            right = models[right_name]
            mismatch = _compare_models(left, right)
            if mismatch:
                pair["status"] = "mismatch"
                pair["first_mismatch"] = mismatch
            assumptions = left.assumptions + right.assumptions
            if assumptions:
                pair["semantic_uncertainty"] = assumptions
                if not mismatch:
                    pair["status"] = "equivalent_with_assumptions"
            pair["conditions"] = {
                left_name: left.provenance,
                right_name: right.provenance,
            }
        pair["digest"] = hash_object(pair)
        pairs.append(pair)
    invalid = any(pair["status"] == "invalid_carrier" for pair in pairs)
    mismatch = any(pair["status"] == "mismatch" for pair in pairs)
    with_assumptions = any(pair["status"] == "equivalent_with_assumptions" for pair in pairs)
    report_core = {
        "schema_version": "carrier-equivalence/1.0",
        "content_hash": content_hash,
        "pairs": pairs,
        "overall": {
            "equivalent": not invalid and not mismatch,
            "strictly_equivalent": not invalid and not mismatch and not with_assumptions,
            "equivalent_with_assumptions": not invalid and not mismatch and with_assumptions,
            "invalid_carrier": invalid,
            "mismatch": mismatch,
            "business_code": "E_EQUIV_MISMATCH" if invalid or mismatch else None,
        },
        "first_mismatch": next((pair["first_mismatch"] for pair in pairs if pair["first_mismatch"]), None),
    }
    report_core["report_digest"] = hash_object(report_core)
    return report_core


def compare_model_to_actual(
    model: CarrierModel,
    actual_frames: list[dict[str, Any]],
    actual_segments: list[dict[str, Any]],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    actual_model = CarrierModel(
        carrier_type="actual_decode",
        schema_version=model.schema_version,
        content=model.content,
        encoding_params=model.encoding_params,
        shots=model.shots,
        metric_policy=model.metric_policy,
        declared_frames=actual_frames,
        segments=actual_segments,
    )
    mismatch = _compare_models(model, actual_model, tolerance)
    result = {
        "carrier": model.carrier_type,
        "status": "equivalent" if mismatch is None else "mismatch",
        "first_mismatch": mismatch,
        "declared_frame_count": len(model.declared_frames),
        "actual_frame_count": len(actual_frames),
        "semantic_uncertainty": model.assumptions,
    }
    result["digest"] = hash_object(result)
    return result


def require_equivalence(report: dict[str, Any]) -> None:
    if not report["overall"]["equivalent"]:
        raise BusinessError(
            "E_EQUIV_MISMATCH",
            "carrier equivalence failed",
            {
                "first_mismatch": report.get("first_mismatch"),
                "pairs": [
                    {"left": pair["left"], "right": pair["right"], "status": pair["status"], "first_mismatch": pair["first_mismatch"]}
                    for pair in report["pairs"]
                ],
            },
        )
