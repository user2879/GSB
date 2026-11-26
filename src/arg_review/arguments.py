from __future__ import annotations

from fractions import Fraction
from typing import Any

from .canonical import fingerprint
from .errors import StageInputError
from .quantization import quantize_score
from .validation import is_plain_int


def token_distance(
    arg_start: int, arg_end: int, trigger_start: int, trigger_end: int
) -> int:
    if arg_end <= trigger_start:
        return trigger_start - arg_end
    if trigger_end <= arg_start:
        return arg_start - trigger_end
    return 0


def _fraction_payload(value: Fraction) -> dict[str, int | str]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": f"{float(value):.9f}",
    }


def argument_cache_key(material: dict[str, Any]) -> str:
    return fingerprint({"artifact": "argument_candidate", **material})


def argument_key_parts(
    event: dict[str, Any],
    argument: dict[str, Any],
    trigger_result: dict[str, Any] | None,
    revision: dict[str, Any],
    config: dict[str, Any],
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    role = argument.get("role")
    raw_confidence = argument.get("confidence_permille")
    extraction = config.get("extraction", {})
    quant_config = config.get("quantization", {})
    role_weights = config.get("role_weights_permille", {})
    schema_event_type = revision["event_types"][event["event_type"]]
    allowed_roles = sorted(schema_event_type.get("roles", []))
    role_weight = int(role_weights.get(role, 600)) if isinstance(role, str) else 600
    arg_content_fp = fingerprint(
        {
            "arg_id": argument.get("arg_id"),
            "role": role,
            "start": argument.get("start"),
            "end": argument.get("end"),
            "confidence_permille": raw_confidence,
        },
        16,
    )
    rule_slice_fp = fingerprint(
        {
            "role": role,
            "role_weight_permille": role_weight,
            "extraction": extraction,
            "allowed_roles": allowed_roles,
        },
        16,
    )
    quant_fp = fingerprint(quant_config, 16)
    event_content_fp = (
        trigger_result["content_fingerprint"]
        if trigger_result
        else fingerprint({"event_id": event.get("event_id"), "tokens": event.get("tokens", [])}, 16)
    )
    material = {
        "event_content_fingerprint": event_content_fp,
        "argument_content_fingerprint": arg_content_fp,
        "trigger_cache_key": (trigger_result or {}).get("cache_key"),
            "role_rule_revision_id": (
                scope["role_rule_revision_id"] if scope else revision["revision_id"]
            ),
            "role_rule_revision_version": (
                scope["role_rule_revision_version"] if scope else revision["version"]
            ),
        "argument_rule_slice_fingerprint": rule_slice_fp,
        "quantization_fingerprint": quant_fp,
        "algorithm": "fixed_point_role_distance_length_raw/v1",
    }
    return {
        "cache_key": argument_cache_key(material),
        "material": material,
        "arg_content_fingerprint": arg_content_fp,
        "rule_pack_fingerprint": rule_slice_fp,
        "quantization_fingerprint": quant_fp,
        "role_weight_permille": role_weight,
    }


def extract_event_arguments(
    event: dict[str, Any],
    trigger_result: dict[str, Any] | None,
    schema_event_type: dict[str, Any],
    revision: dict[str, Any],
    config: dict[str, Any],
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_id = event["event_id"]
    extraction = config.get("extraction", {})
    quant_config = config.get("quantization", {})
    role_weights = config.get("role_weights_permille", {})
    allowed_roles = set(schema_event_type.get("roles", []))
    window = int(extraction.get("window_tokens", 6))
    min_tokens = int(extraction.get("fragment_min_tokens", 1))
    max_tokens = int(extraction.get("fragment_max_tokens", 4))
    ideal_tokens = int(extraction.get("ideal_fragment_tokens", 2))
    raw_w = int(extraction.get("raw_weight_permille", 500))
    role_w = int(extraction.get("role_weight_permille", 300))
    distance_w = int(extraction.get("distance_weight_permille", 100))
    length_w = int(extraction.get("length_weight_permille", 100))

    matches = (trigger_result or {}).get("matches", [])
    arguments = event.get("arguments", [])
    ordered_arguments = sorted(
        arguments,
        key=lambda arg: (
            str(arg.get("role", "")),
            arg.get("start") if is_plain_int(arg.get("start")) else -1,
            arg.get("end") if is_plain_int(arg.get("end")) else -1,
            str(arg.get("arg_id", "")),
        ),
    )

    candidates: list[dict[str, Any]] = []
    for argument in ordered_arguments:
        arg_id = argument.get("arg_id")
        role = argument.get("role")
        start = argument.get("start")
        end = argument.get("end")
        raw_confidence = argument.get("confidence_permille")

        if not isinstance(role, str) or role not in allowed_roles:
            raise StageInputError(
                f"role_not_in_set:{event_id}/{arg_id}/{role}",
                stage_code="STAGE_ARGUMENT_ROLE_INVALID",
            )
        token_count = len(event.get("tokens", []))
        if not is_plain_int(start) or not is_plain_int(end):
            raise StageInputError(
                f"offset_non_integer:{event_id}/{arg_id}",
                stage_code="STAGE_ARGUMENT_OFFSET_INVALID",
            )
        if not (0 <= start <= end <= token_count):
            raise StageInputError(
                f"offset_out_of_bounds:{event_id}/{arg_id}:{start},{end}",
                stage_code="STAGE_ARGUMENT_OFFSET_INVALID",
            )
        if not is_plain_int(raw_confidence) or not (0 <= raw_confidence <= 1000):
            raise StageInputError(
                f"confidence_not_permille:{event_id}/{arg_id}:{raw_confidence}",
                stage_code="STAGE_ARGUMENT_CONFIDENCE_INVALID",
            )

        length = end - start
        nearest = None
        distance = window + 1
        for match in matches:
            candidate_distance = token_distance(
                start, end, match["token_start"], match["token_end"]
            )
            candidate_tuple = (
                candidate_distance,
                match["token_start"],
                match["match_id"],
            )
            if nearest is None or candidate_tuple < nearest[0]:
                nearest = (candidate_tuple, match, candidate_distance)
        if nearest is not None:
            _, nearest_trigger, distance = nearest
            trigger_match_id = nearest_trigger["match_id"]
            trigger_cache_key = trigger_result["cache_key"]
        else:
            nearest_trigger = None
            trigger_match_id = None
            trigger_cache_key = (trigger_result or {}).get("cache_key")

        boundary_ok = min_tokens <= length <= max_tokens
        neighborhood_ok = distance <= window
        eligible = boundary_ok and neighborhood_ok and nearest_trigger is not None
        issues = []
        if not boundary_ok:
            issues.append("E_ARG_FRAGMENT_BOUNDARY")
        if not neighborhood_ok:
            issues.append("E_ARG_OUTSIDE_TRIGGER_NEIGHBORHOOD")
        if nearest_trigger is None:
            issues.append("E_ARG_NO_TRIGGER")

        role_weight = int(role_weights.get(role, 600))
        raw_factor = Fraction(raw_confidence, 1000)
        role_factor = Fraction(role_weight, 1000)
        distance_factor = (
            Fraction(max(window - distance, 0), window) if window > 0 else Fraction(0)
        )
        length_factor = max(
            Fraction(0),
            Fraction(1)
            - Fraction(abs(length - ideal_tokens), max(length, ideal_tokens, 1)),
        )

        components = [
            ("role_weight", role_factor, Fraction(role_w, 1000)),
            ("trigger_distance_decay", distance_factor, Fraction(distance_w, 1000)),
            ("fragment_length_norm", length_factor, Fraction(length_w, 1000)),
            ("raw_candidate_prior", raw_factor, Fraction(raw_w, 1000)),
        ]
        score = Fraction(0)
        component_payload: dict[str, Any] = {}
        for name, factor, coefficient in components:
            component_score = factor * coefficient
            score += component_score
            component_payload[name] = {
                "factor": _fraction_payload(factor),
                "coefficient": _fraction_payload(coefficient),
                "contribution": _fraction_payload(component_score),
            }
        score = min(Fraction(1), max(Fraction(0), score))
        quantized = quantize_score(score, quant_config)
        quantized_score = Fraction(
            quantized["quantized_fixed_int"], quantized["scale"]
        )

        key_parts = argument_key_parts(
            event, argument, trigger_result, revision, config, scope
        )
        arg_content_fp = key_parts["arg_content_fingerprint"]
        rule_slice_fp = key_parts["rule_pack_fingerprint"]
        quant_fp = key_parts["quantization_fingerprint"]
        candidate = {
            "artifact_id": f"arg:{event_id}:{arg_id}",
            "event_id": event_id,
            "arg_id": arg_id,
            "role": role,
            "start": start,
            "end": end,
            "raw_confidence_permille": raw_confidence,
            "fragment_length_tokens": length,
            "distance_tokens": distance if nearest_trigger else None,
            "trigger_match_id": trigger_match_id,
            "trigger_artifact_id": f"trigger:{event_id}",
            "nearest_trigger_term": nearest_trigger["term"] if nearest_trigger else None,
            "eligible": eligible,
            "extraction_issues": issues,
            "role_weight_permille": role_weight,
            "factors": {
                "role_factor": _fraction_payload(role_factor),
                "distance_factor": _fraction_payload(distance_factor),
                "length_factor": _fraction_payload(length_factor),
                "raw_factor": _fraction_payload(raw_factor),
            },
            "components": component_payload,
            "score_numerator": score.numerator,
            "score_denominator": score.denominator,
            "quantized_score_numerator": quantized_score.numerator,
            "quantized_score_denominator": quantized_score.denominator,
            "score_fixed": quantized["quantized_value"],
            "confidence_permille": int(
                quantized_score * 1000
            ),
            "quantization": quantized,
            "argument_content_fingerprint": arg_content_fp,
            "rule_pack_fingerprint": rule_slice_fp,
            "quantization_fingerprint": quant_fp,
            "depends_on": [f"trigger:{event_id}"],
        }
        candidate["cache_key"] = key_parts["cache_key"]
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (item["role"], item["start"], item["end"], item["arg_id"])
    )
    return {
        "artifact_id": f"arguments:{event_id}",
        "event_id": event_id,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "eligible_count": sum(1 for item in candidates if item["eligible"]),
        "schema_revision_id": revision["revision_id"],
    }
