from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes
from .models import SUPPORTED_SIGNATURE_ALGORITHMS, Certificate, FixedParams, ScriptMessage, Spec


@dataclass(frozen=True)
class Quantization:
    value: int
    saturated: bool
    remainder: int
    denominator: int


class FixedMath:
    def __init__(self, params: FixedParams) -> None:
        self.params = params
        self.max_value = (1 << params.max_bits) - 1
        self.min_value = -self.max_value

    def saturate(self, value: int) -> tuple[int, bool]:
        if value > self.max_value:
            return self.max_value, True
        if value < self.min_value:
            return self.min_value, True
        return value, False

    def floor_div(self, numerator: int, denominator: int) -> tuple[int, int]:
        if denominator <= 0:
            raise ValueError("denominator must be positive")
        value = numerator // denominator
        remainder = numerator - value * denominator
        return value, remainder

    def ratio(self, numerator: int, denominator: int) -> Quantization:
        value, remainder = self.floor_div(numerator, denominator)
        saturated_value, saturated = self.saturate(value)
        return Quantization(saturated_value, saturated, remainder, denominator)

    def time(self, timestamp_ms: int) -> Quantization:
        value = timestamp_ms - self.params.time_epoch_ms
        return self.ratio(value * self.params.scale, self.params.time_quantum_ms)

    def path_score(self, length: int) -> Quantization:
        remaining = self.params.max_path_length - (length - 1)
        numerator = remaining * self.params.path_weight * self.params.scale
        return self.ratio(numerator, self.params.max_path_length)


def certificate_digest(cert: Certificate) -> int:
    return int.from_bytes(hashlib.sha256(canonical_bytes(cert.signed_payload())).digest(), "big")


def build_certificate_base(spec: Spec, cert: Certificate) -> dict[str, Any]:
    math = FixedMath(spec.fixed)
    cert_map = spec.cert_map
    issuer = cert_map[cert.issuer]
    verification_modulus = issuer.modulus
    exponent = SUPPORTED_SIGNATURE_ALGORITHMS[cert.algorithm]
    payload_digest = certificate_digest(cert)
    signature_remainder = pow(cert.signature, exponent, verification_modulus)
    digest_remainder = payload_digest % verification_modulus
    signature_q = math.ratio(signature_remainder * spec.fixed.scale, verification_modulus)
    digest_q = math.ratio(digest_remainder * spec.fixed.scale, verification_modulus)
    residual = abs(signature_q.value - digest_q.value)
    modulus_error = 2
    if residual == 0:
        signature_state = "PASS"
    elif residual <= modulus_error:
        signature_state = "INDETERMINATE"
    else:
        signature_state = "FAIL"
    start_q = math.time(cert.not_before_ms)
    end_q = math.time(cert.not_after_ms)
    return {
        "cert_id": cert.cert_id,
        "issuer_id": cert.issuer,
        "self_issued": cert.cert_id == cert.issuer,
        "serial": cert.serial,
        "algorithm": cert.algorithm,
        "exponent": exponent,
        "payload_digest_hex": payload_digest.to_bytes(32, "big").hex(),
        "own_modulus": cert.modulus,
        "verification_modulus": verification_modulus,
        "verification_cert_id": issuer.cert_id,
        "signature": cert.signature,
        "signature_remainder": signature_remainder,
        "digest_remainder": digest_remainder,
        "signature_q": signature_q.value,
        "digest_q": digest_q.value,
        "signature_residual_q": residual,
        "signature_state": signature_state,
        "signature_saturated": signature_q.saturated or digest_q.saturated,
        "not_before_ms": cert.not_before_ms,
        "not_after_ms": cert.not_after_ms,
        "not_before_q": start_q.value,
        "not_after_q": end_q.value,
        "window_saturated": start_q.saturated or end_q.saturated,
        "modulus_error_bound": modulus_error,
    }


def _window_state(
    base: dict[str, Any],
    event_q: int,
    time_error: int,
) -> tuple[str, int, int, int]:
    start_q = base["not_before_q"]
    end_q = base["not_after_q"]
    start_margin = event_q - start_q
    end_margin = end_q - event_q
    if start_margin < 0:
        if -start_margin <= time_error:
            return "INDETERMINATE", start_margin, end_margin, min(start_margin, end_margin)
        return "FAIL", start_margin, end_margin, min(start_margin, end_margin)
    if end_margin <= 0:
        if -end_margin <= time_error:
            return "INDETERMINATE", start_margin, end_margin, min(start_margin, end_margin)
        return "FAIL", start_margin, end_margin, min(start_margin, end_margin)
    if start_margin <= time_error or end_margin <= time_error:
        return "INDETERMINATE", start_margin, end_margin, min(start_margin, end_margin)
    return "PASS", start_margin, end_margin, min(start_margin, end_margin)


def _combine_states(states: list[str]) -> str:
    if "FAIL" in states:
        return "FAIL"
    if "INDETERMINATE" in states:
        return "INDETERMINATE"
    return "PASS"


def _chain_certificate_records(
    spec: Spec,
    cert_bases: dict[str, dict[str, Any]],
    cert_ids: tuple[str, ...],
    arrival_ms: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    math = FixedMath(spec.fixed)
    event_q = math.time(arrival_ms)
    records: list[dict[str, Any]] = []
    structural_errors: list[dict[str, str]] = []
    for index, cert_id in enumerate(cert_ids):
        base = cert_bases[cert_id]
        if index + 1 < len(cert_ids):
            next_id = cert_ids[index + 1]
            if base["issuer_id"] != next_id:
                structural_errors.append(
                    {
                        "cert_id": cert_id,
                        "field": "issuer",
                        "message": f"chain expects {base['issuer_id']}, found {next_id}",
                    }
                )
        time_error = 2
        window_state, start_margin, end_margin, min_margin = _window_state(base, event_q.value, time_error)
        saturation_applied = bool(
            base["signature_saturated"]
            or base["window_saturated"]
            or event_q.saturated
        )
        state_components = [base["signature_state"], window_state]
        if saturation_applied:
            state_components.append("INDETERMINATE")
        state = _combine_states(state_components)
        cumulative = 4 * (index + 1)
        records.append(
            {
                "cert_id": cert_id,
                "issuer_id": base["issuer_id"],
                "algorithm": base["algorithm"],
                "verification_cert_id": base["verification_cert_id"],
                "verification_modulus": base["verification_modulus"],
                "signature": base["signature"],
                "signature_q": base["signature_q"],
                "digest_q": base["digest_q"],
                "signature_residual_q": base["signature_residual_q"],
                "signature_state": base["signature_state"],
                "signature_saturated": base["signature_saturated"],
                "event_q": event_q.value,
                "not_before_q": base["not_before_q"],
                "not_after_q": base["not_after_q"],
                "start_margin_q": start_margin,
                "end_margin_q": end_margin,
                "window_state": window_state,
                "window_saturated": base["window_saturated"] or event_q.saturated,
                "saturation_applied": saturation_applied,
                "min_time_margin_q": min_margin,
                "state": state,
                "step_error_bound": {
                    "modulus_quantization": base["modulus_error_bound"],
                    "time_window": time_error,
                    "path_length": 0,
                    "single_certificate_total": 4,
                    "cumulative": cumulative,
                    "saturation_applied": saturation_applied,
                },
            }
        )
    if event_q.saturated:
        structural_errors.append(
            {
                "cert_id": cert_ids[0] if cert_ids else "-",
                "field": "arrival_ms",
                "message": "event time quantized with saturation",
            }
        )
    return records, structural_errors


def verify_chain(
    spec: Spec,
    cert_bases: dict[str, dict[str, Any]],
    chain_cert_ids: tuple[str, ...],
    arrival_ms: int,
) -> dict[str, Any]:
    records, structural_errors = _chain_certificate_records(spec, cert_bases, chain_cert_ids, arrival_ms)
    math = FixedMath(spec.fixed)
    path_length = len(chain_cert_ids)
    path_quantized = math.path_score(path_length)
    anchor_match = bool(chain_cert_ids) and chain_cert_ids[-1] in spec.trust_anchors
    if path_length > spec.fixed.max_path_length:
        structural_errors.append(
            {
                "cert_id": chain_cert_ids[-1],
                "field": "path_length",
                "message": f"path length {path_length} exceeds {spec.fixed.max_path_length}",
            }
        )
    if not anchor_match:
        structural_errors.append(
            {
                "cert_id": chain_cert_ids[-1] if chain_cert_ids else "-",
                "field": "trust_anchor",
                "message": "chain does not terminate at a declared trust anchor",
            }
        )
    path_state = "PASS"
    if path_length > spec.fixed.max_path_length:
        path_state = "FAIL"
    elif path_quantized.value <= 1 or path_quantized.saturated:
        path_state = "INDETERMINATE"
    cert_states = [record["state"] for record in records]
    if structural_errors:
        chain_state = "FAIL"
    else:
        chain_state = _combine_states(cert_states + [path_state])
    pass_count = sum(1 for state in cert_states if state == "PASS")
    residual_sum_raw = sum(record["signature_residual_q"] for record in records)
    residual_sum, residual_sum_saturated = math.saturate(-residual_sum_raw)
    min_time_margin_raw = min((record["min_time_margin_q"] for record in records), default=0)
    min_time_margin, time_margin_saturated = math.saturate(min_time_margin_raw)
    score_vector = [
        pass_count,
        residual_sum,
        min_time_margin,
        path_quantized.value,
        1 if anchor_match else 0,
        -path_length,
    ]
    score_total_raw = (
        pass_count * (4 * spec.fixed.scale + 1)
        + sum(max(0, spec.fixed.scale - record["signature_residual_q"]) for record in records)
        + max(-spec.fixed.scale, min(min_time_margin, spec.fixed.scale))
        + path_quantized.value
        + (spec.fixed.scale if anchor_match else 0)
        - path_length
    )
    score_total, score_saturated = math.saturate(score_total_raw)
    modulus_contribution = 2 * len(records)
    time_contribution = 2 * len(records)
    path_contribution = 1
    total_bound = modulus_contribution + time_contribution + path_contribution
    boundary_ties: list[dict[str, Any]] = []
    boundary_names = ("not_before_q", "not_after_q")
    for left_index, left_record in enumerate(records):
        for right_record in records[left_index + 1 :]:
            for left_name in boundary_names:
                for right_name in boundary_names:
                    difference = abs(left_record[left_name] - right_record[right_name])
                    if difference <= total_bound:
                        boundary_ties.append(
                            {
                                "left_cert_id": left_record["cert_id"],
                                "left_boundary": left_name,
                                "right_cert_id": right_record["cert_id"],
                                "right_boundary": right_name,
                                "difference_q": difference,
                                "composite_error_bound": total_bound,
                                "tie_break": "certificate_id_ascending",
                            }
                        )
    if (boundary_ties or score_saturated or residual_sum_saturated or time_margin_saturated) and chain_state != "FAIL":
        chain_state = "INDETERMINATE"
    return {
        "certificate_ids": list(chain_cert_ids),
        "path_length": path_length,
        "path_score_q": path_quantized.value,
        "path_state": path_state,
        "path_saturated": path_quantized.saturated,
        "anchor_match": anchor_match,
        "structural_errors": structural_errors,
        "chain_state": chain_state,
        "score_vector": score_vector,
        "score_total_q": score_total,
        "score_saturated": score_saturated or residual_sum_saturated or time_margin_saturated,
        "certificates": records,
        "window_boundary_ties": boundary_ties,
        "error_bound": {
            "modulus_quantization": modulus_contribution,
            "time_window": time_contribution,
            "path_length": path_contribution,
            "total": total_bound,
            "per_step": [
                {
                    "step": index + 1,
                    "cert_id": record["cert_id"],
                    "comparison_bound": 4 * (index + 1),
                }
                for index, record in enumerate(records)
            ],
        },
    }


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    vector = candidate["score_vector"]
    ids = candidate["certificate_ids"]
    return tuple(-value for value in vector) + tuple(ids)


def verify_message(
    spec: Spec,
    cert_bases: dict[str, dict[str, Any]],
    message: ScriptMessage,
) -> dict[str, Any]:
    chain_ids = [message.cert_ids]
    if message.alternative_cert_ids is not None:
        chain_ids.append(message.alternative_cert_ids)
    candidates = [verify_chain(spec, cert_bases, tuple(ids), message.arrival_ms) for ids in chain_ids]
    ordered = sorted(candidates, key=_candidate_sort_key)
    selected = ordered[0]
    affected = selected["chain_state"] == "INDETERMINATE"
    boundary_tie = bool(selected["window_boundary_ties"])
    if len(ordered) == 1:
        decision = selected["chain_state"]
        selection_reason = (
            "single-chain-window-boundary-within-composite-bound"
            if boundary_tie
            else "single-chain-fixed-verification"
        )
    elif all(candidate["chain_state"] == "FAIL" for candidate in ordered):
        decision = "FAIL"
        selection_reason = "all-candidate-chains-failed"
    else:
        runner_up = ordered[1]
        pairwise_bound = selected["error_bound"]["total"] + runner_up["error_bound"]["total"]
        score_gap = selected["score_total_q"] - runner_up["score_total_q"]
        if score_gap <= pairwise_bound:
            decision = "INDETERMINATE"
            affected = True
            selection_reason = "score-difference-within-composite-error-bound"
        else:
            decision = selected["chain_state"]
            selection_reason = (
                "window-boundary-within-composite-bound"
                if boundary_tie and decision == "INDETERMINATE"
                else "lexicographic-score-vector-then-certificate-id"
            )
    if len(ordered) > 1:
        pairwise_bound = selected["error_bound"]["total"] + ordered[1]["error_bound"]["total"]
    else:
        pairwise_bound = None
    selected = dict(selected)
    selected["selection_rank"] = 1
    selected["decision_affected_by_error"] = affected and decision == "INDETERMINATE"
    return {
        "seq": message.seq,
        "sender": message.sender,
        "client_seq": message.client_seq,
        "server_seq": message.server_seq,
        "type": message.message_type,
        "arrival_ms": message.arrival_ms,
        "expected": message.expected,
        "decision": decision,
        "expected_match": decision == message.expected,
        "affected_by_error": affected,
        "selected_cert_ids": selected["certificate_ids"],
        "selection_reason": selection_reason,
        "selection_rule": "score_vector_desc_then_certificate_ids_asc",
        "pairwise_indistinguishability_bound": pairwise_bound,
        "selected_chain": selected,
        "candidates": [
            {
                "rank": index + 1,
                "certificate_ids": candidate["certificate_ids"],
                "chain_state": candidate["chain_state"],
                "score_vector": candidate["score_vector"],
                "score_total_q": candidate["score_total_q"],
                "error_bound_total": candidate["error_bound"]["total"],
            }
            for index, candidate in enumerate(ordered)
        ],
    }
