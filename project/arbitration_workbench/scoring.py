"""Ticket-local evidence scoring and deterministic freshness decay."""

from __future__ import annotations

from decimal import Decimal, localcontext

from .model import ArbitrationConfig, Evidence, Rules
from .numeric import decay_fixed, decimal_text, fingerprint, quantize_fixed


def score_ticket(ticket_id: str, evidences: tuple[Evidence, ...], rules: Rules, config: ArbitrationConfig) -> dict:
    if not evidences:
        raise ValueError(f"ticket {ticket_id} has no evidence")
    scale = config.scale
    ordered = tuple(sorted(evidences, key=lambda item: item.evidence_seq))
    base_timestamp = min(item.timestamp for item in ordered)
    entries: list[dict] = []
    raw_total = 0

    for evidence in ordered:
        weight = rules.weights[evidence.evidence_type]
        trust = rules.trusts[evidence.source_role]
        decay_units = decay_fixed(
            evidence.timestamp - base_timestamp,
            config.decay_half_life,
            scale,
            config.quantization_units,
        )
        decay_value = Decimal(decay_units) / Decimal(scale)
        with localcontext() as ctx:
            ctx.prec = 90
            score_value = weight * trust * decay_value
            score_units = quantize_fixed(score_value, scale, config.quantization_units)
        raw_total += score_units
        entries.append(
            {
                "evidence_seq": evidence.evidence_seq,
                "evidence_type": evidence.evidence_type,
                "source_role": evidence.source_role,
                "timestamp": evidence.timestamp,
                "decay_fixed": decay_units,
                "score_fixed": score_units,
                "factors": {
                    "evidence_type_weight": decimal_text(weight),
                    "source_role_trust": decimal_text(trust),
                    "decay_coefficient": decimal_text(decay_value),
                    "operation_order": "type_weight*role_trust*decay",
                },
            }
        )

    ranked = sorted(entries, key=lambda item: (-item["score_fixed"], -item["decay_fixed"], item["evidence_seq"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    entries.sort(key=lambda item: item["evidence_seq"])
    payload = {
        "ticket_id": ticket_id,
        "base_timestamp": base_timestamp,
        "scale": scale,
        "quantization_units": config.quantization_units,
        "half_life": config.decay_half_life,
        "rank_policy": "score_fixed_desc,decay_fixed_desc,evidence_seq_asc",
        "evidence_scores": entries,
        "raw_total_fixed": raw_total,
    }
    payload["score_fingerprint"] = fingerprint(payload)
    return payload

