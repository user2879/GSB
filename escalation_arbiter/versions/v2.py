"""v2: JSON bundle with exact per-mille integers and float views."""

import json

from ..fixedpoint import (TAG_FLOAT_APPROX, TAG_INT_PERMILLE,
                          float_to_permille, permille_to_float)
from ..models import EvidenceScore, TicketDecision
from .common import V1_REASON_SLOTS


def _precision_tag(permille: int) -> str:
    if permille % 1000 == 0:
        return TAG_INT_PERMILLE
    return TAG_FLOAT_APPROX


def write_bundle(path: str, decisions, rules, config) -> dict:
    sidecar = {"version": "v2", "tickets": {}, "losses": []}
    payload = {
        "format": "v2",
        "decimals": config.decimals,
        "quantum": config.quantum,
        "half_life_ms": config.half_life_ms,
        "fusion": config.fusion,
        "tickets": [],
    }
    for decision in sorted(decisions, key=lambda d: d.tid):
        entry = {
            "tid": decision.tid,
            "base_ts": decision.base_ts,
            "decision_score": decision.decision_score,
            "decision_score_float": permille_to_float(decision.decision_score),
            "precision_tag": _precision_tag(decision.decision_score),
            "boundary": decision.boundary,
            "priority_level": decision.priority_level,
            "priority_label": decision.priority_label,
            "escalate": decision.escalate,
            "reason_codes": list(decision.reason_codes),
            "named_reason_codes": [
                {"code": code, "name": rules.reason_dict.get(code, code)}
                for code in decision.reason_codes
            ],
            "evidence": [],
        }
        ticket_side = {"residuals": {}, "precision": {},
                       "v1_only_losses": []}
        for esc in decision.evidence_scores:
            entry["evidence"].append({
                "seq": esc.seq, "etype": esc.etype, "role": esc.role,
                "ts": esc.ts, "confidence": esc.confidence,
                "score": esc.quantized_score,
                "score_float": permille_to_float(esc.quantized_score),
                "residual": esc.residual,
            })
            ticket_side["residuals"][str(esc.seq)] = esc.residual
            ticket_side["precision"][str(esc.seq)] = _precision_tag(
                esc.quantized_score)
        for code in decision.reason_codes:
            if code not in V1_REASON_SLOTS:
                ticket_side["v1_only_losses"].append(
                    {"field": "reason_code", "code": code})
                sidecar["losses"].append({
                    "tid": decision.tid, "field": "reason_code",
                    "original": code, "stored": None,
                })
        payload["tickets"].append(entry)
        sidecar["tickets"][decision.tid] = ticket_side
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return sidecar


def read_bundle(path: str, sidecar: dict):
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    decisions = []
    for entry in sorted(payload["tickets"], key=lambda row: row["tid"]):
        evidence = []
        for row in sorted(entry["evidence"], key=lambda r: r["seq"]):
            exact = int(row["score"])
            restored = float_to_permille(row["score_float"])
            tag = row.get("precision_tag", _precision_tag(exact))
            if tag == TAG_FLOAT_APPROX:
                exact = exact
            evidence.append(EvidenceScore(
                seq=int(row["seq"]), etype=row["etype"], role=row["role"],
                ts=int(row["ts"]), delta_ms=0, decay=0, weight=0,
                role_cred=0, confidence=int(row["confidence"]),
                factor_1=0, factor_2=0, factor_3=0, round_dir_1="",
                round_dir_2="", round_dir_3="", raw_score=exact,
                quantized_score=exact,
                residual=int(row.get("residual", 0)),
            ))
        decisions.append(TicketDecision(
            tid=entry["tid"], base_ts=int(entry["base_ts"]),
            evidence_scores=evidence,
            final_evidence_scores={e.seq: e.quantized_score
                                   for e in evidence},
            contributions=[],
            decision_score=int(entry["decision_score"]),
            boundary=bool(entry["boundary"]),
            priority_level=int(entry["priority_level"]),
            priority_label=entry["priority_label"],
            escalate=bool(entry["escalate"]),
            reason_codes=list(entry["reason_codes"]),
            stage_keys={},
        ))
    return decisions
