"""v1: fixed-width text rows (lowest expressiveness).

Layout per record (the leading record marker is part of the fixed widths):

  H version|decimals|quantum|half_life_ms|fusion|tier_count
  E tid(12) seq(4) etype(8) role(4) ts(13) confidence(7) score(12) residual(10)
  D tid(12) decision(12) boundary(6) priority(3) label(14) reason_codes(42)

v1 only keeps six reason-code slots, 3-letter role aliases and truncates
display floats to 3 decimals; every loss is mirrored into the sidecar so
the logical (integer per-mille) decision remains restorable.
"""

from .common import (V1_REASON_SLOTS, alias_role,
                     unalias_role)

E_WIDTHS = (12, 4, 8, 4, 13, 7, 12, 10)
D_WIDTHS = (12, 12, 6, 3, 14, 42)
HEADER = "H|v1"


def _pad(text: str, width: int, align: str = "left") -> str:
    text = str(text)
    if len(text) > width:
        text = text[:width]
    if align == "right":
        return text.rjust(width)
    return text.ljust(width)


def _line(prefix, fields, widths):
    cells = [prefix]
    for value, width, align in fields:
        cells.append(_pad(value, width, align))
    line = " ".join(cells)
    return line


def write_bundle(path: str, decisions, rules, config) -> dict:
    sidecar = {"version": "v1", "tickets": {}, "losses": []}
    lines = [
        HEADER,
        "META|{}|{}|{}|{}|{}".format(
            config.decimals, config.quantum, config.half_life_ms,
            config.fusion, len(rules.tiers)),
    ]
    for decision in sorted(decisions, key=lambda d: d.tid):
        tid = decision.tid
        ticket_side = {"residuals": {}, "full_reason_codes":
                       list(decision.reason_codes),
                       "role_aliases": {}, "display_float": {}}
        for esc in decision.evidence_scores:
            fields = [
                (tid, E_WIDTHS[0], "left"),
                (esc.seq, E_WIDTHS[1], "right"),
                (esc.etype, E_WIDTHS[2], "left"),
                (alias_role(esc.role), E_WIDTHS[3], "left"),
                (esc.ts, E_WIDTHS[4], "right"),
                (esc.confidence, E_WIDTHS[5], "right"),
                (esc.quantized_score, E_WIDTHS[6], "right"),
                (esc.residual, E_WIDTHS[7], "right"),
            ]
            lines.append(_line("E", fields, E_WIDTHS))
            ticket_side["residuals"][str(esc.seq)] = esc.residual
            ticket_side["role_aliases"][str(esc.seq)] = esc.role
            if alias_role(esc.role) == "UNK":
                sidecar["losses"].append({
                    "tid": tid, "field": "role", "seq": esc.seq,
                    "original": esc.role, "stored": "UNK",
                })
        slots = [c if c in V1_REASON_SLOTS else "" for c in V1_REASON_SLOTS]
        for code in decision.reason_codes:
            if code not in V1_REASON_SLOTS:
                sidecar["losses"].append({
                    "tid": tid, "field": "reason_code",
                    "original": code, "stored": "",
                })
        fields = [
            (tid, D_WIDTHS[0], "left"),
            (decision.decision_score, D_WIDTHS[1], "right"),
            (str(decision.boundary)[0].upper(), D_WIDTHS[2], "left"),
            (decision.priority_level, D_WIDTHS[3], "right"),
            (decision.priority_label, D_WIDTHS[4], "left"),
            (",".join(slots), D_WIDTHS[5], "left"),
        ]
        lines.append(_line("D", fields, D_WIDTHS))
        if len(decision.reason_codes) > len(V1_REASON_SLOTS):
            sidecar["losses"].append({
                "tid": tid, "field": "reason_slot_count",
                "original": len(decision.reason_codes),
                "stored": len(V1_REASON_SLOTS),
            })
        sidecar["tickets"][tid] = ticket_side
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return sidecar


def read_bundle(path: str, sidecar: dict):
    from ..models import EvidenceScore, TicketDecision
    evidence = {}
    decisions = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line[0] in ("H", "M"):
                continue
            marker = line[0]
            body = line[2:]
            if marker == "E":
                widths = E_WIDTHS
                cuts = []
                pos = 0
                for width in widths:
                    cuts.append((pos, pos + width))
                    pos += width + 1
                cells = [body[a:b].strip() for a, b in cuts]
                tid, seq, etype, role, ts, conf, score, residual = cells
                seq = int(seq)
                full_role = sidecar["tickets"][tid]["role_aliases"].get(
                    str(seq), unalias_role(role))
                evidence.setdefault(tid, []).append(EvidenceScore(
                    seq=seq, etype=etype, role=full_role, ts=int(ts),
                    delta_ms=0, decay=0, weight=0, role_cred=0,
                    confidence=int(conf), factor_1=0, factor_2=0,
                    factor_3=0, round_dir_1="", round_dir_2="",
                    round_dir_3="", raw_score=int(score),
                    quantized_score=int(score), residual=int(residual),
                ))
            elif marker == "D":
                cuts = []
                pos = 0
                for width in D_WIDTHS:
                    cuts.append((pos, pos + width))
                    pos += width + 1
                cells = [body[a:b].strip() for a, b in cuts]
                tid, decision, boundary, level, label, reason_blob = cells
                decisions[tid] = TicketDecision(
                    tid=tid, base_ts=0,
                    evidence_scores=sorted(evidence.get(tid, []),
                                           key=lambda e: e.seq),
                    final_evidence_scores={
                        e.seq: e.quantized_score
                        for e in evidence.get(tid, [])},
                    contributions=[], decision_score=int(decision),
                    boundary=(boundary == "T"), priority_level=int(level),
                    priority_label=label, escalate=int(level) > 0,
                    reason_codes=list(
                        sidecar["tickets"][tid]["full_reason_codes"]),
                    stage_keys={},
                )
    return [decisions[tid] for tid in sorted(decisions)]
