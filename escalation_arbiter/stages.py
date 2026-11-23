"""Per-ticket fixed-point compute stages: decay, relations, arbitration."""

from .fixedpoint import (clamp_low, decay_coeff, mul_permille, quantize,
                         scale_factor)
from .models import EvidenceScore, RelationContribution, TicketDecision


def _round_dir(before: int, after: int) -> str:
    if after == before:
        return "exact"
    return "up" if after > before else "down"


def run_decay_stage(ticket, rules, config):
    """Stage 1: half-life decay and fixed-point evidence scoring."""
    base_ts = ticket.base_ts()
    scale = 10 ** config.decimals
    out = []
    for ev in ticket.sorted_evidence():
        delta = ev.ts - base_ts
        coeff = decay_coeff(delta, config.half_life_ms)
        weight = scale_factor(rules.type_weights[ev.etype], config.decimals)
        role = scale_factor(rules.role_cred[ev.role], config.decimals)
        conf = scale_factor(ev.confidence, config.decimals)

        factor_1 = mul_permille(weight, role)
        factor_2 = mul_permille(factor_1, conf)
        factor_3 = mul_permille(factor_2, coeff)

        if config.quantum and config.quantum > 0:
            quantized, residual = quantize(factor_3, config.quantum)
        else:
            quantized, residual = factor_3, 0

        out.append(EvidenceScore(
            seq=ev.seq, etype=ev.etype, role=ev.role, ts=ev.ts,
            delta_ms=delta, decay=coeff,
            weight=rules.type_weights[ev.etype],
            role_cred=rules.role_cred[ev.role],
            confidence=ev.confidence,
            factor_1=factor_1, factor_2=factor_2, factor_3=factor_3,
            round_dir_1=_round_dir(weight, factor_1),
            round_dir_2=_round_dir(factor_1, factor_2),
            round_dir_3=_round_dir(factor_2, factor_3),
            raw_score=factor_3, quantized_score=quantized, residual=residual,
        ))
    return base_ts, out


def _pairs_for(entry, evidence_scores):
    left_seqs = [e.seq for e in evidence_scores if e.etype == entry.left]
    right_seqs = [e.seq for e in evidence_scores if e.etype == entry.right]
    return [(a, b) for a in sorted(left_seqs) for b in sorted(right_seqs)]


def run_relation_stage(evidence_scores, rules, config):
    """Stage 2: mutex deductions and support bonuses in fixed point."""
    current = {e.seq: e.quantized_score for e in evidence_scores}
    contributions = []
    reason_codes = {"mutex": [], "support": []}

    def apply(entries, strength_permille, kind):
        for entry in entries:
            code = f"{'MX' if kind == 'mutex' else 'SP'}{entry.rid}"
            hit = False
            for left_seq, right_seq in _pairs_for(entry, evidence_scores):
                lb, rb = current[left_seq], current[right_seq]
                if kind == "mutex":
                    la = clamp_low(mul_permille(lb, strength_permille),
                                   config.zero_floor)
                    ra = clamp_low(mul_permille(rb, strength_permille),
                                   config.zero_floor)
                else:
                    la = mul_permille(lb, strength_permille)
                    ra = mul_permille(rb, strength_permille)
                current[left_seq], current[right_seq] = la, ra
                hit = True
                contributions.append(RelationContribution(
                    rid=entry.rid, kind=kind, left_seq=left_seq,
                    right_seq=right_seq, reason_code=code,
                    left_before=lb, right_before=rb,
                    left_delta=la - lb, right_delta=ra - rb,
                    left_after=la, right_after=ra,
                ))
            if hit:
                reason_codes[kind].append(code)

    apply(rules.mutex, config.deduct_permille, "mutex")
    apply(rules.support, config.bonus_permille, "support")
    ordered = (sorted(set(reason_codes["mutex"])) +
               sorted(set(reason_codes["support"])))
    return current, contributions, ordered


def _select_tier(score, tiers):
    ordered = sorted(tiers, key=lambda t: t.limit)
    chosen = None
    boundary = False
    for tier in ordered:
        if score >= tier.limit:
            chosen = tier
            if score == tier.limit:
                boundary = True
    return chosen, boundary


def run_arbitration_stage(tid, base_ts, evidence_scores, final_scores,
                          contributions, reason_codes, rules, config):
    """Stage 3: aggregate decision score, priority and reason set."""
    ordered_seq = [e.seq for e in evidence_scores]
    values = [final_scores[seq] for seq in ordered_seq]
    if config.fusion == "sum":
        decision = sum(values)
    else:
        decision = (sum(values) // len(values)) if values else 0

    tier, boundary = _select_tier(decision, rules.tiers)
    if tier is None:
        level, label, escalate = 0, "NO_ESCALATION", False
        codes = []
    else:
        level, label, escalate = tier.level, tier.label, True
        valid = [c for c in reason_codes if c in rules.reason_dict]
        codes = sorted(set(valid),
                       key=lambda c: (0 if c.startswith("MX") else 1,
                                      _rid_key(c)))
    return TicketDecision(
        tid=tid, base_ts=base_ts, evidence_scores=evidence_scores,
        final_evidence_scores={seq: final_scores[seq] for seq in ordered_seq},
        contributions=contributions, decision_score=decision,
        boundary=boundary, priority_level=level, priority_label=label,
        escalate=escalate, reason_codes=codes, stage_keys={},
    )


def _rid_key(code):
    prefix = code[:2]
    try:
        return int(code[2:])
    except ValueError:
        return code[2:]


def compute_ticket(ticket, rules, config):
    """Run the three compute stages for one ticket in fixed order."""
    base_ts, scored = run_decay_stage(ticket, rules, config)
    final_scores, contributions, reason_codes = run_relation_stage(
        scored, rules, config)
    return run_arbitration_stage(
        ticket.tid, base_ts, scored, final_scores, contributions,
        reason_codes, rules, config)
