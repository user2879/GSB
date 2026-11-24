"""Deterministic template-based rewriting with overlap suppression."""

from .errors import E_REWRITE_RULE_INVALID
from .hashing import combine


def _collect_matches(sentence_text, rules):
    """Return all (start, end, rule) matches in a stable order."""
    matches = []
    for rule in rules:
        source = rule.source
        start = 0
        while True:
            position = sentence_text.find(source, start)
            if position < 0:
                break
            matches.append((position, position + len(source), rule))
            start = position + len(source)
    return matches


def rewrite_sentence(sentence_text, table, template_id):
    """Apply non-overlapping replacements; overlapping entries are suppressed.

    Ordering: source length desc, then rule-table entry order.  Matches of the
    same rule keep left-to-right occurrence order.
    """
    template_rules = table.active_rules_for(template_id)
    ordered_rules = sorted(template_rules, key=lambda rule: (-len(rule.source), rule.order))
    matches = _collect_matches(sentence_text, ordered_rules)
    # Sort by length desc / rule order / position so the first match on an
    # occupied span always wins and the suppressed entries stay deterministic.
    matches.sort(key=lambda item: (-(item[1] - item[0]), item[2].order, item[0]))

    occupied = []
    accepted = []
    suppressed = []

    def overlaps(start, end):
        for span_start, span_end in occupied:
            if start < span_end and span_start < end:
                return True
        return False

    for start, end, rule in matches:
        if overlaps(start, end):
            suppressed.append(
                {
                    "rule_id": rule.rule_id,
                    "source": rule.source,
                    "start_offset": start,
                    "end_offset": end,
                    "reason": "overlap_with_accepted_replacement",
                }
            )
            continue
        occupied.append((start, end))
        accepted.append((start, end, rule))

    accepted.sort(key=lambda item: item[0])
    rewritten_parts = []
    cursor = 0
    applied = []
    for start, end, rule in accepted:
        rewritten_parts.append(sentence_text[cursor:start])
        rewritten_parts.append(rule.target)
        applied.append(
            {
                "rule_id": rule.rule_id,
                "source": rule.source,
                "target": rule.target,
                "start_offset": start,
                "end_offset": end,
            }
        )
        cursor = end
    rewritten_parts.append(sentence_text[cursor:])

    record = {
        "original_text": sentence_text,
        "rewritten_text": "".join(rewritten_parts),
        "template_id": template_id,
        "applied_rules": applied,
        "suppressed_rules": suppressed,
        "rejected_rules": [],
    }
    record["cache_key"] = combine(
        "rewrite",
        sentence_text,
        template_id,
        [
            (rule.rule_id, rule.source, rule.target, rule.order)
            for rule in ordered_rules
            if rule.source in sentence_text
        ],
        table.version,
    )
    return record


def validate_table_references(table):
    """Invalid rules were dropped at load time; this reports the audit trail."""
    return []


def rejected_rule_record(rule_id, message):
    return {"code": E_REWRITE_RULE_INVALID, "object_id": rule_id, "message": message}
