from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import E_TEMPLATE_MISSING
from .models import Sentence, SynonymRule, SynonymTable


@dataclass(frozen=True)
class ReplacementCandidate:
    rule: SynonymRule
    start: int
    end: int

    def priority_key(self) -> tuple[int, int, int, int]:
        return (-(self.end - self.start), self.rule.order, self.start, self.end)


def _find_occurrences(text: str, rule: SynonymRule) -> list[ReplacementCandidate]:
    candidates: list[ReplacementCandidate] = []
    cursor = 0
    step = max(1, len(rule.source))
    while True:
        found = text.find(rule.source, cursor)
        if found < 0:
            break
        candidates.append(
            ReplacementCandidate(rule=rule, start=found, end=found + len(rule.source))
        )
        cursor = found + step
    return candidates


def _overlaps(left: ReplacementCandidate, right: ReplacementCandidate) -> bool:
    return left.start < right.end and right.start < left.end


def rewrite_sentence(
    sentence: Sentence,
    table: SynonymTable,
    active_template_id: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    declared_templates = set(table.templates)
    active_rules = [
        rule
        for rule in table.rules
        if rule.template_id == active_template_id and rule.template_id in declared_templates
    ]
    missing_rules = [
        rule for rule in table.rules if rule.template_id not in declared_templates
    ]

    reserved_candidates: list[ReplacementCandidate] = []
    for rule in missing_rules:
        reserved_candidates.extend(_find_occurrences(sentence.text, rule))

    candidates: list[ReplacementCandidate] = []
    for rule in active_rules:
        candidates.extend(_find_occurrences(sentence.text, rule))
    candidates.sort(key=lambda candidate: candidate.priority_key())

    accepted: list[ReplacementCandidate] = []
    suppressed: list[dict[str, Any]] = []
    for candidate in candidates:
        reserved_by = [
            existing for existing in reserved_candidates if _overlaps(existing, candidate)
        ]
        if reserved_by:
            suppressed.append(
                {
                    "sentence_index": sentence.sentence_index,
                    "rule_id": candidate.rule.rule_id,
                    "template_id": candidate.rule.template_id,
                    "source": candidate.rule.source,
                    "target": candidate.rule.target,
                    "start": candidate.start,
                    "end": candidate.end,
                    "matched_text": sentence.text[candidate.start : candidate.end],
                    "status": "suppressed",
                    "reason": "missing_template_reserved_span",
                    "suppressed_by": [item.rule.rule_id for item in reserved_by],
                }
            )
            continue
        overlapping = [existing for existing in accepted if _overlaps(existing, candidate)]
        if overlapping:
            suppressed.append(
                {
                    "sentence_index": sentence.sentence_index,
                    "rule_id": candidate.rule.rule_id,
                    "template_id": candidate.rule.template_id,
                    "source": candidate.rule.source,
                    "target": candidate.rule.target,
                    "start": candidate.start,
                    "end": candidate.end,
                    "matched_text": sentence.text[candidate.start : candidate.end],
                    "status": "suppressed",
                    "reason": "overlap",
                    "suppressed_by": [item.rule.rule_id for item in overlapping],
                }
            )
            continue
        accepted.append(candidate)

    accepted.sort(key=lambda item: (item.start, item.end, item.rule.order))
    replacement_records: list[dict[str, Any]] = []
    pieces: list[str] = []
    cursor = 0
    for candidate in accepted:
        if cursor < candidate.start:
            pieces.append(sentence.text[cursor : candidate.start])
        pieces.append(candidate.rule.target)
        cursor = candidate.end
        replacement_records.append(
            {
                "sentence_index": sentence.sentence_index,
                "rule_id": candidate.rule.rule_id,
                "template_id": candidate.rule.template_id,
                "source": candidate.rule.source,
                "target": candidate.rule.target,
                "start": candidate.start,
                "end": candidate.end,
                "matched_text": sentence.text[candidate.start : candidate.end],
                "status": "applied",
                "rule_order": candidate.rule.order,
            }
        )
    pieces.append(sentence.text[cursor:])
    rewritten = "".join(pieces)

    errors: list[dict[str, Any]] = []
    for rule in missing_rules:
        matches = _find_occurrences(sentence.text, rule)
        if not matches:
            continue
        found = matches[0].start
        end = matches[0].end
        errors.append(
            {
                "code": E_TEMPLATE_MISSING,
                "stage": "rewrite",
                "sentence_index": sentence.sentence_index,
                "rule_id": rule.rule_id,
                "template_id": rule.template_id,
                "declared_templates": sorted(declared_templates),
                "source": rule.source,
                "start": found,
                "end": end,
                "preserved_fragment": sentence.text[found:end],
                "preserved_occurrence_count": len(matches),
                "decision": "rule skipped; original fragment preserved",
            }
        )
    return rewritten, replacement_records, suppressed, errors
