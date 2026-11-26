"""Revision-chain synthesis and deterministic longest-span matching."""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import FailureCollector, E_BAD_WEIGHT
from .fingerprints import entry_fingerprint
from .models import EffectiveEntry, RevisionChain, TextItem, TriggerOp


@dataclass(frozen=True)
class MatchCandidate:
    """One raw trigger occurrence inside a text."""

    entry_id: str
    intent: str
    weight: int
    start: int
    end: int  # exclusive token index
    term: tuple[str, ...]

    @property
    def span(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SuppressedCandidate:
    """A candidate swallowed by a higher-priority overlapping trigger."""

    entry_id: str
    start: int
    end: int
    swallowed_by: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "start": self.start,
            "end": self.end,
            "swallowed_by": self.swallowed_by,
            "reason": self.reason,
        }


@dataclass
class MatchResult:
    """The deterministic match outcome for one text."""

    text_id: str
    accepted: list[MatchCandidate] = field(default_factory=list)
    suppressed: list[SuppressedCandidate] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text_id": self.text_id,
            "accepted": [
                {
                    "entry_id": c.entry_id,
                    "intent": c.intent,
                    "weight": c.weight,
                    "start": c.start,
                    "end": c.end,
                    "term": list(c.term),
                }
                for c in self.accepted
            ],
            "suppressed": [s.to_dict() for s in self.suppressed],
        }


def _validate_weight(op: TriggerOp, failures: FailureCollector) -> bool:
    if not isinstance(op.weight, int) or isinstance(op.weight, bool):
        failures.add(
            E_BAD_WEIGHT,
            "weight must be an integer",
            "trigger_op",
            op.entry_id,
            weight=repr(op.weight),
        )
        return False
    if op.weight < 0:
        failures.add(
            E_BAD_WEIGHT,
            "weight must be non-negative",
            "trigger_op",
            op.entry_id,
            weight=op.weight,
        )
        return False
    return True


def synthesize_library(
    chain: RevisionChain,
    failures: FailureCollector,
) -> dict[str, EffectiveEntry]:
    """Walk revisions root->head and synthesize the effective trigger library.

    Later upserts override term/weight/intent and record provenance; deletes
    remove an entry.  Invalid ops are skipped and recorded, never aborting the
    rest of the chain.
    """
    live: dict[str, EffectiveEntry] = {}
    for revision in chain.ordered():
        for op in revision.ops:
            if op.op == "delete":
                live.pop(op.entry_id, None)
                continue
            if op.op != "upsert":
                failures.add(
                    "E_BAD_COMMAND",
                    f"unknown trigger op {op.op!r}",
                    "trigger_op",
                    op.entry_id,
                )
                continue
            if not _validate_weight(op, failures):
                continue
            existing = live.get(op.entry_id)
            overrides = (
                existing.overrides + (revision.revision_id,)
                if existing is not None
                else ()
            )
            live[op.entry_id] = EffectiveEntry(
                entry_id=op.entry_id,
                term=tuple(op.term),
                weight=op.weight,
                intent=op.intent,
                born_revision=(
                    existing.born_revision if existing is not None
                    else revision.revision_id
                ),
                last_revision=revision.revision_id,
                overrides=overrides,
            )
    return dict(sorted(live.items(), key=lambda kv: kv[0]))


def library_entry_fingerprints(library: dict[str, EffectiveEntry]) -> dict[str, str]:
    return {
        entry_id: entry_fingerprint(
            entry_id, entry.term, entry.weight, entry.intent
        )
        for entry_id, entry in library.items()
    }


def _enumerate_candidates(
    text: TextItem, library: dict[str, EffectiveEntry]
) -> list[MatchCandidate]:
    tokens = text.tokens
    candidates: list[MatchCandidate] = []
    # Build term -> entries lookup; iterate entries in deterministic entry order.
    for entry_id in sorted(library):
        entry = library[entry_id]
        term = entry.term
        if not term:
            continue
        n = len(term)
        for start in range(0, len(tokens) - n + 1):
            if tuple(tokens[start:start + n]) == term:
                candidates.append(
                    MatchCandidate(
                        entry_id=entry.entry_id,
                        intent=entry.intent,
                        weight=entry.weight,
                        start=start,
                        end=start + n,
                        term=term,
                    )
                )
    return candidates


def match_text(
    text: TextItem, library: dict[str, EffectiveEntry]
) -> MatchResult:
    """Longest-span-first deterministic overlap resolution.

    Priority: longer span wins; ties broken by term lexicographic order, then
    by ascending start position, then by entry_id for full determinism.  Any
    candidate overlapping an already accepted one is suppressed and records
    the swallowing trigger.
    """
    result = MatchResult(text_id=text.text_id)
    candidates = _enumerate_candidates(text, library)
    ordered = sorted(
        candidates,
        key=lambda c: (-c.span, c.term, c.start, c.entry_id),
    )
    occupied: list[tuple[int, int, str]] = []
    accepted: list[MatchCandidate] = []
    for candidate in ordered:
        swallower = next(
            (
                owner
                for (s, e, owner) in occupied
                if candidate.start < e and s < candidate.end
            ),
            None,
        )
        if swallower is not None:
            result.suppressed.append(
                SuppressedCandidate(
                    entry_id=candidate.entry_id,
                    start=candidate.start,
                    end=candidate.end,
                    swallowed_by=swallower,
                    reason="overlap_longer_or_lex_priority",
                )
            )
            continue
        occupied.append((candidate.start, candidate.end, candidate.entry_id))
        accepted.append(candidate)

    # Emit accepted matches in reading order, then start, then entry id.
    accepted.sort(key=lambda c: (c.start, c.end, c.entry_id))
    # Suppressed list keeps resolution (priority) order for stable audit.
    result.accepted = accepted
    return result
