"""Revision-chain composition into an effective trigger view."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from .errors import Codes, IssueCollector
from .fingerprints import digest, trigger_entry_fingerprint
from .models import Revision, RuleView, TriggerEntry


def _lineage_for_tip(
    tip: str,
    revisions: Mapping[str, Revision],
    issues: IssueCollector,
) -> list[Revision]:
    path: list[Revision] = []
    seen: set[str] = set()
    current: str | None = tip
    while current is not None:
        if current in seen:
            issues.add(
                Codes.REVISION_CYCLE,
                "cycle encountered while composing effective view; tip skipped",
                scope="rules",
                entity_id=tip,
                retained=False,
            )
            return []
        revision = revisions.get(current)
        if revision is None:
            issues.add(
                Codes.PARENT_MISSING,
                "parent revision missing during composition; branch truncated",
                scope="rules",
                entity_id=current,
                retained=False,
            )
            break
        seen.add(current)
        path.append(revision)
        current = revision.parent_id
    path.reverse()
    return path


def compose_effective_rules(
    revisions: Iterable[Revision],
    active_revision_ids: Iterable[str],
    issues: IssueCollector | None = None,
) -> RuleView:
    issues = issues or IssueCollector()
    revision_map = {revision.revision_id: revision for revision in revisions}
    active = tuple(dict.fromkeys(active_revision_ids))
    if not active:
        issues.add(
            Codes.ACTIVE_REVISION_INVALID,
            "no active revisions supplied",
            scope="rules",
            retained=False,
        )
        return RuleView((), (), (), digest({"active": [], "entries": []}))

    path_revisions: dict[str, Revision] = {}
    branch_rank: dict[str, int] = {}
    depth_by_id: dict[str, int] = {}
    for rank, tip in enumerate(active):
        if tip not in revision_map:
            issues.add(
                Codes.ACTIVE_REVISION_MISSING,
                "active revision is missing; tip skipped",
                scope="rules",
                entity_id=tip,
                retained=False,
            )
            continue
        lineage = _lineage_for_tip(tip, revision_map, issues)
        for depth, revision in enumerate(lineage):
            path_revisions.setdefault(revision.revision_id, revision)
            branch_rank.setdefault(revision.revision_id, rank)
            depth_by_id.setdefault(revision.revision_id, depth)
    if not path_revisions:
        return RuleView((), active, (), digest({"active": list(active), "entries": []}))

    ordered_lineage = sorted(
        path_revisions.values(),
        key=lambda revision: (
            depth_by_id[revision.revision_id],
            branch_rank[revision.revision_id],
            revision.seq,
            revision.revision_id,
        ),
    )
    entries: dict[str, TriggerEntry] = {}
    next_order = 0
    for revision in ordered_lineage:
        for op in revision.ops:
            existing = entries.get(op.trigger_id)
            if op.op == "add":
                if existing is None:
                    entry = TriggerEntry(
                        trigger_id=op.trigger_id,
                        terms=op.terms or (),
                        intent_id=op.intent_id or "",
                        weight=op.weight or 0,
                        slot_pattern=op.slot_pattern or (),
                        source_revision=revision.revision_id,
                        entry_order=next_order,
                        provenance=(
                            {
                                "revision_id": revision.revision_id,
                                "op": "add",
                                "previous_fingerprint": None,
                            },
                        ),
                    )
                    next_order += 1
                else:
                    entry = replace(
                        existing,
                        terms=op.terms if op.terms is not None else existing.terms,
                        intent_id=op.intent_id if op.intent_id is not None else existing.intent_id,
                        weight=op.weight if op.weight is not None else existing.weight,
                        slot_pattern=op.slot_pattern
                        if op.slot_pattern is not None
                        else existing.slot_pattern,
                        source_revision=revision.revision_id,
                        provenance=existing.provenance
                        + (
                            {
                                "revision_id": revision.revision_id,
                                "op": "add-override",
                                "previous_fingerprint": trigger_entry_fingerprint(existing),
                            },
                        ),
                    )
                entries[op.trigger_id] = entry
            elif op.op == "update":
                if existing is None:
                    issues.add(
                        Codes.REVISION_OP_INVALID,
                        "update target does not exist in the parent view; op skipped",
                        scope="rules",
                        entity_id=f"{revision.revision_id}:{op.trigger_id}",
                    )
                    continue
                entry = replace(
                    existing,
                    terms=op.terms if op.terms is not None else existing.terms,
                    intent_id=op.intent_id if op.intent_id is not None else existing.intent_id,
                    weight=op.weight if op.weight is not None else existing.weight,
                    slot_pattern=op.slot_pattern
                    if op.slot_pattern is not None
                    else existing.slot_pattern,
                    source_revision=revision.revision_id,
                    provenance=existing.provenance
                    + (
                        {
                            "revision_id": revision.revision_id,
                            "op": "update",
                            "previous_fingerprint": trigger_entry_fingerprint(existing),
                        },
                    ),
                )
                entries[op.trigger_id] = entry
            elif op.op == "delete":
                if existing is None:
                    issues.add(
                        Codes.REVISION_OP_INVALID,
                        "delete target does not exist in the parent view; op skipped",
                        scope="rules",
                        entity_id=f"{revision.revision_id}:{op.trigger_id}",
                    )
                    continue
                del entries[op.trigger_id]

    effective = tuple(sorted(entries.values(), key=lambda entry: (entry.entry_order, entry.trigger_id)))
    chain_payload = {
        "active": list(active),
        "lineage": [
            {
                "revision_id": revision.revision_id,
                "parent_id": revision.parent_id,
                "seq": revision.seq,
                "ops": [op.to_dict() for op in revision.ops],
            }
            for revision in ordered_lineage
        ],
    }
    return RuleView(
        entries=effective,
        active_revision_ids=active,
        lineage_revision_ids=tuple(revision.revision_id for revision in ordered_lineage),
        chain_version=digest(chain_payload),
    )


def materialize_revision_state(
    revisions: Iterable[Revision],
    revision_id: str,
    issues: IssueCollector | None = None,
) -> RuleView:
    """Compose the exact parent lineage ending at one revision."""

    return compose_effective_rules(revisions, (revision_id,), issues)

