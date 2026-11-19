"""Feed loading, strong validation, canonical ordering, and slot grouping."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json_bytes
from .constants import FEED_DIGEST_DOMAIN


class FeedError(Exception):
    """A whole-feed rejection with the first offending field location."""

    code = "E_FEED"

    def __init__(
        self,
        message: str,
        *,
        field_path: str,
        arrival_seq: Any = None,
        entry_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field_path = field_path
        self.arrival_seq = arrival_seq
        self.entry_index = entry_index

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field_path": self.field_path,
            "arrival_seq": self.arrival_seq,
            "entry_index": self.entry_index,
        }


@dataclass(frozen=True)
class FeedEntry:
    sample_id: str
    arrival_seq: int
    batch_slot: int
    admitted: bool
    reject_reason: str | None
    source_index: int

    @property
    def field_prefix(self) -> str:
        return f"entries[{self.source_index}]"

    def normalized(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "arrival_seq": self.arrival_seq,
            "batch_slot": self.batch_slot,
            "admitted": self.admitted,
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True)
class SlotMember:
    sample_id: str
    arrival_seq: int
    rank: int
    source_index: int


@dataclass(frozen=True)
class OverwrittenEntry:
    sample_id: str
    arrival_seq: int
    batch_slot: int
    kept_arrival_seq: int
    source_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "arrival_seq": self.arrival_seq,
            "batch_slot": self.batch_slot,
            "kept_arrival_seq": self.kept_arrival_seq,
        }


@dataclass(frozen=True)
class FailedEntry:
    sample_id: str
    arrival_seq: int
    batch_slot: int
    reason: str
    source_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "arrival_seq": self.arrival_seq,
            "batch_slot": self.batch_slot,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SlotGroup:
    batch_slot: int
    members: tuple[SlotMember, ...]
    overwritten: tuple[OverwrittenEntry, ...]
    failed: tuple[FailedEntry, ...]

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member.sample_id for member in self.members)


@dataclass(frozen=True)
class FeedPlan:
    supported_slots: tuple[int, ...]
    entries_by_arrival: tuple[FeedEntry, ...]
    groups: tuple[SlotGroup, ...]
    empty_slots: tuple[int, ...]
    feed_digest: str

    @property
    def failed_entries(self) -> tuple[FailedEntry, ...]:
        return tuple(
            item
            for group in self.groups
            for item in group.failed
        )

    @property
    def overwritten_entries(self) -> tuple[OverwrittenEntry, ...]:
        return tuple(
            item
            for group in self.groups
            for item in group.overwritten
        )

    def group_by_slot(self) -> dict[int, SlotGroup]:
        return {group.batch_slot: group for group in self.groups}


def load_feed(
    path: Path,
    *,
    supported_slots: Sequence[int],
    known_sample_ids: Iterable[str],
) -> FeedPlan:
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FeedError(
            f"cannot read feed file: {exc}",
            field_path="$",
            arrival_seq=None,
        ) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FeedError(
            f"invalid JSON: {exc.msg} at line {exc.lineno}, column {exc.colno}",
            field_path="$",
            arrival_seq=None,
        ) from exc
    return parse_feed(
        value,
        supported_slots=supported_slots,
        known_sample_ids=known_sample_ids,
    )


def parse_feed(
    value: Any,
    *,
    supported_slots: Sequence[int],
    known_sample_ids: Iterable[str],
) -> FeedPlan:
    slots = tuple(sorted(int(slot) for slot in supported_slots))
    if not slots:
        raise ValueError("supported_slots must not be empty")
    if len(set(slots)) != len(slots):
        raise ValueError("supported_slots contains duplicates")
    slot_set = set(slots)
    known = {str(item) for item in known_sample_ids}

    if isinstance(value, list):
        entries_raw = value
        schema = None
    elif isinstance(value, dict):
        if "entries" not in value:
            raise FeedError(
                "top-level object must contain entries",
                field_path="$.entries",
                arrival_seq=None,
            )
        schema = value.get("schema")
        if schema is not None and not isinstance(schema, str):
            raise FeedError(
                "schema must be a string when present",
                field_path="$.schema",
                arrival_seq=None,
            )
        entries_raw = value["entries"]
    else:
        raise FeedError(
            "feed root must be an object with entries or an array",
            field_path="$",
            arrival_seq=None,
        )
    if not isinstance(entries_raw, list):
        raise FeedError(
            "entries must be an array",
            field_path="$.entries",
            arrival_seq=None,
        )

    entries: list[FeedEntry] = []
    for index, item in enumerate(entries_raw):
        prefix = f"entries[{index}]"
        if not isinstance(item, dict):
            raise FeedError(
                "entry must be an object",
                field_path=prefix,
                arrival_seq=None,
                entry_index=index,
            )
        raw_arrival = item.get("arrival_seq")
        sample_id = item.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id != sample_id.strip():
            raise FeedError(
                "sample_id must be a non-empty string without surrounding whitespace",
                field_path=f"{prefix}.sample_id",
                arrival_seq=raw_arrival,
                entry_index=index,
            )
        if len(sample_id) > 256 or any(ord(ch) < 32 for ch in sample_id):
            raise FeedError(
                "sample_id is too long or contains a control character",
                field_path=f"{prefix}.sample_id",
                arrival_seq=raw_arrival,
                entry_index=index,
            )
        if type(raw_arrival) is not int:
            raise FeedError(
                "arrival_seq must be an integer",
                field_path=f"{prefix}.arrival_seq",
                arrival_seq=raw_arrival,
                entry_index=index,
            )
        raw_slot = item.get("batch_slot")
        if type(raw_slot) is not int:
            raise FeedError(
                "batch_slot must be an integer",
                field_path=f"{prefix}.batch_slot",
                arrival_seq=raw_arrival,
                entry_index=index,
            )
        if raw_slot not in slot_set:
            raise FeedError(
                f"batch_slot {raw_slot} is not in supported slots {list(slots)}",
                field_path=f"{prefix}.batch_slot",
                arrival_seq=raw_arrival,
                entry_index=index,
            )
        admitted = item.get("admitted")
        if type(admitted) is not bool:
            raise FeedError(
                "admitted must be a boolean",
                field_path=f"{prefix}.admitted",
                arrival_seq=raw_arrival,
                entry_index=index,
            )
        reject_reason: str | None = None
        if "reject_reason" in item:
            raw_reason = item["reject_reason"]
            if raw_reason is not None and not isinstance(raw_reason, str):
                raise FeedError(
                    "reject_reason must be a string or null",
                    field_path=f"{prefix}.reject_reason",
                    arrival_seq=raw_arrival,
                    entry_index=index,
                )
            reject_reason = raw_reason
        elif "reason" in item:
            raw_reason = item["reason"]
            if raw_reason is not None and not isinstance(raw_reason, str):
                raise FeedError(
                    "reason must be a string or null",
                    field_path=f"{prefix}.reason",
                    arrival_seq=raw_arrival,
                    entry_index=index,
                )
            reject_reason = raw_reason
        if not admitted and (reject_reason is None or not reject_reason.strip()):
            reject_reason = "admission_rejected"
        if admitted:
            reject_reason = None
        entries.append(
            FeedEntry(
                sample_id=sample_id,
                arrival_seq=raw_arrival,
                batch_slot=raw_slot,
                admitted=admitted,
                reject_reason=reject_reason,
                source_index=index,
            )
        )

    _validate_arrival_uniqueness(entries)
    ordered = tuple(sorted(entries, key=lambda item: (item.arrival_seq, item.source_index)))
    _validate_known_samples(ordered, known)
    _validate_success_sample_slot_uniqueness(ordered)

    groups: list[SlotGroup] = []
    for slot in slots:
        successful = [item for item in ordered if item.admitted and item.batch_slot == slot]
        failed = tuple(
            FailedEntry(
                sample_id=item.sample_id,
                arrival_seq=item.arrival_seq,
                batch_slot=item.batch_slot,
                reason=item.reject_reason or "admission_rejected",
                source_index=item.source_index,
            )
            for item in ordered
            if (not item.admitted) and item.batch_slot == slot
        )
        by_sample: dict[str, list[FeedEntry]] = {}
        for item in successful:
            by_sample.setdefault(item.sample_id, []).append(item)
        kept: list[FeedEntry] = []
        overwritten: list[OverwrittenEntry] = []
        for sample_id in sorted(by_sample):
            candidates = sorted(
                by_sample[sample_id], key=lambda item: (item.arrival_seq, item.source_index)
            )
            winner = candidates[0]
            kept.append(winner)
            overwritten.extend(
                OverwrittenEntry(
                    sample_id=item.sample_id,
                    arrival_seq=item.arrival_seq,
                    batch_slot=slot,
                    kept_arrival_seq=winner.arrival_seq,
                    source_index=item.source_index,
                )
                for item in candidates[1:]
            )
        kept.sort(key=lambda item: (item.arrival_seq, item.sample_id, item.source_index))
        members = tuple(
            SlotMember(
                sample_id=item.sample_id,
                arrival_seq=item.arrival_seq,
                rank=rank,
                source_index=item.source_index,
            )
            for rank, item in enumerate(kept, start=1)
        )
        overwritten.sort(key=lambda item: (item.arrival_seq, item.sample_id))
        groups.append(
            SlotGroup(
                batch_slot=slot,
                members=members,
                overwritten=tuple(overwritten),
                failed=failed,
            )
        )

    empty_slots = tuple(group.batch_slot for group in groups if not group.members)
    digest = digest_feed(ordered, schema=schema)
    return FeedPlan(
        supported_slots=slots,
        entries_by_arrival=ordered,
        groups=tuple(groups),
        empty_slots=empty_slots,
        feed_digest=digest,
    )


def digest_feed(entries: Sequence[FeedEntry], *, schema: Any = None) -> str:
    normalized = [entry.normalized() for entry in sorted(entries, key=lambda item: item.arrival_seq)]
    h = hashlib.sha256()
    h.update(FEED_DIGEST_DOMAIN)
    h.update(
        canonical_json_bytes(
            {
                "schema": schema,
                "entries": normalized,
            }
        )
    )
    h.update(b"\n")
    return h.hexdigest()


def _validate_arrival_uniqueness(entries: Sequence[FeedEntry]) -> None:
    seen: dict[int, FeedEntry] = {}
    for item in sorted(entries, key=lambda entry: (entry.arrival_seq, entry.source_index)):
        if item.arrival_seq in seen:
            raise FeedError(
                f"duplicate arrival_seq {item.arrival_seq}",
                field_path=f"{item.field_prefix}.arrival_seq",
                arrival_seq=item.arrival_seq,
                entry_index=item.source_index,
            )
        seen[item.arrival_seq] = item


def _validate_known_samples(entries: Sequence[FeedEntry], known: set[str]) -> None:
    for item in entries:
        if item.admitted and item.sample_id not in known:
            raise FeedError(
                f"sample_id {item.sample_id!r} is not present in the deterministic dataset",
                field_path=f"{item.field_prefix}.sample_id",
                arrival_seq=item.arrival_seq,
                entry_index=item.source_index,
            )


def _validate_success_sample_slot_uniqueness(entries: Sequence[FeedEntry]) -> None:
    """Allow retransmission in one slot, but reject an ambiguous cross-slot sample."""

    slots_by_sample: dict[str, set[int]] = {}
    first_by_sample: dict[str, FeedEntry] = {}
    for item in entries:
        if not item.admitted:
            continue
        slots_by_sample.setdefault(item.sample_id, set()).add(item.batch_slot)
        first_by_sample.setdefault(item.sample_id, item)
    for item in entries:
        if not item.admitted:
            continue
        slots = slots_by_sample[item.sample_id]
        if len(slots) <= 1:
            continue
        primary_slot = first_by_sample[item.sample_id].batch_slot
        if item.batch_slot != primary_slot:
            raise FeedError(
                f"successful sample_id {item.sample_id!r} appears in multiple slots "
                f"({primary_slot} and {item.batch_slot}); cross-slot duplicate is ambiguous",
                field_path=f"{item.field_prefix}.batch_slot",
                arrival_seq=item.arrival_seq,
                entry_index=item.source_index,
            )
