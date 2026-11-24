"""Bidirectional receipt conversion with an explicit downgrade sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

from .errors import E_CONVERSION_TARGET, Failure
from .jsonutil import MISSING, decode_missing, is_missing, json_ready, pretty_dumps
from .models import ALL_RECEIPT_FIELDS, R1, R2, R3, VERSION_FIELDS, VERSIONS, Receipt


VERSION_ORDER = {R1: 1, R2: 2, R3: 3}


def sidecar_key(receipt: Receipt) -> str:
    return f"{receipt.batch_id}|{receipt.request_id}|{receipt.output_digest}"


@dataclass
class SidecarStore:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def capture(self, receipt: Receipt, fields: tuple[str, ...]) -> None:
        key = sidecar_key(receipt)
        entry = self.entries.setdefault(
            key,
            {
                "batch_id": receipt.batch_id,
                "request_id": receipt.request_id,
                "output_digest": receipt.output_digest,
                "origin_uid": receipt.uid,
                "restored_fields": {},
            },
        )
        for name in fields:
            entry["restored_fields"][name] = json_ready(getattr(receipt, name))

    def restore(self, receipt: Receipt, field_name: str) -> tuple[Any, bool]:
        entry = self.entries.get(sidecar_key(receipt))
        if entry is None:
            return MISSING, False
        values = entry.get("restored_fields", {})
        if field_name not in values:
            return MISSING, False
        return decode_missing(values[field_name]), True

    def prune(self, receipt: Receipt, field_names: tuple[str, ...]) -> tuple[str, ...]:
        entry = self.entries.get(sidecar_key(receipt))
        if entry is None:
            return field_names
        values = entry.get("restored_fields", {})
        pruned = []
        for name in field_names:
            if name in values:
                del values[name]
                pruned.append(name)
        return tuple(pruned)

    def as_dict(self) -> dict[str, Any]:
        return {"entries": {key: json_ready(self.entries[key]) for key in sorted(self.entries)}}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SidecarStore":
        entries_raw = raw.get("entries", {})
        if not isinstance(entries_raw, dict):
            return cls()
        return cls(entries={str(key): dict(value) for key, value in entries_raw.items() if isinstance(value, dict)})


@dataclass(frozen=True)
class ConversionEvent:
    sequence: int
    receipt_uid: str
    sidecar_key: str
    operation: str
    from_version: str
    to_version: str
    source_version: str
    downgraded_fields: tuple[str, ...] = ()
    restored_fields: tuple[str, ...] = ()
    irreversible_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    completeness: str = "complete"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "receipt_uid": self.receipt_uid,
            "sidecar_key": self.sidecar_key,
            "operation": self.operation,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "source_version": self.source_version,
            "downgraded_fields": list(self.downgraded_fields),
            "restored_fields": list(self.restored_fields),
            "irreversible_fields": list(self.irreversible_fields),
            "missing_fields": list(self.missing_fields),
            "completeness": self.completeness,
        }


@dataclass
class ConversionJournal:
    events: list[ConversionEvent] = field(default_factory=list)

    def add(self, event: ConversionEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda item: (item.sequence, item.receipt_uid, item.to_version))

    def as_dict(self) -> dict[str, Any]:
        events = [event.as_dict() for event in self.events]
        return {
            "schema_version": "mmledger-conversion-ledger-v1",
            "entries": events,
            "events": events,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConversionJournal":
        items = raw.get("entries", raw.get("events", []))
        if not isinstance(items, list):
            return cls()
        events: list[ConversionEvent] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            try:
                events.append(
                    ConversionEvent(
                        sequence=int(item["sequence"]),
                        receipt_uid=str(item["receipt_uid"]),
                        sidecar_key=str(item["sidecar_key"]),
                        operation=str(item["operation"]),
                        from_version=str(item["from_version"]),
                        to_version=str(item["to_version"]),
                        source_version=str(item["source_version"]),
                        downgraded_fields=tuple(item.get("downgraded_fields", [])),
                        restored_fields=tuple(item.get("restored_fields", [])),
                        irreversible_fields=tuple(item.get("irreversible_fields", [])),
                        missing_fields=tuple(item.get("missing_fields", [])),
                        completeness=str(item.get("completeness", "complete")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        events.sort(key=lambda event: (event.sequence, event.receipt_uid, event.to_version))
        return cls(events=events)


def convert_receipt(
    receipt: Receipt,
    target_version: str,
    sidecar: SidecarStore,
    *,
    sequence: int = 1,
    prune_fields: tuple[str, ...] = (),
) -> tuple[Receipt | None, ConversionEvent | None, Failure | None]:
    if target_version not in VERSIONS:
        return (
            None,
            None,
            Failure(
                E_CONVERSION_TARGET,
                f"unsupported target version {target_version!r}",
                receipt.source_file or "conversion",
                receipt.source_index,
                item=receipt.uid,
            ),
        )

    current_fields = VERSION_FIELDS[receipt.format_version]
    target_fields = VERSION_FIELDS[target_version]
    dropped = tuple(name for name in current_fields if name not in target_fields)
    added = tuple(name for name in target_fields if name not in current_fields)
    restored: list[str] = []
    irreversible: list[str] = []

    if dropped:
        sidecar.capture(receipt, dropped)
    pruned_fields: tuple[str, ...] = ()
    if prune_fields:
        pruned_fields = sidecar.prune(receipt, prune_fields)

    changed: dict[str, Any] = {}
    for name in dropped:
        changed[name] = MISSING
    for name in target_fields:
        if name in current_fields:
            continue
        restored_value, found = sidecar.restore(receipt, name)
        if found:
            changed[name] = restored_value
            restored.append(name)
        else:
            changed[name] = MISSING
            irreversible.append(name)

    if target_version < receipt.format_version:
        operation = "downgrade"
    elif target_version > receipt.format_version:
        operation = "upgrade"
    else:
        operation = "rewrite"

    new_provenance = dict(receipt.provenance)
    conversions = list(new_provenance.get("conversions", []))
    conversions.append({"from": receipt.format_version, "to": target_version, "operation": operation})
    new_provenance["conversions"] = conversions
    converted = replace(
        receipt,
        **changed,
        format_version=target_version,
        provenance=new_provenance,
    )
    for name in VERSION_FIELDS[target_version]:
        if is_missing(getattr(converted, name)):
            if name not in irreversible:
                irreversible.append(name)
    irreversible = sorted(set(irreversible).union(pruned_fields))

    event = ConversionEvent(
        sequence=sequence,
        receipt_uid=receipt.uid,
        sidecar_key=sidecar_key(receipt),
        operation=operation,
        from_version=receipt.format_version,
        to_version=target_version,
        source_version=receipt.source_version,
        downgraded_fields=dropped,
        restored_fields=tuple(restored),
        irreversible_fields=tuple(irreversible),
        missing_fields=converted.missing_fields,
        completeness=converted.completeness,
    )
    return converted, event, None


def receipt_to_file_payload(receipt: Receipt) -> dict[str, Any]:
    return {
        "version": receipt.format_version,
        "receipts": [receipt.as_fields(include_all=False)],
        "notes": {
            "source_version": receipt.source_version,
            "completeness": receipt.completeness,
            "missing_fields": list(receipt.missing_fields),
        },
    }


def write_conversion_bundle(
    receipt: Receipt,
    *,
    receipt_path: Path,
    sidecar_path: Path | None,
    sidecar: SidecarStore,
    journal: ConversionJournal,
    journal_path: Path | None,
) -> None:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(pretty_dumps(receipt_to_file_payload(receipt)), encoding="utf-8")
    if sidecar_path is not None:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(pretty_dumps(sidecar.as_dict()), encoding="utf-8")
    if journal_path is not None:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(pretty_dumps(journal.as_dict()), encoding="utf-8")


def restored_value(receipt: Receipt, field_name: str) -> Any:
    if field_name not in ALL_RECEIPT_FIELDS:
        raise KeyError(field_name)
    value = getattr(receipt, field_name)
    return None if is_missing(value) else value
