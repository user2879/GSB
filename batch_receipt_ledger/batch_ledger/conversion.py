from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import MISSING, OPTIONAL_FIELDS, CanonicalReceipt, is_missing
from .receipts import parse_receipt, receipt_to_version_dict

SIDECAR_FORMAT = "receipt-sidecar-v1"


@dataclass
class SidecarStore:
    """Preserves fields that are absent when downgraded to an older schema."""

    format: str = SIDECAR_FORMAT
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def put_receipt(self, receipt: CanonicalReceipt, *, sidecar_key: str | None = None) -> str:
        key = sidecar_key or receipt.receipt_ref
        preserved = {name: getattr(receipt, name) for name in OPTIONAL_FIELDS}
        self.entries[key] = {
            "origin_version": receipt.origin_version or receipt.source_version,
            "receipt_ref": receipt.receipt_ref,
            "fields": preserved,
        }
        return key

    def has(self, key: str) -> bool:
        return key in self.entries

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "entries": dict(sorted(self.entries.items()))}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SidecarStore":
        if not data:
            return cls()
        return cls(format=data.get("format", SIDECAR_FORMAT), entries=dict(data.get("entries", {})))


def upgrade_document(
    doc: dict[str, Any],
    target_version: str,
    sidecar: SidecarStore | dict[str, Any] | None = None,
    *,
    sidecar_key: str | None = None,
    restore_origin: bool = True,
) -> CanonicalReceipt:
    """Parse a low-version document and restore missing fields from sidecar."""

    current = parse_receipt(doc)
    if target_version == current.source_version:
        return current
    if target_version not in ("r1", "r2", "r3"):
        raise ValueError(f"unsupported target version {target_version}")

    store = sidecar if isinstance(sidecar, SidecarStore) else SidecarStore.from_dict(sidecar)
    key = sidecar_key or current.receipt_ref
    record = store.get(key)
    restored: dict[str, Any] = {name: getattr(current, name) for name in OPTIONAL_FIELDS}
    origin_version = target_version
    if record:
        fields = record.get("fields", {})
        for name in OPTIONAL_FIELDS:
            if is_missing(restored.get(name)) and name in fields:
                restored[name] = fields[name]
        if restore_origin:
            origin_version = record.get("origin_version", target_version)

    kwargs = {
        "source_version": target_version,
        "origin_version": origin_version,
        "batch_id": current.batch_id,
        "request_id": current.request_id,
        "output_digest": current.output_digest,
        "receipt_ref": current.receipt_ref,
    }
    kwargs.update(restored)
    return CanonicalReceipt(**kwargs)


def convert_receipt(
    receipt: CanonicalReceipt,
    target_version: str,
    *,
    sidecar: SidecarStore | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Convert to a target schema. Downgrades automatically write sidecar."""

    if target_version not in ("r1", "r2", "r3"):
        raise ValueError(f"unsupported target version {target_version}")
    sidecar_entry = None
    rank = {"r1": 1, "r2": 2, "r3": 3}
    if sidecar is not None and rank[target_version] < rank[receipt.source_version]:
        key = sidecar.put_receipt(receipt)
        sidecar_entry = sidecar.get(key)
    doc = receipt_to_version_dict(receipt, target_version)
    return doc, sidecar_entry


@dataclass
class ConversionBook:
    downgrade_entries: list[dict[str, Any]] = field(default_factory=list)
    conversion_entries: list[dict[str, Any]] = field(default_factory=list)

    def record_downgrade(
        self,
        receipt: CanonicalReceipt,
        target_version: str,
        sidecar_key: str,
        *,
        ordinal: int | None = None,
    ) -> dict[str, Any]:
        entry = {
            "entry_key": f"{receipt.receipt_ref}:{receipt.source_version}->{target_version}",
            "ordinal": len(self.downgrade_entries) if ordinal is None else ordinal,
            "receipt_ref": receipt.receipt_ref,
            "from_version": receipt.source_version,
            "to_version": target_version,
            "sidecar_key": sidecar_key,
            "preserved_fields": list(OPTIONAL_FIELDS),
            "degraded_r3_fields": ["rank_in_batch", "supersedes_ref", "param_digest"],
        }
        self.downgrade_entries.append(entry)
        self.downgrade_entries.sort(key=lambda item: (item["ordinal"], item["entry_key"]))
        return entry

    def record_conversion(self, receipt: CanonicalReceipt, target_version: str, restored: bool) -> None:
        self.conversion_entries.append(
            {
                "entry_key": f"{receipt.receipt_ref}:{receipt.source_version}->{target_version}",
                "receipt_ref": receipt.receipt_ref,
                "source_version": receipt.source_version,
                "origin_version": receipt.origin_version,
                "target_version": target_version,
                "restored_from_sidecar": restored,
                "completeness_label": receipt.completeness_label,
                "completeness_score": receipt.completeness_score,
            }
        )
        self.conversion_entries.sort(key=lambda item: item["entry_key"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversion_entry_count": len(self.conversion_entries),
            "downgrade_entry_count": len(self.downgrade_entries),
            "downgrade_entries": list(self.downgrade_entries),
            "conversion_entries": list(self.conversion_entries),
        }


def completeness_stats(receipts: list[CanonicalReceipt]) -> dict[str, Any]:
    by_version = {"r1": 0, "r2": 0, "r3": 0}
    by_label: dict[str, int] = {}
    missing_field_counts = {name: 0 for name in OPTIONAL_FIELDS}
    complete = 0
    for receipt in receipts:
        by_version[receipt.source_version] = by_version.get(receipt.source_version, 0) + 1
        by_label[receipt.completeness_label] = by_label.get(receipt.completeness_label, 0) + 1
        if not receipt.missing_fields:
            complete += 1
        for name in receipt.missing_fields:
            missing_field_counts[name] += 1
    return {
        "entry_count": len(receipts),
        "complete_count": complete,
        "by_source_version": by_version,
        "by_completeness_label": dict(sorted(by_label.items())),
        "missing_field_counts": missing_field_counts,
    }


def conversion_roundtrip_audit(receipts: list[CanonicalReceipt]) -> dict[str, Any]:
    """Downgrade versioned receipts to r1 with sidecar and restore them."""

    sidecar = SidecarStore()
    book = ConversionBook()
    restored_receipts: list[CanonicalReceipt] = []
    field_matches: list[bool] = []
    ordered = sorted(
        receipts,
        key=lambda item: (item.batch_id, item.request_id, item.source_version, item.receipt_ref),
    )

    for ordinal, direct in enumerate(ordered):
        if direct.source_version == "r1":
            restored = direct
        else:
            key = sidecar.put_receipt(direct)
            book.record_downgrade(direct, "r1", key, ordinal=ordinal)
            r1_document = {
                "v": "r1",
                "batch_id": direct.batch_id,
                "request_id": direct.request_id,
                "output_digest": direct.output_digest,
            }
            restored = upgrade_document(r1_document, direct.source_version, sidecar)
            book.record_conversion(restored, direct.source_version, True)
        restored_receipts.append(restored)
        field_matches.append(restored.to_dict() == direct.to_dict())

    direct_stats = completeness_stats(list(ordered))
    roundtrip_stats = completeness_stats(restored_receipts)
    r3_downgrades = sum(1 for item in book.downgrade_entries if item["from_version"] == "r3")
    r2_downgrades = sum(1 for item in book.downgrade_entries if item["from_version"] == "r2")
    return {
        "one_to_one": len(restored_receipts) == len(receipts),
        "entry_count": len(receipts),
        "all_fields_match": all(field_matches),
        "mismatch_count": sum(1 for item in field_matches if not item),
        "r3_to_r1_downgrade_count": r3_downgrades,
        "r2_to_r1_downgrade_count": r2_downgrades,
        "completeness_matches": direct_stats == roundtrip_stats,
        "direct_completeness": direct_stats,
        "roundtrip_completeness": roundtrip_stats,
        "conversion_book": book.to_dict(),
        "sidecar": sidecar.to_dict(),
    }
