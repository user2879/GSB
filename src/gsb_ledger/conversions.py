"""Loss-aware conversion between receipt versions r1, r2 and r3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import Receipt, is_missing, jsonable, missing
from .util import atomic_write_json, read_json


VERSION_ORDER = {"r1": 1, "r2": 2, "r3": 3}
VERSION_FIELDS = {
    "r1": ("batch_id", "request_id", "output_digest"),
    "r2": (
        "batch_id",
        "request_id",
        "output_digest",
        "param_version",
        "duration_ms",
        "worker_id",
    ),
    "r3": (
        "batch_id",
        "request_id",
        "output_digest",
        "param_version",
        "duration_ms",
        "worker_id",
        "rank",
        "covered_refs",
        "param_digest",
    ),
}
RECEIPT_FIELDS = (
    "param_version",
    "duration_ms",
    "worker_id",
    "rank",
    "covered_refs",
    "param_digest",
)


@dataclass(frozen=True)
class DowngradeRecord:
    receipt_key: str
    source_version: str
    target_version: str
    preserved_fields: dict[str, Any]
    lost_fields: tuple[str, ...]
    sidecar_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_key": self.receipt_key,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "preserved_fields": jsonable(self.preserved_fields),
            "lost_fields": list(self.lost_fields),
            "sidecar_id": self.sidecar_id,
        }


@dataclass(frozen=True)
class ConversionResult:
    receipt: Receipt
    target_version: str
    source_version: str
    direction: str
    preserved_fields: dict[str, Any]
    lost_fields: tuple[str, ...]
    integrity_label: str
    downgrade: DowngradeRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_version": self.target_version,
            "source_version": self.source_version,
            "direction": self.direction,
            "preserved_fields": jsonable(self.preserved_fields),
            "lost_fields": list(self.lost_fields),
            "integrity_label": self.integrity_label,
            "downgrade": None if self.downgrade is None else self.downgrade.to_dict(),
        }


@dataclass
class Sidecar:
    """JSON side channel keyed by physical receipt identity."""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_version: int = 1

    def put(self, record: DowngradeRecord) -> None:
        existing = self.records.get(record.receipt_key)
        if existing is None:
            self.records[record.receipt_key] = record.to_dict()
            return
        preserved = dict(existing.get("preserved_fields", {}))
        for key, value in record.preserved_fields.items():
            if key not in preserved or not is_missing(value):
                preserved[key] = jsonable(value)
        existing["preserved_fields"] = preserved
        existing["source_version"] = record.source_version
        existing["target_version"] = record.target_version
        existing["lost_fields"] = sorted(
            set(existing.get("lost_fields", [])) | set(record.lost_fields)
        )

    def get(self, receipt_key: str) -> dict[str, Any] | None:
        return self.records.get(receipt_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "records": {
                key: self.records[key]
                for key in sorted(self.records)
            },
        }

    def save(self, path: str | Path) -> None:
        atomic_write_json(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "Sidecar":
        sidecar_path = Path(path)
        if not sidecar_path.exists():
            return cls()
        raw = read_json(sidecar_path)
        if not isinstance(raw, dict) or not isinstance(raw.get("records", {}), dict):
            return cls()
        return cls(
            records={
                str(key): value
                for key, value in raw.get("records", {}).items()
                if isinstance(value, dict)
            },
            schema_version=int(raw.get("schema_version", 1)),
        )


def _field_value(receipt: Receipt, field_name: str) -> Any:
    return getattr(receipt, field_name)


def _current_missing(receipt: Receipt) -> set[str]:
    result = set(receipt.missing_fields)
    for field_name in RECEIPT_FIELDS:
        if is_missing(_field_value(receipt, field_name)):
            result.add(field_name)
    return result


def _restore_value(raw: Any, field_name: str) -> Any:
    if isinstance(raw, dict) and "__missing__" in raw:
        return missing(field_name)
    if field_name == "covered_refs" and isinstance(raw, list):
        return tuple(raw)
    return raw


def convert_receipt(
    receipt: Receipt,
    target_version: str,
    *,
    sidecar: Sidecar | None = None,
    source_version: str | None = None,
) -> ConversionResult:
    if target_version not in VERSION_ORDER:
        raise ValueError(f"unsupported target version {target_version!r}")
    source_version = source_version or receipt.version
    if source_version not in VERSION_ORDER:
        raise ValueError(f"unsupported source version {source_version!r}")
    target_rank = VERSION_ORDER[target_version]
    source_rank = VERSION_ORDER[source_version]
    if target_rank == source_rank:
        return ConversionResult(
            receipt=receipt.with_fields(version=target_version),
            target_version=target_version,
            source_version=source_version,
            direction="identity",
            preserved_fields={},
            lost_fields=(),
            integrity_label=receipt.completeness,
        )

    current_missing = _current_missing(receipt)
    updates: dict[str, Any] = {"version": target_version}
    preserved: dict[str, Any] = {}
    lost_fields: list[str] = []
    sidecar_record = None if sidecar is None else sidecar.get(receipt.key)

    target_fields = set(VERSION_FIELDS[target_version])
    source_fields = set(VERSION_FIELDS[source_version])
    all_fields = set(VERSION_FIELDS["r3"])

    if target_rank < source_rank:
        # Downgrade: hide unsupported values in the sidecar instead of dropping them.
        for field_name in sorted(all_fields - target_fields):
            value = _field_value(receipt, field_name)
            if not is_missing(value):
                preserved[field_name] = value
            updates[field_name] = missing(field_name)
        truly_lost = [
            field_name
            for field_name in sorted(all_fields - target_fields)
            if field_name not in preserved
            and not is_missing(_field_value(receipt, field_name))
        ]
        missing_fields = sorted((current_missing | (all_fields - target_fields)))
        sidecar_id = f"sidecar:{receipt.key}:{source_version}->{target_version}"
        downgrade = DowngradeRecord(
            receipt_key=receipt.key,
            source_version=source_version,
            target_version=target_version,
            preserved_fields=preserved,
            lost_fields=tuple(truly_lost),
            sidecar_id=sidecar_id,
        )
        if sidecar is not None:
            sidecar.put(downgrade)
        converted = receipt.with_fields(
            **updates,
            missing_fields=tuple(missing_fields),
            completeness=(
                "downgraded-sidecar-complete" if not truly_lost else "downgraded-with-loss"
            ),
        )
        return ConversionResult(
            receipt=converted,
            target_version=target_version,
            source_version=source_version,
            direction="downgrade",
            preserved_fields=preserved,
            lost_fields=tuple(truly_lost),
            integrity_label=converted.completeness,
            downgrade=downgrade,
        )

    # Upgrade: restore fields from a sidecar when available. Missing source
    # fields remain explicit MISSING markers and are named in lost_fields.
    preserved_from_sidecar: dict[str, Any] = {}
    sidecar_preserved = (
        sidecar_record.get("preserved_fields", {})
        if isinstance(sidecar_record, dict)
        else {}
    )
    for field_name in sorted(target_fields - source_fields):
        value = _field_value(receipt, field_name)
        if not is_missing(value):
            continue
        if isinstance(sidecar_preserved, dict) and field_name in sidecar_preserved:
            restored = _restore_value(sidecar_preserved[field_name], field_name)
            updates[field_name] = restored
            preserved_from_sidecar[field_name] = restored
            current_missing.discard(field_name)
        else:
            updates[field_name] = missing(field_name)
            current_missing.add(field_name)
            lost_fields.append(field_name)

    converted = receipt.with_fields(
        **updates,
        missing_fields=tuple(sorted(current_missing)),
        completeness=(
            "restored-from-sidecar"
            if preserved_from_sidecar and not lost_fields
            else "partial-restore"
            if preserved_from_sidecar
            else "explicit-missing"
        ),
    )
    return ConversionResult(
        receipt=converted,
        target_version=target_version,
        source_version=source_version,
        direction="upgrade",
        preserved_fields=preserved_from_sidecar,
        lost_fields=tuple(sorted(set(lost_fields))),
        integrity_label=converted.completeness,
    )


def receipt_json_payload(receipt: Receipt) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "batch_id": receipt.batch_id,
        "request_id": receipt.request_id,
        "output_digest": receipt.output_digest,
    }
    for field_name in RECEIPT_FIELDS:
        if field_name in VERSION_FIELDS.get(receipt.version, ()) or not is_missing(
            _field_value(receipt, field_name)
        ):
            payload[field_name] = jsonable(_field_value(receipt, field_name))
    missing_fields = sorted(_current_missing(receipt))
    if missing_fields:
        payload["missing_fields"] = missing_fields
    return payload


def roundtrip_via_sidecar(receipt: Receipt, through_version: str) -> tuple[ConversionResult, ConversionResult, Sidecar]:
    sidecar = Sidecar()
    down = convert_receipt(receipt, through_version, sidecar=sidecar)
    up = convert_receipt(
        down.receipt,
        receipt.version,
        sidecar=sidecar,
        source_version=through_version,
    )
    return down, up, sidecar
