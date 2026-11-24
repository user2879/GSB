"""Receipt codecs for the three historical formats and the unified view.

Formats:
  r1: batch_id, request_id, output_digest
  r2: r1 + param_version, elapsed_ms, worker_id
  r3: r2 + rank, duplicate_ref, param_digest

Decoding detects the format version, validates required fields and raises
``E_FIELD`` for unknown versions, missing fields, negative elapsed time, out of
range rank or dangling duplicate references. Normalization fills absent fields
with explicit, traceable defaults and records both the source version and the
set of filled fields together with per-field rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import sha256_hex
from .errors import ChainError, FieldError

SUPPORTED_VERSIONS = ("r1", "r2", "r3")
DEFAULT_PARAM_VERSION = 1
DEFAULT_ELAPSED_MS = 0
DEFAULT_WORKER_ID = "unknown"
DEFAULT_RANK = 0
DEFAULT_DUPLICATE_REF: str | None = None
DEFAULT_PARAM_DIGEST = "unavailable"


@dataclass(frozen=True)
class FillRecord:
    field_name: str
    default: Any
    rationale: str

    def to_dict(self) -> dict:
        return {"field": self.field_name, "default": self.default, "rationale": self.rationale}


@dataclass
class NormalizedReceipt:
    source_version: str
    batch_id: str
    request_id: str
    output_digest: str
    param_version: int
    elapsed_ms: int
    worker_id: str
    rank: int
    duplicate_ref: str | None
    param_digest: str
    filled_fields: tuple[FillRecord, ...] = field(default_factory=tuple)

    def filled_field_names(self) -> tuple[str, ...]:
        return tuple(record.field_name for record in self.filled_fields)

    def to_dict(self) -> dict:
        return {
            "source_version": self.source_version,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_digest": self.output_digest,
            "param_version": self.param_version,
            "elapsed_ms": self.elapsed_ms,
            "worker_id": self.worker_id,
            "rank": self.rank,
            "duplicate_ref": self.duplicate_ref,
            "param_digest": self.param_digest,
            "filled_fields": [record.to_dict() for record in self.filled_fields],
        }


def _require(mapping: dict, key: str, version: str, entry: str) -> Any:
    if key not in mapping:
        raise FieldError(
            f"required field {key!r} missing in {version} receipt",
            entry=entry,
            field=key,
            format=version,
        )
    return mapping[key]


def _require_int(mapping: dict, key: str, version: str, entry: str) -> int:
    value = _require(mapping, key, version, entry)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FieldError(
            f"field {key!r} must be an integer in {version} receipt",
            entry=entry,
            field=key,
            value=repr(value),
        )
    return value


def _require_str(mapping: dict, key: str, version: str, entry: str) -> str:
    value = _require(mapping, key, version, entry)
    if not isinstance(value, str) or not value:
        raise FieldError(
            f"field {key!r} must be a non-empty string in {version} receipt",
            entry=entry,
            field=key,
            value=repr(value),
        )
    return value


def decode_receipt(raw: dict, known_request_ids: set[str] | None = None) -> NormalizedReceipt:
    """Decode a raw receipt dict into the unified internal view."""

    if not isinstance(raw, dict):
        raise FieldError("receipt must be an object", entry="<receipt>", field="<entry>")
    version = raw.get("format_version")
    if version not in SUPPORTED_VERSIONS:
        raise FieldError(
            "unknown receipt format version",
            entry="<receipt>",
            field="format_version",
            value=repr(version),
        )
    entry = f"receipt[{version}]"
    batch_id = _require_str(raw, "batch_id", version, entry)
    request_id = _require_str(raw, "request_id", version, entry)
    output_digest = _require_str(raw, "output_digest", version, entry)

    filled: list[FillRecord] = []

    if version == "r1":
        param_version = DEFAULT_PARAM_VERSION
        elapsed_ms = DEFAULT_ELAPSED_MS
        worker_id = DEFAULT_WORKER_ID
        rank = DEFAULT_RANK
        duplicate_ref = DEFAULT_DUPLICATE_REF
        param_digest = DEFAULT_PARAM_DIGEST
        filled.extend(
            [
                FillRecord(
                    "param_version",
                    DEFAULT_PARAM_VERSION,
                    "r1 predates parameter versioning; anchor to initial version 1",
                ),
                FillRecord(
                    "elapsed_ms",
                    DEFAULT_ELAPSED_MS,
                    "r1 carries no timing; use explicit integer placeholder 0",
                ),
                FillRecord(
                    "worker_id",
                    DEFAULT_WORKER_ID,
                    "r1 predates worker identity; use explicit sentinel 'unknown'",
                ),
                FillRecord(
                    "rank",
                    DEFAULT_RANK,
                    "r1 predates in-batch rank; use explicit stable default 0",
                ),
                FillRecord(
                    "duplicate_ref",
                    DEFAULT_DUPLICATE_REF,
                    "r1 predates duplicate delivery; use explicit null (no override)",
                ),
                FillRecord(
                    "param_digest",
                    DEFAULT_PARAM_DIGEST,
                    "r1 predates parameter snapshots; use explicit sentinel 'unavailable'",
                ),
            ]
        )
        return NormalizedReceipt(
            version,
            batch_id,
            request_id,
            output_digest,
            param_version,
            elapsed_ms,
            worker_id,
            rank,
            duplicate_ref,
            param_digest,
            tuple(filled),
        )

    param_version = _require_int(raw, "param_version", version, entry)
    elapsed_ms = _require_int(raw, "elapsed_ms", version, entry)
    if elapsed_ms < 0:
        raise FieldError(
            "elapsed_ms must be a non-negative integer",
            entry=entry,
            field="elapsed_ms",
            value=elapsed_ms,
        )
    worker_id = _require_str(raw, "worker_id", version, entry)

    if version == "r2":
        rank = DEFAULT_RANK
        duplicate_ref = DEFAULT_DUPLICATE_REF
        param_digest = DEFAULT_PARAM_DIGEST
        filled.extend(
            [
                FillRecord(
                    "rank",
                    DEFAULT_RANK,
                    "r2 predates stable rank; use explicit stable default 0",
                ),
                FillRecord(
                    "duplicate_ref",
                    DEFAULT_DUPLICATE_REF,
                    "r2 predates duplicate delivery; use explicit null (no override)",
                ),
                FillRecord(
                    "param_digest",
                    DEFAULT_PARAM_DIGEST,
                    "r2 predates parameter snapshots; use explicit sentinel 'unavailable'",
                ),
            ]
        )
        return NormalizedReceipt(
            version,
            batch_id,
            request_id,
            output_digest,
            param_version,
            elapsed_ms,
            worker_id,
            rank,
            duplicate_ref,
            param_digest,
            tuple(filled),
        )

    rank = _require_int(raw, "rank", version, entry)
    if rank < 0:
        raise FieldError(
            "rank must be a non-negative integer",
            entry=entry,
            field="rank",
            value=rank,
        )
    if "duplicate_ref" not in raw:
        raise FieldError(
            "required field 'duplicate_ref' missing in r3 receipt",
            entry=entry,
            field="duplicate_ref",
            format="r3",
        )
    duplicate_ref = raw["duplicate_ref"]
    if duplicate_ref is not None:
        if not isinstance(duplicate_ref, str) or not duplicate_ref:
            raise FieldError(
                "duplicate_ref must be null or a non-empty string",
                entry=entry,
                field="duplicate_ref",
                value=repr(duplicate_ref),
            )
        if known_request_ids is not None and duplicate_ref not in known_request_ids:
            raise FieldError(
                "duplicate_ref points at an unknown request (dangling reference)",
                entry=entry,
                field="duplicate_ref",
                value=duplicate_ref,
            )
    param_digest = _require_str(raw, "param_digest", version, entry)
    return NormalizedReceipt(
        version,
        batch_id,
        request_id,
        output_digest,
        param_version,
        elapsed_ms,
        worker_id,
        rank,
        duplicate_ref,
        param_digest,
        tuple(filled),
    )


def encode_receipt(receipt: NormalizedReceipt, version: str | None = None) -> dict:
    """Encode the unified view back into one of the three historical formats."""

    target = version or receipt.source_version
    if target not in SUPPORTED_VERSIONS:
        raise FieldError(
            "unknown receipt format version",
            entry="<receipt>",
            field="format_version",
            value=repr(target),
        )
    base = {
        "format_version": target,
        "batch_id": receipt.batch_id,
        "request_id": receipt.request_id,
        "output_digest": receipt.output_digest,
    }
    if target == "r1":
        return base
    base.update(
        {
            "param_version": receipt.param_version,
            "elapsed_ms": receipt.elapsed_ms,
            "worker_id": receipt.worker_id,
        }
    )
    if target == "r2":
        return base
    base.update(
        {
            "rank": receipt.rank,
            "duplicate_ref": receipt.duplicate_ref,
            "param_digest": receipt.param_digest,
        }
    )
    return base


def validate_chain(receipts: list[NormalizedReceipt]) -> None:
    """Cross-receipt dependency check: duplicate overrides must resolve.

    Raises ``E_CHAIN`` at the first dangling reference. Used by the replay and
    verification paths where a reference may have been tampered with after the
    original ``E_FIELD`` decode validation.
    """

    known = {receipt.request_id for receipt in receipts}
    for receipt in receipts:
        if receipt.duplicate_ref is not None and receipt.duplicate_ref not in known:
            raise ChainError(
                "duplicate_ref dangles: overridden request id is not present",
                entry=f"receipt[{receipt.source_version}]",
                field="duplicate_ref",
                request_id=receipt.request_id,
                value=receipt.duplicate_ref,
            )


def batch_summary_digest(receipts: list[NormalizedReceipt]) -> str:
    """Order-independent digest over the full normalized batch view.

    Sorting by request id guarantees that changing in-batch processing order
    never changes the summary.
    """

    payload = sorted(
        (receipt.to_dict() for receipt in receipts),
        key=lambda item: item["request_id"],
    )
    return sha256_hex(payload)
