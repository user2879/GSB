"""r1/r2/r3 receipt codec, validation and normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_json, digest_json, value_byte_length
from .errors import DiagnosticError, fail


R1_FIELDS = ("batch_id", "request_id", "output_digest")
R2_FIELDS = R1_FIELDS + ("param_version", "duration_ms", "worker_id")
R3_FIELDS = R2_FIELDS + ("rank", "duplicate_refs", "param_digest")
FIELD_NAMES = {"r1": R1_FIELDS, "r2": R2_FIELDS, "r3": R3_FIELDS}


@dataclass(frozen=True)
class DecodedReceipt:
    source_format: str
    batch_id: str
    request_id: str
    output_digest: str
    param_version: int | None = None
    duration_ms: int | None = None
    worker_id: str | None = None
    rank: int | None = None
    duplicate_refs: tuple[str, ...] = ()
    param_digest: str | None = None
    raw_fields: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class NormalizationContext:
    """Authoritative batch data used only for fields absent from old formats."""

    worker_id: str
    param_version: int
    duration_ms: int
    rank: int
    duplicate_refs: tuple[str, ...]
    param_digest: str


@dataclass(frozen=True)
class NormalizedReceipt:
    source_format: str
    filled_fields: tuple[str, ...]
    fill_basis: Mapping[str, str]
    batch_id: str
    request_id: str
    output_digest: str
    worker_id: str | None
    param_version: int | None
    duration_ms: int | None
    rank: int | None
    duplicate_refs: tuple[str, ...]
    param_digest: str | None

    def attribution(self) -> dict[str, Any]:
        """The format-independent ownership conclusion.

        Wall-clock duration and provenance metadata are deliberately excluded.
        """

        return {
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_digest": self.output_digest,
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "rank": self.rank,
            "duplicate_refs": list(self.duplicate_refs),
            "param_digest": self.param_digest,
        }

    def attribution_digest(self) -> str:
        return digest_json({"schema": "receipt-attribution-v1", **self.attribution()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "filled_fields": list(self.filled_fields),
            "fill_basis": dict(sorted(self.fill_basis.items())),
            "attribution": self.attribution(),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class DecodedBatch:
    receipts: tuple[DecodedReceipt, ...]
    batch_id: str
    known_request_ids: frozenset[str] = field(default_factory=frozenset)


def _receipt_error(index: int, field_path: str, detail: str) -> "None":
    fail("E_FIELD", f"receipt[{index}].{field_path}", detail)


def _require_string(value: Any, index: int, field: str) -> str:
    if not isinstance(value, str) or not value:
        _receipt_error(index, field, "expected a non-empty string")
    return value


def _require_int(value: Any, index: int, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _receipt_error(index, field, "expected an integer (booleans are not integers)")
    if minimum is not None and value < minimum:
        _receipt_error(index, field, f"must be >= {minimum}")
    return value


def encode_receipt(format_version: str, values: Mapping[str, Any]) -> str:
    """Encode a receipt as canonical JSON with byte lengths for every field."""

    if format_version not in FIELD_NAMES:
        fail("E_FIELD", "receipt.format", f"unsupported receipt format: {format_version}")
    field_names = FIELD_NAMES[format_version]
    fields: list[dict[str, Any]] = []
    for name in field_names:
        if name not in values:
            fail("E_FIELD", f"receipt.{name}", "missing required field")
        value = values[name]
        fields.append(
            {
                "name": name,
                "length": value_byte_length(value),
                "value": value,
            }
        )
    return canonical_json(
        {
            "format": format_version,
            "field_count": len(fields),
            "fields": fields,
        }
    )


def decode_receipt(
    payload: str | bytes,
    *,
    receipt_index: int = 0,
    known_request_ids: Iterable[str] | None = None,
) -> DecodedReceipt:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            _receipt_error(receipt_index, "utf8", f"invalid UTF-8: {exc.reason}")
    if not isinstance(payload, str):
        _receipt_error(receipt_index, "payload", "expected JSON text")
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        _receipt_error(receipt_index, "payload", f"invalid JSON: {exc.msg}")
    if not isinstance(document, dict):
        _receipt_error(receipt_index, "payload", "expected an object")
    if set(document) != {"format", "field_count", "fields"}:
        _receipt_error(receipt_index, "fields", "envelope must contain exactly format, field_count, fields")
    format_version = document["format"]
    if format_version not in FIELD_NAMES:
        _receipt_error(receipt_index, "format", f"unsupported format: {format_version!r}")
    field_count = document["field_count"]
    if type(field_count) is not int or field_count < 0:
        _receipt_error(receipt_index, "field_count", "expected a non-negative integer")
    raw_fields = document["fields"]
    if not isinstance(raw_fields, list):
        _receipt_error(receipt_index, "fields", "expected an array")
    if field_count != len(raw_fields):
        _receipt_error(
            receipt_index,
            "field_count",
            f"declared {field_count} but found {len(raw_fields)} fields",
        )
    expected_names = FIELD_NAMES[format_version]
    if len(raw_fields) != len(expected_names):
        _receipt_error(
            receipt_index,
            "fields",
            f"{format_version} requires {len(expected_names)} fields, found {len(raw_fields)}",
        )

    decoded_values: dict[str, Any] = {}
    raw_pairs: list[tuple[str, Any]] = []
    for position, (expected_name, entry) in enumerate(zip(expected_names, raw_fields)):
        field_path = f"fields[{position}]"
        if not isinstance(entry, dict) or set(entry) != {"name", "length", "value"}:
            _receipt_error(receipt_index, field_path, "expected exactly name, length, value")
        name = entry["name"]
        if name != expected_name:
            _receipt_error(
                receipt_index,
                f"{field_path}.name",
                f"expected field {expected_name!r}, found {name!r}",
            )
        declared_length = entry["length"]
        if type(declared_length) is not int or declared_length < 0:
            _receipt_error(receipt_index, f"{field_path}.length", "expected a non-negative integer")
        actual_length = value_byte_length(entry["value"])
        if declared_length != actual_length:
            _receipt_error(
                receipt_index,
                f"{field_path}.length",
                f"declared {declared_length} bytes but value uses {actual_length}",
            )
        decoded_values[expected_name] = entry["value"]
        raw_pairs.append((expected_name, entry["value"]))

    batch_id = _require_string(decoded_values["batch_id"], receipt_index, "batch_id")
    request_id = _require_string(decoded_values["request_id"], receipt_index, "request_id")
    output_digest = _require_string(
        decoded_values["output_digest"], receipt_index, "output_digest"
    )
    param_version: int | None = None
    duration_ms: int | None = None
    worker_id: str | None = None
    rank: int | None = None
    duplicate_refs: tuple[str, ...] = ()
    param_digest: str | None = None

    if format_version in ("r2", "r3"):
        param_version = _require_int(
            decoded_values["param_version"],
            receipt_index,
            "param_version",
            minimum=0,
        )
        duration_ms = _require_int(
            decoded_values["duration_ms"],
            receipt_index,
            "duration_ms",
            minimum=0,
        )
        worker_id = _require_string(decoded_values["worker_id"], receipt_index, "worker_id")
    if format_version == "r3":
        rank = _require_int(decoded_values["rank"], receipt_index, "rank", minimum=0)
        raw_duplicates = decoded_values["duplicate_refs"]
        if not isinstance(raw_duplicates, list):
            _receipt_error(receipt_index, "duplicate_refs", "expected an array")
        references: list[str] = []
        seen: set[str] = set()
        known = set(known_request_ids) if known_request_ids is not None else None
        for duplicate_index, value in enumerate(raw_duplicates):
            if not isinstance(value, str) or not value:
                _receipt_error(
                    receipt_index,
                    f"duplicate_refs[{duplicate_index}]",
                    "expected a non-empty request id",
                )
            if value in seen:
                _receipt_error(
                    receipt_index,
                    f"duplicate_refs[{duplicate_index}]",
                    f"duplicate reference: {value}",
                )
            if value == request_id:
                _receipt_error(
                    receipt_index,
                    f"duplicate_refs[{duplicate_index}]",
                    "a receipt cannot reference its own request",
                )
            if known is not None and value not in known:
                _receipt_error(
                    receipt_index,
                    f"duplicate_refs[{duplicate_index}]",
                    f"dangling request reference: {value}",
                )
            seen.add(value)
            references.append(value)
        duplicate_refs = tuple(sorted(references))
        param_digest = _require_string(
            decoded_values["param_digest"], receipt_index, "param_digest"
        )

    return DecodedReceipt(
        source_format=format_version,
        batch_id=batch_id,
        request_id=request_id,
        output_digest=output_digest,
        param_version=param_version,
        duration_ms=duration_ms,
        worker_id=worker_id,
        rank=rank,
        duplicate_refs=duplicate_refs,
        param_digest=param_digest,
        raw_fields=tuple(raw_pairs),
    )


def decode_receipt_batch(
    payloads: Sequence[str | bytes],
    *,
    known_request_ids: Iterable[str] | None = None,
    batch_size: int | None = None,
) -> DecodedBatch:
    if not payloads:
        fail("E_FIELD", "receipt[0]", "at least one receipt is required")
    receipts = tuple(
        decode_receipt(
            payload,
            receipt_index=index,
            known_request_ids=known_request_ids,
        )
        for index, payload in enumerate(payloads)
    )
    batch_id = receipts[0].batch_id
    request_ids: set[str] = set()
    output_digests: set[str] = set()
    ranks: list[tuple[int, int]] = []
    for index, receipt in enumerate(receipts):
        if receipt.batch_id != batch_id:
            _receipt_error(index, "batch_id", "all receipts in a batch must share batch_id")
        if receipt.request_id in request_ids:
            _receipt_error(index, "request_id", f"duplicate request_id: {receipt.request_id}")
        request_ids.add(receipt.request_id)
        output_digests.add(receipt.output_digest)
        if receipt.rank is not None:
            ranks.append((index, receipt.rank))
    if len(output_digests) != 1:
        fail("E_FIELD", "receipt.output_digest", "receipts in one batch disagree on output_digest")

    if ranks:
        if batch_size is None:
            fail("E_FIELD", "batch_size", "required to validate r3 rank bounds")
        if type(batch_size) is not int or batch_size < 1:
            fail("E_FIELD", "batch_size", "expected a positive integer")
        if len(receipts) > batch_size:
            fail(
                "E_FIELD",
                "receipt",
                f"{len(receipts)} receipts exceed batch_size {batch_size}",
            )
        seen_ranks: set[int] = set()
        for index, rank in ranks:
            if rank >= batch_size:
                _receipt_error(index, "rank", f"rank {rank} is outside [0, {batch_size})")
            if rank in seen_ranks:
                _receipt_error(index, "rank", f"duplicate rank in batch: {rank}")
            seen_ranks.add(rank)
        if len(ranks) == batch_size and seen_ranks != set(range(batch_size)):
            fail("E_FIELD", "receipt.rank", "r3 ranks must cover the whole batch exactly")
    return DecodedBatch(
        receipts=receipts,
        batch_id=batch_id,
        known_request_ids=frozenset(known_request_ids or ()),
    )


def normalize_receipt(
    receipt: DecodedReceipt,
    context: NormalizationContext | None = None,
) -> NormalizedReceipt:
    missing: list[str] = []
    basis: dict[str, str] = {}

    def fill(value: Any, name: str, fallback: Any) -> Any:
        if value is not None:
            return value
        missing.append(name)
        if context is None:
            basis[name] = f"absent_from_{receipt.source_format}_and_no_authoritative_context"
            return fallback
        basis[name] = "authoritative_batch_context"
        return getattr(context, name)

    # An empty duplicate list is a value only for r3. r1/r2 therefore use a
    # sentinel during normalization while retaining their source field set.
    duplicate_value: tuple[str, ...] | None
    if receipt.source_format == "r3":
        duplicate_value = receipt.duplicate_refs
    else:
        duplicate_value = None

    worker_id = fill(receipt.worker_id, "worker_id", None)
    param_version = fill(receipt.param_version, "param_version", None)
    duration_ms = fill(receipt.duration_ms, "duration_ms", None)
    rank = fill(receipt.rank, "rank", None)
    duplicate_refs = fill(duplicate_value, "duplicate_refs", ())
    param_digest = fill(receipt.param_digest, "param_digest", None)
    return NormalizedReceipt(
        source_format=receipt.source_format,
        filled_fields=tuple(missing),
        fill_basis=dict(basis),
        batch_id=receipt.batch_id,
        request_id=receipt.request_id,
        output_digest=receipt.output_digest,
        worker_id=worker_id,
        param_version=param_version,
        duration_ms=duration_ms,
        rank=rank,
        duplicate_refs=duplicate_refs,
        param_digest=param_digest,
    )


def normalize_batch(
    decoded: DecodedBatch,
    contexts: Mapping[str, NormalizationContext] | None = None,
) -> tuple[NormalizedReceipt, ...]:
    contexts = contexts or {}
    return tuple(
        normalize_receipt(receipt, contexts.get(receipt.request_id))
        for receipt in decoded.receipts
    )


def normalize_mixed_receipts(
    payloads: Sequence[str | bytes],
    *,
    known_request_ids: Iterable[str],
    batch_size: int,
    contexts: Mapping[str, NormalizationContext],
) -> tuple[NormalizedReceipt, ...]:
    decoded = decode_receipt_batch(
        payloads,
        known_request_ids=known_request_ids,
        batch_size=batch_size,
    )
    return normalize_batch(decoded, contexts)
