"""Decode, fill, validate, and normalize historical receipt formats."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DiagnosticError, is_int
from .jsonio import atomic_write_bytes, canonical_json, parse_json_text, sha256_canonical
from .model import batch_output_digest
from .planner import BatchPlan, DispatchPlan
from .spec import MergeRules, RequestSet, WorkerSet


HEX_64 = re.compile(r"^[0-9a-f]{64}$")
BASE_FIELDS = ("batch_id", "request_id", "output_digest")
FORMAT_FIELDS = {
    "r1": set(BASE_FIELDS),
    "r2": set(BASE_FIELDS)
    | {"param_version_seq", "latency_ms", "worker_id"},
    "r3": set(BASE_FIELDS)
    | {
        "param_version_seq",
        "latency_ms",
        "worker_id",
        "rank_in_batch",
        "superseded_request_ids",
        "param_digest",
    },
}


@dataclass(frozen=True)
class RawReceipt:
    format_version: str
    line_number: int
    source: str
    value: dict[str, Any]


@dataclass(frozen=True)
class NormalizedReceipt:
    source_format: str
    source_line: int
    source_object: str
    batch_id: str
    request_id: str
    output_digest: str
    worker_id: str
    param_version_seq: int
    latency_ms: int
    rank_in_batch: int
    superseded_request_ids: tuple[str, ...]
    param_digest: str
    filled_fields: tuple[str, ...]
    fill_basis: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "source_line": self.source_line,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_digest": self.output_digest,
            "worker_id": self.worker_id,
            "param_version_seq": self.param_version_seq,
            "latency_ms": self.latency_ms,
            "rank_in_batch": self.rank_in_batch,
            "superseded_request_ids": list(self.superseded_request_ids),
            "param_digest": self.param_digest,
            "filled_fields": list(self.filled_fields),
            "fill_basis": dict(self.fill_basis),
        }


@dataclass(frozen=True)
class NormalizedLedger:
    schema_version: int
    batches: tuple[dict[str, Any], ...]
    content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "batches": list(self.batches),
            "content_digest": self.content_digest,
        }


def read_receipt_file(
    path: str | Path,
    format_version: str,
) -> list[RawReceipt]:
    if format_version not in FORMAT_FIELDS:
        raise DiagnosticError(
            code="E_FIELD",
            message=f"unsupported receipt format {format_version!r}",
            object=str(path),
            field="format_version",
        )
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DiagnosticError(
            code="E_FIELD",
            message=f"cannot read receipt file: {exc.strerror or exc}",
            object=str(source),
            field="<file>",
        ) from exc
    receipts: list[RawReceipt] = []
    for line_number, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        value = parse_json_text(
            text, object_name=str(source), code="E_FIELD"
        )
        if not isinstance(value, dict):
            raise DiagnosticError(
                code="E_FIELD",
                message="receipt line must be a JSON object",
                object=str(source),
                field="<object>",
                line=line_number,
            )
        receipts.append(
            RawReceipt(
                format_version=format_version,
                line_number=line_number,
                source=str(source),
                value=value,
            )
        )
    if not receipts:
        raise DiagnosticError(
            code="E_FIELD",
            message="receipt file contains no receipt rows",
            object=str(source),
            field="<file>",
        )
    return receipts


def _error(
    receipt: RawReceipt,
    field: str,
    message: str,
    *,
    code: str = "E_FIELD",
) -> DiagnosticError:
    return DiagnosticError(
        code=code,
        message=message,
        object=receipt.source,
        field=field,
        line=receipt.line_number,
    )


def _require_string(receipt: RawReceipt, field: str) -> str:
    value = receipt.value[field]
    if not isinstance(value, str) or not value:
        raise _error(receipt, field, "expected a non-empty string")
    return value


def _require_int(receipt: RawReceipt, field: str, *, minimum: int = 0) -> int:
    value = receipt.value[field]
    if not is_int(value) or value < minimum:
        raise _error(receipt, field, f"expected an integer >= {minimum}")
    return value


def _require_digest(receipt: RawReceipt, field: str) -> str:
    value = _require_string(receipt, field)
    if HEX_64.fullmatch(value) is None:
        raise _error(receipt, field, "expected a lowercase SHA-256 digest")
    return value


def _validate_shape(receipt: RawReceipt) -> None:
    allowed = FORMAT_FIELDS[receipt.format_version]
    for field in receipt.value:
        if field not in allowed:
            raise _error(receipt, field, f"field is not part of {receipt.format_version}")
    for field in allowed:
        if field not in receipt.value:
            raise _error(receipt, field, "missing required receipt field")


def _resolve_fill(
    strategy: str,
    batch: BatchPlan,
    rank_by_request_id: dict[str, int],
    request_id: str,
) -> Any:
    if strategy == "batch.worker_id":
        return batch.worker_id
    if strategy == "batch.param_version_seq":
        return batch.param_version_seq
    if strategy == "batch.param_digest":
        return batch.param_digest
    if strategy == "request.arrival_rank":
        return rank_by_request_id[request_id]
    if strategy == "constant:0":
        return 0
    if strategy == "empty_list":
        return []
    raise ValueError(f"unhandled fill strategy {strategy}")


def normalize_receipts(
    raw_by_format: dict[str, Iterable[RawReceipt]],
    *,
    request_set: RequestSet,
    worker_set: WorkerSet,
    plan: DispatchPlan,
    rules: MergeRules,
) -> NormalizedLedger:
    plan_by_id = plan.by_id()
    worker_map = worker_set.by_id()
    existing_request_ids = set(request_set.by_id())
    normalized: list[NormalizedReceipt] = []
    rank_seen: dict[str, dict[int, RawReceipt]] = {}
    receipt_seen: dict[tuple[str, str], RawReceipt] = {}

    for format_version in ("r1", "r2", "r3"):
        fill = rules.fill_for(format_version)
        for receipt in raw_by_format.get(format_version, ()):
            _validate_shape(receipt)
            batch_id = _require_string(receipt, "batch_id")
            batch = plan_by_id.get(batch_id)
            if batch is None:
                raise _error(
                    receipt,
                    "batch_id",
                    f"unknown batch {batch_id!r}",
                    code="E_REF",
                )
            request_id = _require_string(receipt, "request_id")
            if request_id not in existing_request_ids:
                raise _error(
                    receipt,
                    "request_id",
                    f"unknown request {request_id!r}",
                    code="E_REF",
                )
            if request_id not in batch.request_ids:
                raise _error(
                    receipt,
                    "request_id",
                    f"request {request_id!r} is not assigned to batch {batch_id!r}",
                    code="E_REF",
                )
            digest = _require_digest(receipt, "output_digest")
            filled_fields: list[str] = []
            fill_basis: dict[str, str] = {}
            values = dict(receipt.value)
            rank_by_request_id = batch.rank_by_request_id()
            for field, strategy in sorted(fill.items()):
                if field not in values:
                    values[field] = _resolve_fill(
                        strategy, batch, rank_by_request_id, request_id
                    )
                    filled_fields.append(field)
                    fill_basis[field] = strategy

            worker_id = _string_value(receipt, values, "worker_id")
            if worker_id not in worker_map:
                raise _error(
                    receipt,
                    "worker_id",
                    f"unknown worker {worker_id!r}",
                    code="E_REF",
                )
            if worker_id != batch.worker_id:
                raise _error(
                    receipt,
                    "worker_id",
                    f"worker does not own batch {batch_id!r}",
                    code="E_REF",
                )
            param_version_seq = _int_value(
                receipt, values, "param_version_seq", minimum=1
            )
            latency_ms = _int_value(receipt, values, "latency_ms", minimum=0)
            rank_in_batch = _int_value(receipt, values, "rank_in_batch", minimum=0)
            param_digest = _digest_value(receipt, values, "param_digest")
            supersedes = _string_list_value(
                receipt, values, "superseded_request_ids"
            )
            for reference in supersedes:
                if reference not in existing_request_ids:
                    raise _error(
                        receipt,
                        "superseded_request_ids",
                        f"duplicate-delivery reference {reference!r} does not exist",
                        code="E_REF",
                    )
            if rank_in_batch >= len(batch.request_ids):
                raise _error(
                    receipt,
                    "rank_in_batch",
                    (
                        f"rank {rank_in_batch} is outside batch size "
                        f"{len(batch.request_ids)}"
                    ),
                )
            prior_rank = rank_seen.setdefault(batch_id, {}).get(rank_in_batch)
            if prior_rank is not None:
                raise _error(
                    receipt,
                    "rank_in_batch",
                    (
                        f"duplicate rank {rank_in_batch}; first used at line "
                        f"{prior_rank.line_number}"
                    ),
                )
            rank_seen[batch_id][rank_in_batch] = receipt
            receipt_key = (batch_id, request_id)
            prior_receipt = receipt_seen.get(receipt_key)
            if prior_receipt is not None:
                raise _error(
                    receipt,
                    "request_id",
                    (
                        f"duplicate receipt for batch/request; first at line "
                        f"{prior_receipt.line_number}"
                    ),
                )
            receipt_seen[receipt_key] = receipt
            normalized.append(
                NormalizedReceipt(
                    source_format=format_version,
                    source_line=receipt.line_number,
                    source_object=receipt.source,
                    batch_id=batch_id,
                    request_id=request_id,
                    output_digest=digest,
                    worker_id=worker_id,
                    param_version_seq=param_version_seq,
                    latency_ms=latency_ms,
                    rank_in_batch=rank_in_batch,
                    superseded_request_ids=tuple(supersedes),
                    param_digest=param_digest,
                    filled_fields=tuple(sorted(filled_fields)),
                    fill_basis=fill_basis,
                )
            )

    by_batch: dict[str, list[NormalizedReceipt]] = {}
    for receipt in normalized:
        by_batch.setdefault(receipt.batch_id, []).append(receipt)
    ledger_batches: list[dict[str, Any]] = []
    for batch_id in sorted(by_batch, key=lambda value: _batch_sort_key(plan_by_id[value])):
        batch = plan_by_id[batch_id]
        rows = sorted(
            by_batch[batch_id], key=lambda item: (item.rank_in_batch, item.request_id)
        )
        versions = {row.param_version_seq for row in rows}
        digests = {row.param_digest for row in rows}
        workers = {row.worker_id for row in rows}
        if len(versions) != 1 or len(digests) != 1 or len(workers) != 1:
            first = rows[0]
            raise DiagnosticError(
                code="E_FIELD",
                message="receipt rows in a batch disagree on parameter attribution",
                object=first.source_object,
                field="param_version_seq/param_digest/worker_id",
                line=first.source_line,
            )
        outputs = {row.request_id: row.output_digest for row in rows}
        attribution_digest = batch_output_digest(
            outputs,
            param_version_seq=rows[0].param_version_seq,
            param_digest=rows[0].param_digest,
        )
        ledger_batches.append(
            {
                "batch_id": batch.batch_id,
                "worker_id": batch.worker_id,
                "batch_seq": batch.batch_seq,
                "request_ids": list(batch.request_ids),
                "arrival_seqs": list(batch.arrival_seqs),
                "param_version_seq": rows[0].param_version_seq,
                "param_digest": rows[0].param_digest,
                "source_formats": sorted({row.source_format for row in rows}),
                "attribution_digest": attribution_digest,
                "receipts": [
                    row.to_dict()
                    for row in sorted(
                        rows, key=lambda item: (item.request_id, item.source_line)
                    )
                ],
            }
        )
    content = {"schema_version": 1, "batches": ledger_batches}
    return NormalizedLedger(
        schema_version=1,
        batches=tuple(ledger_batches),
        content_digest=sha256_canonical(content),
    )


def _string_value(
    receipt: RawReceipt, values: dict[str, Any], field: str
) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value:
        raise _error(receipt, field, "expected a non-empty string")
    return value


def _int_value(
    receipt: RawReceipt,
    values: dict[str, Any],
    field: str,
    *,
    minimum: int,
) -> int:
    value = values.get(field)
    if not is_int(value) or value < minimum:
        raise _error(receipt, field, f"expected an integer >= {minimum}")
    return value


def _digest_value(
    receipt: RawReceipt, values: dict[str, Any], field: str
) -> str:
    value = values.get(field)
    if not isinstance(value, str) or HEX_64.fullmatch(value) is None:
        raise _error(receipt, field, "expected a lowercase SHA-256 digest")
    return value


def _string_list_value(
    receipt: RawReceipt, values: dict[str, Any], field: str
) -> list[str]:
    value = values.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise _error(receipt, field, "expected an array of non-empty strings")
    if len(set(value)) != len(value):
        raise _error(receipt, field, "duplicate-delivery references must be unique")
    return list(value)


def _batch_sort_key(batch: BatchPlan) -> tuple[str, int, str]:
    return (batch.worker_id, batch.batch_seq, batch.batch_id)


def normalized_ledger_json(ledger: NormalizedLedger) -> str:
    return canonical_json(ledger.to_dict())


def _validate_encoded_values(receipt: RawReceipt) -> None:
    _validate_shape(receipt)
    _require_string(receipt, "batch_id")
    _require_string(receipt, "request_id")
    _require_digest(receipt, "output_digest")
    if receipt.format_version in {"r2", "r3"}:
        _require_string(receipt, "worker_id")
        _require_int(receipt, "param_version_seq", minimum=1)
        _require_int(receipt, "latency_ms", minimum=0)
    if receipt.format_version == "r3":
        _require_int(receipt, "rank_in_batch", minimum=0)
        _require_digest(receipt, "param_digest")
        _string_list_value(receipt, receipt.value, "superseded_request_ids")


def encode_receipt(
    value: dict[str, Any],
    format_version: str,
    *,
    object_name: str = "receipt",
) -> dict[str, Any]:
    if format_version not in FORMAT_FIELDS:
        raise DiagnosticError(
            code="E_FIELD",
            message=f"unsupported receipt format {format_version!r}",
            object=object_name,
            field="format_version",
        )
    encoded = {
        field: value[field]
        for field in sorted(FORMAT_FIELDS[format_version])
        if field in value
    }
    raw = RawReceipt(
        format_version=format_version,
        line_number=1,
        source=object_name,
        value=encoded,
    )
    _validate_encoded_values(raw)
    return encoded


def encode_receipts(
    values: Iterable[dict[str, Any]],
    format_version: str,
    *,
    object_name: str = "receipt",
) -> list[dict[str, Any]]:
    return [
        encode_receipt(value, format_version, object_name=object_name)
        for value in values
    ]


def write_receipt_file(
    path: str | Path,
    values: Iterable[dict[str, Any]],
    format_version: str,
) -> None:
    encoded = encode_receipts(values, format_version, object_name=str(path))
    text = "".join(canonical_json(row) + "\n" for row in encoded)
    try:
        atomic_write_bytes(path, text.encode("utf-8"))
    except OSError as exc:
        raise DiagnosticError(
            code="E_FIELD",
            message=f"cannot write receipt file: {exc.strerror or exc}",
            object=str(path),
            field="<file>",
        ) from exc
