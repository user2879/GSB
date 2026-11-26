from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json, digest_json
from .errors import FormatError

FORMAT_VERSIONS = {"q1": 1, "q2": 2, "q3": 3}
COST_VECTOR_WIDTH = 5
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")

MIN_WIDTHS = {
    "receipt_id": 8,
    "plan_id": 16,
    "shard_id": 3,
    "executor_id": 1,
    "lease_id": 1,
}

DEFAULT_EXECUTOR_ID = "unknown-executor"
DEFAULT_COST_VECTOR = (0, 0, 0, 0, 0)
DEFAULT_MONO_NS = 0
DEFAULT_LEASE_ID = "none"
DEFAULT_LEASE_STATE = "unknown"


def _format_error(message: str, field: str) -> FormatError:
    return FormatError(message, field=field)


def _require_string(payload: Mapping[str, Any], field: str, *, minimum: int | None = None) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise _format_error("field must be a string", field)
    if minimum is not None and len(value) < minimum:
        raise _format_error(f"field width is below minimum {minimum}", field)
    return value


def _require_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _format_error("field must be an integer", field)
    if minimum is not None and value < minimum:
        raise _format_error(f"field must be >= {minimum}", field)
    if maximum is not None and value > maximum:
        raise _format_error(f"field must be <= {maximum}", field)
    return value


def _validate_digest(payload: Mapping[str, Any], field: str) -> str:
    value = _require_string(payload, field, minimum=64)
    if len(value) != 64 or _DIGEST_RE.fullmatch(value) is None:
        raise _format_error("field must be a 64-character lowercase hexadecimal digest", field)
    return value


def _validate_cost_vector(payload: Mapping[str, Any], field: str) -> tuple[int, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or len(value) != COST_VECTOR_WIDTH:
        raise _format_error(f"field must be an integer vector of width {COST_VECTOR_WIDTH}", field)
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise _format_error("cost vector entries must be non-negative integers", f"{field}[{index}]")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    plan_id: str
    shard_id: str
    shard_index: int
    total_shards: int
    status: str
    row_count: int
    result_digest: str
    executor_id: str
    cost_vector: tuple[int, ...]
    mono_ns: int
    lease_id: str
    lease_state: str
    result: dict[str, Any] | None = None

    def core_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "plan_id": self.plan_id,
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "total_shards": self.total_shards,
            "status": self.status,
            "row_count": self.row_count,
            "result_digest": self.result_digest,
            "executor_id": self.executor_id,
            "cost_vector": list(self.cost_vector),
            "mono_ns": self.mono_ns,
            "lease_id": self.lease_id,
            "lease_state": self.lease_state,
            "result": self.result,
        }

    @property
    def content_digest(self) -> str:
        return digest_json(self.core_dict())

    @property
    def adoption_digest(self) -> str:
        return digest_json(
            {
                "receipt_id": self.receipt_id,
                "plan_id": self.plan_id,
                "shard_id": self.shard_id,
                "shard_index": self.shard_index,
                "total_shards": self.total_shards,
                "status": self.status,
                "row_count": self.row_count,
                "result_digest": self.result_digest,
            }
        )


@dataclass(frozen=True)
class NormalizedReceipt:
    receipt: Receipt
    source_format: str
    source_version: int
    backfilled_fields: tuple[str, ...]
    backfill_basis: dict[str, str]

    @property
    def content_digest(self) -> str:
        return self.receipt.content_digest

    @property
    def adoption_digest(self) -> str:
        return self.receipt.adoption_digest

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "source_version": self.source_version,
            "backfilled_fields": list(self.backfilled_fields),
            "backfill_basis": self.backfill_basis,
            "content_digest": self.content_digest,
            "adoption_digest": self.adoption_digest,
            "receipt": self.receipt.core_dict(),
        }


def normalize_receipt(payload: Mapping[str, Any]) -> NormalizedReceipt:
    format_version = payload.get("format")
    if format_version not in FORMAT_VERSIONS:
        raise _format_error("unknown receipt format version", "format")
    version = FORMAT_VERSIONS[format_version]

    receipt_id = _require_string(payload, "receipt_id", minimum=MIN_WIDTHS["receipt_id"])
    plan_id = _require_string(payload, "plan_id", minimum=MIN_WIDTHS["plan_id"])
    shard_id = _require_string(payload, "shard_id", minimum=MIN_WIDTHS["shard_id"])
    shard_index = _require_int(payload, "shard_index", minimum=0)
    total_shards = _require_int(payload, "total_shards", minimum=1)
    if shard_index >= total_shards:
        raise _format_error("shard_index is outside [0,total_shards)", "shard_index")

    status = _require_string(payload, "status")
    if status not in ("ok", "error"):
        raise _format_error("status must be 'ok' or 'error'", "status")
    row_count = _require_int(payload, "row_count", minimum=0)
    result_digest = _validate_digest(payload, "result_digest")

    backfilled_fields: list[str] = []
    backfill_basis: dict[str, str] = {}

    if version >= 2:
        executor_id = _require_string(payload, "executor_id", minimum=MIN_WIDTHS["executor_id"])
        cost_vector = _validate_cost_vector(payload, "cost_vector")
    else:
        executor_id = DEFAULT_EXECUTOR_ID
        cost_vector = DEFAULT_COST_VECTOR
        backfilled_fields.append("executor_id")
        backfill_basis["executor_id"] = "missing_in_q1:use_unknown-executor"
        backfilled_fields.append("cost_vector")
        backfill_basis["cost_vector"] = "missing_in_q1:use_[0,0,0,0,0]"

    if version >= 3:
        mono_ns = _require_int(payload, "mono_ns", minimum=0)
        lease_id = _require_string(payload, "lease_id", minimum=MIN_WIDTHS["lease_id"])
        lease_state = _require_string(payload, "lease_state")
        result = payload.get("result")
        if result is not None and not isinstance(result, dict):
            raise _format_error("result must be an object when present", "result")
    else:
        mono_ns = DEFAULT_MONO_NS
        lease_id = DEFAULT_LEASE_ID
        lease_state = DEFAULT_LEASE_STATE
        result = None
        backfilled_fields.append("mono_ns")
        backfill_basis["mono_ns"] = f"missing_in_{format_version}:use_0"
        backfilled_fields.append("lease_id")
        backfill_basis["lease_id"] = f"missing_in_{format_version}:use_none"
        backfilled_fields.append("lease_state")
        backfill_basis["lease_state"] = f"missing_in_{format_version}:use_unknown"
        backfilled_fields.append("result")
        backfill_basis["result"] = f"missing_in_{format_version}:use_null"

    receipt = Receipt(
        receipt_id=receipt_id,
        plan_id=plan_id,
        shard_id=shard_id,
        shard_index=shard_index,
        total_shards=total_shards,
        status=status,
        row_count=row_count,
        result_digest=result_digest,
        executor_id=executor_id,
        cost_vector=cost_vector,
        mono_ns=mono_ns,
        lease_id=lease_id,
        lease_state=lease_state,
        result=result,
    )
    backfilled_fields.sort()
    return NormalizedReceipt(
        receipt=receipt,
        source_format=format_version,
        source_version=version,
        backfilled_fields=tuple(backfilled_fields),
        backfill_basis=backfill_basis,
    )


def encode_receipt(receipt: Receipt, format_version: str) -> dict[str, Any]:
    if format_version not in FORMAT_VERSIONS:
        raise _format_error("unknown receipt format version", "format")
    version = FORMAT_VERSIONS[format_version]
    payload: dict[str, Any] = {
        "format": format_version,
        "receipt_id": receipt.receipt_id,
        "plan_id": receipt.plan_id,
        "shard_id": receipt.shard_id,
        "shard_index": receipt.shard_index,
        "total_shards": receipt.total_shards,
        "status": receipt.status,
        "row_count": receipt.row_count,
        "result_digest": receipt.result_digest,
    }
    if version >= 2:
        payload["executor_id"] = receipt.executor_id
        payload["cost_vector"] = list(receipt.cost_vector)
    if version >= 3:
        payload["mono_ns"] = receipt.mono_ns
        payload["lease_id"] = receipt.lease_id
        payload["lease_state"] = receipt.lease_state
        payload["result"] = receipt.result
    return payload


def decode_receipt(payload: Mapping[str, Any]) -> NormalizedReceipt:
    return normalize_receipt(payload)


def make_receipt(
    *,
    receipt_id: str,
    plan_id: str,
    shard_id: str,
    shard_index: int,
    total_shards: int,
    status: str,
    row_count: int,
    result_digest: str,
    executor_id: str,
    cost_vector: tuple[int, ...] | list[int],
    mono_ns: int,
    lease_id: str,
    lease_state: str,
    result: dict[str, Any] | None = None,
) -> Receipt:
    receipt = Receipt(
        receipt_id=receipt_id,
        plan_id=plan_id,
        shard_id=shard_id,
        shard_index=shard_index,
        total_shards=total_shards,
        status=status,
        row_count=row_count,
        result_digest=result_digest,
        executor_id=executor_id,
        cost_vector=tuple(cost_vector),
        mono_ns=mono_ns,
        lease_id=lease_id,
        lease_state=lease_state,
        result=result,
    )
    normalize_receipt(encode_receipt(receipt, "q3"))
    return receipt


def write_receipt_bundle(directory: str | Path, receipt: Receipt, *, prefix: str | None = None) -> list[Path]:
    bundle_dir = Path(directory)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or receipt.shard_id.replace("/", "_")
    written: list[Path] = []
    for format_version in ("q1", "q2", "q3"):
        path = bundle_dir / f"{stem}.{format_version}.json"
        payload = encode_receipt(receipt, format_version)
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        written.append(path)
    return written


def load_receipt_file(path: str | Path) -> NormalizedReceipt:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _format_error(f"invalid JSON: {exc.msg}", f"line:{exc.lineno}:column:{exc.colno}") from exc
    except OSError as exc:
        raise _format_error(f"cannot read receipt: {exc}", str(path)) from exc
    return normalize_receipt(raw)


def normalize_bundle(bundle_dir: str | Path, stem: str) -> dict[str, NormalizedReceipt]:
    results: dict[str, NormalizedReceipt] = {}
    for format_version in ("q1", "q2", "q3"):
        results[format_version] = load_receipt_file(Path(bundle_dir) / f"{stem}.{format_version}.json")
    return results
