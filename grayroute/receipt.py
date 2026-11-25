"""Codec and normalization for the three historical receipt formats.

Format evolution:
  r1: batch_id, request_id, output_summary
  r2: + param_version_seq, elapsed_ms, worker_id
  r3: + rank_in_batch, superseded_ref, param_summary_snapshot

Decoding validates each format's fields and raises :class:`FieldError`
(``E_FIELD``) locating the first illegal receipt and field.  Normalization
fills missing historical fields, records the filled set plus an explicit fill
basis, and produces a canonical internal representation that is byte-identical
when the same receipt is decoded repeatedly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes, content_hash
from .constants import SUPPORTED_RECEIPT_FORMATS
from .errors import FieldError


# fields present at each historical format level
FIELDS_BY_FORMAT: dict[int, tuple[str, ...]] = {
    1: ("batch_id", "request_id", "output_summary"),
    2: ("batch_id", "request_id", "output_summary",
        "param_version_seq", "elapsed_ms", "worker_id"),
    3: ("batch_id", "request_id", "output_summary",
        "param_version_seq", "elapsed_ms", "worker_id",
        "rank_in_batch", "superseded_ref", "param_summary_snapshot"),
}

PARAM_SUMMARY_LENGTH = 16  # sha256 hex prefix used for snapshots


@dataclass(frozen=True)
class BatchContext:
    """Authoritative batch facts used to fill historically missing fields."""
    batch_id: str
    worker_id: str
    param_version_seq: int
    param_summary: str
    rank_by_request: dict[str, int]
    request_ids: tuple[str, ...]
    output_summary_by_request: dict[str, str]


@dataclass(frozen=True)
class Receipt:
    format_version: int
    fields: dict[str, Any]


@dataclass(frozen=True)
class NormalizedReceipt:
    format_version: int            # source format
    batch_id: str
    request_id: str
    output_summary: str
    param_version_seq: int
    elapsed_ms: int
    worker_id: str
    rank_in_batch: int
    superseded_ref: str | None
    param_summary_snapshot: str
    filled_fields: tuple[str, ...]
    fill_basis: dict[str, str]

    def canonical(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def content_summary(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "batch_id": self.batch_id,
            "request_id": self.request_id,
            "output_summary": self.output_summary,
            "param_version_seq": self.param_version_seq,
            "elapsed_ms": self.elapsed_ms,
            "worker_id": self.worker_id,
            "rank_in_batch": self.rank_in_batch,
            "superseded_ref": self.superseded_ref,
            "param_summary_snapshot": self.param_summary_snapshot,
            "filled_fields": list(self.filled_fields),
            "fill_basis": dict(sorted(self.fill_basis.items())),
        }


def _err(receipt_id: str, field_name: str, message: str) -> FieldError:
    return FieldError(message, object_id=receipt_id, field=field_name)


def _base_id(raw: dict[str, Any]) -> str:
    return str(raw.get("request_id", raw.get("batch_id", "unknown")))


def decode_receipt(raw: dict[str, Any], context: BatchContext,
                   known_requests: set[str] | None = None) -> Receipt:
    """Validate a raw receipt against its declared format version."""
    rid = _base_id(raw)
    if "format_version" not in raw:
        raise _err(rid, "format_version", "missing format_version")
    fv = raw["format_version"]
    if isinstance(fv, bool) or fv not in SUPPORTED_RECEIPT_FORMATS:
        raise _err(rid, "format_version",
                   "format_version must be one of 1,2,3")
    allowed = set(FIELDS_BY_FORMAT[fv]) | {"format_version"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _err(rid, unknown[0], f"field not allowed in r{fv}")
    for name in FIELDS_BY_FORMAT[fv]:
        if name not in raw:
            raise _err(rid, name, f"missing field '{name}' in r{fv}")

    def check_str(name: str) -> str:
        value = raw[name]
        if not isinstance(value, str):
            raise _err(rid, name, f"{name} must be a string")
        return value

    batch_id = check_str("batch_id")
    request_id = check_str("request_id")
    output_summary = check_str("output_summary")
    if batch_id != context.batch_id:
        raise _err(rid, "batch_id", "receipt batch_id does not match batch")
    if request_id not in context.rank_by_request:
        raise _err(rid, "request_id", "request not part of batch")
    if output_summary != context.output_summary_by_request.get(request_id):
        raise _err(rid, "output_summary",
                   "output summary disagrees with deterministic backend")

    if fv >= 2:
        elapsed = raw["elapsed_ms"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, int) or elapsed < 0:
            raise _err(rid, "elapsed_ms", "elapsed_ms must be a non-negative int")
        if not isinstance(raw["worker_id"], str):
            raise _err(rid, "worker_id", "worker_id must be a string")
        if raw["worker_id"] != context.worker_id:
            raise _err(rid, "worker_id", "worker_id does not match batch worker")
        pv = raw["param_version_seq"]
        if isinstance(pv, bool) or not isinstance(pv, int) or pv <= 0:
            raise _err(rid, "param_version_seq",
                       "param_version_seq must be a positive int")

    if fv >= 3:
        rank = raw["rank_in_batch"]
        size = len(context.request_ids)
        if isinstance(rank, bool) or not isinstance(rank, int) \
                or not (0 <= rank < size):
            raise _err(rid, "rank_in_batch",
                       f"rank must be an int in [0,{size})")
        if rank != context.rank_by_request[request_id]:
            raise _err(rid, "rank_in_batch",
                       "rank does not match stable batch ordering")
        ref = raw["superseded_ref"]
        if ref is not None:
            if not isinstance(ref, str):
                raise _err(rid, "superseded_ref",
                           "superseded_ref must be a string or null")
            pool = known_requests if known_requests is not None \
                else set(context.request_ids)
            if ref not in pool:
                raise _err(rid, "superseded_ref",
                           "superseded_ref points to an unknown request")
            if ref == request_id:
                raise _err(rid, "superseded_ref",
                           "superseded_ref cannot reference itself")
        snap = raw["param_summary_snapshot"]
        if not isinstance(snap, str):
            raise _err(rid, "param_summary_snapshot", "must be a string")
        if len(snap) != PARAM_SUMMARY_LENGTH:
            raise _err(rid, "param_summary_snapshot",
                       f"length must be {PARAM_SUMMARY_LENGTH} hex chars")
        if snap != context.param_summary[:PARAM_SUMMARY_LENGTH]:
            raise _err(rid, "param_summary_snapshot",
                       "snapshot does not match frozen parameter summary")
    return Receipt(format_version=fv, fields=dict(raw))


def normalize(receipt: Receipt, context: BatchContext,
              known_requests: set[str] | None = None) -> NormalizedReceipt:
    """Decode-then-fill a raw receipt into the unified internal record."""
    decoded = decode_receipt(receipt.fields, context, known_requests)
    fv = decoded.format_version
    raw = decoded.fields
    rid = raw["request_id"]
    filled: list[str] = []
    basis: dict[str, str] = {}

    def fill(name: str, value: Any, why: str):
        filled.append(name)
        basis[name] = why
        return value

    pv = raw.get("param_version_seq")
    if fv < 2:
        pv = fill("param_version_seq", context.param_version_seq,
                  f"r{fv} omitted it; taken from frozen batch parameter version")
    elapsed = raw.get("elapsed_ms")
    if fv < 2:
        elapsed = fill("elapsed_ms", 0,
                       "r1 has no timing; neutral zero per normalization policy")
    worker = raw.get("worker_id")
    if fv < 2:
        worker = fill("worker_id", context.worker_id,
                      "r1 omitted it; taken from batch worker binding")
    rank = raw.get("rank_in_batch")
    if fv < 3:
        rank = fill("rank_in_batch", context.rank_by_request[rid],
                    "derived from stable arrival ordering within the batch")
    ref = raw.get("superseded_ref")
    if fv < 3:
        ref = fill("superseded_ref", None,
                   "no duplicate-delivery info before r3; normalized to null")
    snap = raw.get("param_summary_snapshot")
    if fv < 3:
        snap = fill("param_summary_snapshot",
                    context.param_summary[:PARAM_SUMMARY_LENGTH],
                    "copied from frozen batch parameter summary")
    return NormalizedReceipt(
        format_version=fv,
        batch_id=raw["batch_id"],
        request_id=rid,
        output_summary=raw["output_summary"],
        param_version_seq=pv,
        elapsed_ms=elapsed,
        worker_id=worker,
        rank_in_batch=rank,
        superseded_ref=ref,
        param_summary_snapshot=snap,
        filled_fields=tuple(filled),
        fill_basis=basis,
    )


def normalize_raw(raw: dict[str, Any], context: BatchContext,
                  known_requests: set[str] | None = None) -> NormalizedReceipt:
    return normalize(Receipt(raw.get("format_version", 0), dict(raw)),
                     context, known_requests)


# --------------------------------------------------------------------------- #
# encoders (used by the gateway when producing new receipts)
# --------------------------------------------------------------------------- #
def encode_receipt(format_version: int, context: BatchContext, request_id: str,
                   output_summary: str, elapsed_ms: int,
                   superseded_ref: str | None = None) -> dict[str, Any]:
    if format_version not in SUPPORTED_RECEIPT_FORMATS:
        raise _err(request_id, "format_version", "unsupported target format")
    base = {
        "format_version": format_version,
        "batch_id": context.batch_id,
        "request_id": request_id,
        "output_summary": output_summary,
    }
    if format_version >= 2:
        base.update({
            "param_version_seq": context.param_version_seq,
            "elapsed_ms": elapsed_ms,
            "worker_id": context.worker_id,
        })
    if format_version >= 3:
        base.update({
            "rank_in_batch": context.rank_by_request[request_id],
            "superseded_ref": superseded_ref,
            "param_summary_snapshot":
                context.param_summary[:PARAM_SUMMARY_LENGTH],
        })
    return base
