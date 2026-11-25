from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import FieldError, ReferenceError
from .models import Batch, Plan
from .spec import is_plain_int


DIGEST_LENGTH = 64
FORMATS = ("r1", "r2", "r3")


@dataclass(frozen=True)
class BatchReceiptContext:
    batch_id: str
    worker_id: str
    request_ids: tuple[str, ...]
    rank_by_request: dict[str, int]
    param_version: int
    param_digest: str

    @property
    def size(self) -> int:
        return len(self.request_ids)


@dataclass(frozen=True)
class ReceiptContext:
    known_request_ids: frozenset[str]
    batches: dict[str, BatchReceiptContext] = field(default_factory=dict)
    known_worker_ids: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_plan(
        cls,
        plan: Plan,
        worker_param_versions: dict[str, int],
        worker_param_digests: dict[str, str],
        known_request_ids: Iterable[str] | None = None,
    ) -> "ReceiptContext":
        batches: dict[str, BatchReceiptContext] = {}
        request_ids: set[str] = set(known_request_ids or [])
        for batch in plan.batches:
            request_ids.update(batch.request_ids)
            batches[batch.batch_id] = BatchReceiptContext(
                batch_id=batch.batch_id,
                worker_id=batch.worker_id,
                request_ids=batch.request_ids,
                rank_by_request=dict(batch.stable_ranks),
                param_version=int(worker_param_versions[batch.worker_id]),
                param_digest=worker_param_digests[batch.worker_id],
            )
        return cls(
            known_request_ids=frozenset(request_ids),
            batches=batches,
            known_worker_ids=frozenset(worker_param_versions),
        )


def normalize_digest(value: Any, *, line: int | None = None, field_name: str = "digest") -> str:
    if not isinstance(value, str):
        raise FieldError(f"{field_name} 必须是 sha256 摘要字符串", line=line, field=field_name)
    raw = value[7:] if value.startswith("sha256:") else value
    if len(raw) != DIGEST_LENGTH or any(ch not in "0123456789abcdef" for ch in raw.lower()):
        raise FieldError(
            f"{field_name} 长度或字符不自洽，应为 {DIGEST_LENGTH} 位十六进制摘要",
            line=line,
            field=field_name,
            expected_length=DIGEST_LENGTH,
            actual_length=len(raw),
        )
    return raw.lower()


def _parse_int_field(value: Any, field_name: str, line: int | None, *, minimum: int) -> int:
    if not is_plain_int(value) or int(value) < minimum:
        raise FieldError(
            f"{field_name} 必须是不小于 {minimum} 的整数",
            line=line,
            field=field_name,
            value=value,
        )
    return int(value)


def decode_receipt_line(line: str, line_number: int = 1) -> dict[str, Any]:
    text = line.strip()
    if not text or text.startswith("#"):
        return {}
    parts = text.split("|")
    format_tag = parts[0].lower()
    expected_lengths = {"r1": 4, "r2": 7, "r3": 10}
    if format_tag not in expected_lengths:
        raise FieldError("未知回执格式标记", line=line_number, field="format", value=parts[0])
    expected = expected_lengths[format_tag]
    if len(parts) != expected:
        raise FieldError(
            f"{format_tag} 字段长度不自洽：期望 {expected} 段，实际 {len(parts)} 段",
            line=line_number,
            field="record_length",
            expected=expected,
            actual=len(parts),
        )
    if format_tag == "r1":
        return {
            "format": "r1",
            "batch_id": parts[1],
            "request_id": parts[2],
            "output_digest": parts[3],
            "_line": line_number,
        }
    if format_tag == "r2":
        return {
            "format": "r2",
            "batch_id": parts[1],
            "request_id": parts[2],
            "output_digest": parts[3],
            "param_version": parts[4],
            "elapsed_ms": parts[5],
            "worker_id": parts[6],
            "_line": line_number,
        }
    return {
        "format": "r3",
        "batch_id": parts[1],
        "request_id": parts[2],
        "output_digest": parts[3],
        "param_version": parts[4],
        "elapsed_ms": parts[5],
        "worker_id": parts[6],
        "rank": parts[7],
        "supersedes_ref": None if parts[8] == "-" else parts[8],
        "param_digest": parts[9],
        "_line": line_number,
    }


def _coerce_numbers(raw: dict[str, Any], line: int | None) -> dict[str, Any]:
    result = dict(raw)
    for key in ("param_version", "elapsed_ms", "rank"):
        if key not in result or result[key] is None:
            continue
        value = result[key]
        if isinstance(value, str) and value.lstrip("-").isdigit():
            result[key] = int(value)
        elif key == "param_version" and isinstance(value, str):
            raise FieldError("param_version 必须是整数序号", line=line, field=key, value=value)
    return result


def normalize_receipt(
    raw: dict[str, Any] | str,
    context: ReceiptContext | None = None,
    *,
    line_number: int | None = None,
) -> dict[str, Any]:
    if isinstance(raw, str):
        decoded = decode_receipt_line(raw, line_number or 1)
        if not decoded:
            return {}
        raw = decoded
    line = line_number or raw.get("_line")
    raw = _coerce_numbers(raw, line)
    version = raw.get("format") or raw.get("format_version") or raw.get("v")
    if version not in FORMATS:
        raise FieldError("回执 format 必须是 r1/r2/r3", line=line, field="format", value=version)

    batch_id = raw.get("batch_id")
    request_id = raw.get("request_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise FieldError("batch_id 必须是非空字符串", line=line, field="batch_id")
    if not isinstance(request_id, str) or not request_id:
        raise FieldError("request_id 必须是非空字符串", line=line, field="request_id")
    output_digest = normalize_digest(raw.get("output_digest"), line=line, field_name="output_digest")

    batch_context = context.batches.get(batch_id) if context else None
    if context and batch_context is None:
        raise ReferenceError("回执引用了不存在的批次", line=line, field="batch_id", batch_id=batch_id)
    if context and request_id not in context.known_request_ids:
        raise ReferenceError("回执引用了不存在的请求", line=line, field="request_id", request_id=request_id)
    if batch_context and request_id not in batch_context.request_ids:
        raise ReferenceError(
            "request_id 不属于回执声明的批次",
            line=line,
            field="request_id",
            batch_id=batch_id,
            request_id=request_id,
        )

    backfilled: list[str] = []
    basis: dict[str, str] = {}

    worker_id = raw.get("worker_id")
    if version == "r1":
        if batch_context is not None:
            worker_id = batch_context.worker_id
            backfilled.append("worker_id")
            basis["worker_id"] = "batch-context:worker_assignment"
        param_version = batch_context.param_version if batch_context else 0
        backfilled.append("param_version")
        basis["param_version"] = "batch-context:frozen_param_version"
        elapsed_ms = 0
        backfilled.append("elapsed_ms")
        basis["elapsed_ms"] = "r1-default:no_timing_field"
        rank = batch_context.rank_by_request.get(request_id, 0) if batch_context else 0
        backfilled.append("rank")
        basis["rank"] = "planner:stable_arrival_rank"
        supersedes_ref = None
        backfilled.append("supersedes_ref")
        basis["supersedes_ref"] = "r1-default:no_duplicate_claim"
        param_digest = batch_context.param_digest if batch_context else "0" * 64
        backfilled.append("param_digest")
        basis["param_digest"] = "batch-context:frozen_param_digest"
    elif version == "r2":
        param_version = _parse_int_field(raw.get("param_version"), "param_version", line, minimum=1)
        elapsed_ms = _parse_int_field(raw.get("elapsed_ms"), "elapsed_ms", line, minimum=0)
        if not isinstance(worker_id, str) or not worker_id:
            raise FieldError("worker_id 必须是非空字符串", line=line, field="worker_id")
        rank = batch_context.rank_by_request.get(request_id, 0) if batch_context else 0
        backfilled.append("rank")
        basis["rank"] = "planner:stable_arrival_rank"
        supersedes_ref = None
        backfilled.append("supersedes_ref")
        basis["supersedes_ref"] = "r2-default:no_duplicate_claim"
        if batch_context is not None:
            param_digest = batch_context.param_digest
            backfilled.append("param_digest")
            basis["param_digest"] = "batch-context:frozen_param_digest_for_param_version"
        else:
            param_digest = "0" * 64
            backfilled.append("param_digest")
            basis["param_digest"] = "missing-context:zero_placeholder_not_trusted"
    else:
        param_version = _parse_int_field(raw.get("param_version"), "param_version", line, minimum=1)
        elapsed_ms = _parse_int_field(raw.get("elapsed_ms"), "elapsed_ms", line, minimum=0)
        if not isinstance(worker_id, str) or not worker_id:
            raise FieldError("worker_id 必须是非空字符串", line=line, field="worker_id")
        rank = _parse_int_field(raw.get("rank"), "rank", line, minimum=0)
        if batch_context is not None and rank >= batch_context.size:
            raise FieldError(
                f"rank {rank} 越界，批次大小为 {batch_context.size}",
                line=line,
                field="rank",
                rank=rank,
                batch_size=batch_context.size,
            )
        supersedes_ref = raw.get("supersedes_ref")
        if supersedes_ref in ("", "-"):
            supersedes_ref = None
        if supersedes_ref is not None and not isinstance(supersedes_ref, str):
            raise FieldError("supersedes_ref 必须是请求标识、null 或 -", line=line, field="supersedes_ref")
        param_digest = normalize_digest(raw.get("param_digest"), line=line, field_name="param_digest")

    if context and worker_id and context.known_worker_ids and worker_id not in context.known_worker_ids:
        raise ReferenceError("回执引用了不存在的工作进程", line=line, field="worker_id", worker_id=worker_id)

    if supersedes_ref is not None and context is not None and supersedes_ref not in context.known_request_ids:
        raise ReferenceError(
            "被覆盖的重复投递引用指向不存在的请求",
            line=line,
            field="supersedes_ref",
            supersedes_ref=supersedes_ref,
        )

    normalized = {
        "source_format_version": version,
        "format_version": "normalized-v1",
        "batch_id": batch_id,
        "request_id": request_id,
        "output_digest": output_digest,
        "param_version": param_version,
        "elapsed_ms": elapsed_ms,
        "worker_id": worker_id,
        "rank": rank,
        "supersedes_ref": supersedes_ref,
        "param_digest": param_digest,
        "backfilled_fields": sorted(set(backfilled)),
        "backfill_basis": dict(sorted(basis.items())),
        "source_line": line,
    }
    return normalized


def normalize_receipts(
    records: Iterable[dict[str, Any] | str],
    context: ReceiptContext | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        line_number = index if isinstance(record, str) else record.get("_line", index) if isinstance(record, dict) else index
        item = normalize_receipt(record, context, line_number=line_number)
        if item:
            normalized.append(item)

    by_batch: dict[str, list[dict[str, Any]]] = {}
    for item in normalized:
        by_batch.setdefault(item["batch_id"], []).append(item)
    for batch_id, items in sorted(by_batch.items()):
        seen_ranks: dict[int, int | None] = {}
        batch_context = context.batches.get(batch_id) if context else None
        for item in sorted(items, key=lambda entry: (entry.get("source_line") or 0, entry["request_id"])):
            rank = int(item["rank"])
            if batch_context is not None and rank >= batch_context.size:
                raise FieldError("批内名次越界", line=item.get("source_line"), field="rank", batch_id=batch_id, rank=rank)
            if rank in seen_ranks:
                raise FieldError(
                    "批内名次重复",
                    line=item.get("source_line"),
                    field="rank",
                    batch_id=batch_id,
                    rank=rank,
                    first_source_line=seen_ranks[rank],
                )
            seen_ranks[rank] = item.get("source_line")
    return normalized


def encode_receipt_line(receipt: dict[str, Any], target_format: str | None = None) -> str:
    version = target_format or receipt.get("source_format_version") or "r3"
    if version not in FORMATS:
        raise FieldError("目标格式必须是 r1/r2/r3", field="target_format", value=version)
    output_digest = receipt["output_digest"]
    if version == "r1":
        return "|".join(["R1", receipt["batch_id"], receipt["request_id"], output_digest])
    if version == "r2":
        return "|".join(
            [
                "R2",
                receipt["batch_id"],
                receipt["request_id"],
                output_digest,
                str(receipt["param_version"]),
                str(receipt["elapsed_ms"]),
                receipt["worker_id"],
            ]
        )
    supersedes = receipt.get("supersedes_ref") or "-"
    return "|".join(
        [
            "R3",
            receipt["batch_id"],
            receipt["request_id"],
            output_digest,
            str(receipt["param_version"]),
            str(receipt["elapsed_ms"]),
            receipt["worker_id"],
            str(receipt["rank"]),
            supersedes,
            receipt["param_digest"],
        ]
    )


def build_receipts_from_forward(
    result: dict[str, Any],
    *,
    supersedes_by_request: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Build normalized r3 receipts for every request in a forward result."""

    supersedes_by_request = supersedes_by_request or {}
    receipts: list[dict[str, Any]] = []
    for item in sorted(result.get("per_request", []), key=lambda entry: int(entry["rank"])):
        request_id = item["request_id"]
        receipts.append(
            {
                "source_format_version": "r3",
                "format_version": "normalized-v1",
                "batch_id": result["batch_id"],
                "request_id": request_id,
                "output_digest": result["output_digest"],
                "param_version": int(result["param_version"]),
                "elapsed_ms": int(result.get("elapsed_ms", 0)),
                "worker_id": result["worker_id"],
                "rank": int(item["rank"]),
                "supersedes_ref": supersedes_by_request.get(request_id),
                "param_digest": result["param_digest"],
                "backfilled_fields": [],
                "backfill_basis": {},
                "source_line": None,
            }
        )
    return receipts


def attach_encoded_receipts(result: dict[str, Any]) -> dict[str, Any]:
    receipts = build_receipts_from_forward(result)
    result["normalized_receipts"] = receipts
    result["receipt_lines"] = [encode_receipt_line(receipt, "r3") for receipt in receipts]
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any] | str]:
    records: list[dict[str, Any] | str] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            try:
                item = json.loads(stripped)
                item["_line"] = line_number
                records.append(item)
            except json.JSONDecodeError as exc:
                raise FieldError("回执 JSON 语法损坏", line=exc.lineno, field="$") from exc
        else:
            records.append(line)
    return records
