"""r1/r2/r3 三版本双向转换与降级信息旁路。

升级路径上缺失字段以显式缺失标记 :data:`~receipt_ledger.models.MISSING`
补齐并重新计算完整性标签；降级路径上被剥离的字段整体写入旁路元数据，
旁路中保存原始引用与标签，使再升级后可逐字段还原，归属裁决保持不变。
"""

from __future__ import annotations

from dataclasses import replace

from .models import MISSING, Receipt
from .receipts import R2_FIELDS, R3_FIELDS, integrity_tag
from .util import canonical_json, short_ref

SIDECAR_KIND = "receipt-ledger-sidecar/v1"


def receipt_ref(receipt: Receipt) -> str:
    """回执稳定引用：同一物理回执的版本往返共享同一引用。"""

    identity = (
        receipt.batch_id,
        receipt.request_id,
        receipt.output_digest,
        receipt.integrity_tag,
    )
    return short_ref(identity)


def completeness_label(receipt: Receipt) -> str:
    if not receipt.missing:
        return "complete"
    return "incomplete:" + ",".join(receipt.missing)


def to_payload(receipt: Receipt, version: str) -> dict:
    """导出为指定版本的磁盘载荷，缺失位写显式标记。"""

    if version not in ("r1", "r2", "r3"):
        raise ValueError(f"未知目标版本: {version}")
    payload: dict = {
        "version": version,
        "batch_id": receipt.batch_id,
        "request_id": receipt.request_id,
        "output_digest": receipt.output_digest,
    }
    optional = {
        "param_version": receipt.param_version,
        "duration_ms": receipt.duration_ms,
        "worker_id": receipt.worker_id,
        "rank_in_batch": receipt.rank_in_batch,
        "supersedes": list(receipt.supersedes),
        "param_summary": receipt.param_summary,
    }
    include = ()
    if version == "r2":
        include = R2_FIELDS
    elif version == "r3":
        include = R2_FIELDS + R3_FIELDS
    missing_set = set(receipt.missing)
    for field_name in include:
        value = optional[field_name]
        if value is None or field_name in missing_set or (
            field_name == "supersedes" and not value
        ):
            payload[field_name] = MISSING
        else:
            payload[field_name] = list(value) if field_name == "supersedes" else value
    payload["integrity_tag"] = integrity_tag(payload)
    return payload


def upgrade(receipt: Receipt, target: str) -> Receipt:
    order = ("r1", "r2", "r3")
    if order.index(target) <= order.index(receipt.source_version):
        raise ValueError("upgrade 仅支持向更高版本转换")
    payload = to_payload(receipt, target)
    if receipt.integrity_tag:
        payload["integrity_tag"] = receipt.integrity_tag
    return _from_payload(receipt, payload, target)


def downgrade(receipt: Receipt, target: str) -> tuple[Receipt, dict]:
    order = ("r1", "r2", "r3")
    if order.index(target) >= order.index(receipt.source_version):
        raise ValueError("downgrade 仅支持向更低版本转换")
    payload = to_payload(receipt, target)
    payload["integrity_tag"] = receipt.integrity_tag
    sidecar = build_sidecar(receipt)
    downgraded = _from_payload(receipt, payload, target)
    return downgraded, sidecar


def build_sidecar(receipt: Receipt) -> dict:
    """把降级剥离的全部字段连同原始引用写入旁路元数据。"""

    return {
        "kind": SIDECAR_KIND,
        "original_version": receipt.source_version,
        "original_ref": receipt_ref(receipt),
        "original_integrity_tag": receipt.integrity_tag,
        "source_file": receipt.source_file,
        "line": receipt.line,
        "fields": {
            "param_version": receipt.param_version,
            "duration_ms": receipt.duration_ms,
            "worker_id": receipt.worker_id,
            "rank_in_batch": receipt.rank_in_batch,
            "supersedes": list(receipt.supersedes),
            "param_summary": receipt.param_summary,
        },
        "missing": list(receipt.missing),
        "sidecar_hash": short_ref(
            [
                receipt.batch_id,
                receipt.request_id,
                receipt.output_digest,
                receipt.integrity_tag,
            ]
        ),
    }


def restore(down_payload: dict, sidecar: dict) -> Receipt:
    """从降级载荷与旁路元数据逐字段还原原始高版本回执。"""

    if sidecar.get("kind") != SIDECAR_KIND:
        raise ValueError("旁路元数据类型无法识别")
    version = sidecar["original_version"]
    fields = sidecar["fields"]
    payload: dict = {
        "version": version,
        "batch_id": down_payload["batch_id"],
        "request_id": down_payload["request_id"],
        "output_digest": down_payload["output_digest"],
    }
    for field_name in R2_FIELDS + R3_FIELDS:
        value = fields[field_name]
        if value is None:
            payload[field_name] = MISSING
        else:
            payload[field_name] = value
    payload["integrity_tag"] = sidecar["original_integrity_tag"]
    restored = Receipt(
        source_version=version,
        source_file=sidecar["source_file"],
        line=sidecar["line"],
        batch_id=payload["batch_id"],
        request_id=payload["request_id"],
        output_digest=payload["output_digest"],
        param_version=fields["param_version"],
        duration_ms=fields["duration_ms"],
        worker_id=fields["worker_id"],
        rank_in_batch=fields["rank_in_batch"],
        supersedes=tuple(fields["supersedes"]),
        param_summary=fields["param_summary"],
        missing=tuple(sidecar["missing"]),
        integrity_tag=sidecar["original_integrity_tag"],
    )
    return restored


def round_trip(receipt: Receipt) -> tuple[Receipt, dict, dict]:
    """降级到 r1 再还原，返回 (还原回执, r1 载荷, 旁路)。"""

    downgraded, sidecar = downgrade(receipt, "r1")
    payload = to_payload(downgraded, "r1")
    restored = restore(payload, sidecar)
    return restored, payload, sidecar


def conversion_record(action: str, receipt: Receipt, target: str, ref: str) -> dict:
    """账本中的转换（含降级）条目。"""

    return {
        "action": action,
        "from_version": receipt.source_version,
        "to_version": target,
        "receipt_ref": ref,
        "batch_id": receipt.batch_id,
        "request_id": receipt.request_id,
        "integrity_tag": receipt.integrity_tag,
        "completeness": completeness_label(receipt),
    }


def _from_payload(original: Receipt, payload: dict, version: str) -> Receipt:
    missing = tuple(
        name
        for name in R2_FIELDS + R3_FIELDS
        if name in payload and payload[name] == MISSING
    )
    return replace(
        original,
        source_version=version,
        param_version=payload.get("param_version") if payload.get("param_version") != MISSING else None,
        duration_ms=payload.get("duration_ms") if payload.get("duration_ms") != MISSING else None,
        worker_id=payload.get("worker_id") if payload.get("worker_id") != MISSING else None,
        rank_in_batch=(
            payload.get("rank_in_batch") if payload.get("rank_in_batch") != MISSING else None
        ),
        supersedes=(
            ()
            if payload.get("supersedes") in (None, MISSING)
            else tuple(payload["supersedes"])
        ),
        param_summary=(
            payload.get("param_summary") if payload.get("param_summary") != MISSING else None
        ),
        missing=missing,
        integrity_tag=payload["integrity_tag"],
    )


def sidecar_canonical(sidecar: dict) -> str:
    return canonical_json(sidecar)
