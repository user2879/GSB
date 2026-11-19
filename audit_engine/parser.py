"""JSON 清单解析与字段校验。

非法记录产出含记录序号与 E_PARSE 的拒绝记录，解析继续，不中止整批。
"""

from __future__ import annotations

import json
from typing import Any

from .errors import E_PARSE, RecordParseError
from .models import (
    AclEntry,
    LinkDecl,
    OP_WHITELIST,
    REPRESENTATION_WHITELIST,
    Record,
    RejectedRecord,
    ROUND_NEAREST,
    ROUND_TRUNCATE,
    TruthMarker,
)
from .permission import parse_permission


def _require(obj: dict[str, Any], key: str, index: int) -> Any:
    if key not in obj:
        raise RecordParseError(f"记录缺少必填字段 {key!r}", index)
    return obj[key]


def _parse_links(raw: Any, index: int) -> tuple[LinkDecl, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RecordParseError("links 必须为数组", index)
    links: list[LinkDecl] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RecordParseError("links 条目必须为对象", index)
        link_type = item.get("link_type", "symlink")
        if link_type not in ("symlink", "hardlink"):
            raise RecordParseError(f"未知链接类型 {link_type!r}", index)
        at = item.get("at")
        target = item.get("target")
        if not isinstance(at, str) or not at.strip():
            raise RecordParseError("链接声明 at 必须为非空字符串", index)
        if not isinstance(target, str) or not target.strip():
            raise RecordParseError("链接声明 target 必须为非空字符串", index)
        steps = item.get("steps", 1)
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise RecordParseError("链接 steps 必须为正整数", index)
        links.append(LinkDecl(at=at, target=target, link_type=link_type, steps=steps))
    return tuple(links)


def _parse_acl(raw: Any, index: int, rounding: str) -> tuple[AclEntry, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise RecordParseError("acl 必须为数组", index)
    entries: list[AclEntry] = []
    for order, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RecordParseError("acl 条目必须为对象", index)
        subject = item.get("subject")
        if not isinstance(subject, str) or not subject.strip():
            raise RecordParseError("ACL 条目 subject 必须为非空字符串", index)
        access = item.get("access")
        if access not in ("allow", "deny"):
            raise RecordParseError("ACL 条目 access 只能为 allow / deny", index)
        representation = item.get("representation", REPRESENTATION_WHITELIST[0])
        if representation not in REPRESENTATION_WHITELIST:
            raise RecordParseError("ACL 权限口径标记非法", index)
        parsed = parse_permission(representation, item.get("mask"), rounding)
        inherited = bool(item.get("inherited", False))
        source_depth = item.get("source_depth", 0)
        if not isinstance(source_depth, int) or isinstance(source_depth, bool) or source_depth < 0:
            raise RecordParseError("source_depth 必须为非负整数", index)
        entries.append(
            AclEntry(
                subject=subject,
                access=access,
                mask=parsed["mask"],
                inherited=inherited,
                source_depth=source_depth,
                epsilon=float(parsed["epsilon"]),
                order=order,
            )
        )
    return tuple(entries)


def _parse_truth(raw: Any, index: int) -> TruthMarker | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RecordParseError("truth 必须为对象", index)
    for key in ("sync_marker", "raw_path_bytes", "raw_perm_value", "expected_epsilon"):
        if key not in raw:
            raise RecordParseError(f"truth 缺少字段 {key!r}", index)
    interval = raw["expected_epsilon"]
    if not isinstance(interval, list) or len(interval) != 2:
        raise RecordParseError("expected_epsilon 必须为 [lo, hi] 区间", index)
    lo, hi = interval
    if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or lo < 0 or hi < lo:
        raise RecordParseError("expected_epsilon 区间非法", index)
    return TruthMarker(
        sync_marker=str(raw["sync_marker"]),
        raw_path_bytes=str(raw["raw_path_bytes"]),
        raw_perm_value=raw["raw_perm_value"],
        expected_epsilon_lo=float(lo),
        expected_epsilon_hi=float(hi),
    )


def parse_record(obj: dict[str, Any], index: int) -> Record:
    record_id = _require(obj, "id", index)
    if not isinstance(record_id, str) or not record_id.strip():
        raise RecordParseError("id 必须为非空字符串", index)
    subject = _require(obj, "subject", index)
    if not isinstance(subject, str) or not subject.strip():
        raise RecordParseError("subject 必须为非空字符串", index)
    operation = _require(obj, "operation", index)
    if operation not in OP_WHITELIST:
        raise RecordParseError(f"operation 必须属于 {OP_WHITELIST}", index)
    target_path = _require(obj, "target_path", index)
    if not isinstance(target_path, str) or not target_path.strip():
        raise RecordParseError("target_path 必须为非空字符串", index)
    if "\x00" in target_path:
        raise RecordParseError("target_path 含 NUL 字符，路径语法非法", index)

    representation = _require(obj, "representation", index)
    if representation not in REPRESENTATION_WHITELIST:
        raise RecordParseError(
            f"representation 只能为 {REPRESENTATION_WHITELIST}", index
        )
    rounding = obj.get("rounding", ROUND_TRUNCATE)
    if rounding not in (ROUND_TRUNCATE, ROUND_NEAREST):
        raise RecordParseError("rounding 只能为 truncate / nearest", index)

    perm_value = _require(obj, "perm_value", index)
    parsed = parse_permission(representation, perm_value, rounding)

    links = _parse_links(obj.get("links"), index)
    acl = _parse_acl(obj.get("acl"), index, rounding)
    truth = _parse_truth(obj.get("truth"), index)

    return Record(
        index=index,
        record_id=record_id,
        subject=subject,
        operation=operation,
        target_path=target_path,
        representation=representation,
        perm_value=perm_value,
        rounding=rounding,
        links=links,
        acl=acl,
        truth=truth,
        extra={"parsed_permission": parsed},
    )


def parse_manifest(payload: Any) -> tuple[list[Record], list[RejectedRecord], dict[str, Any]]:
    """解析整份清单，返回 (合法记录, 拒绝记录, 文件级元数据)。"""
    if not isinstance(payload, dict):
        raise RecordParseError("清单根对象必须为 JSON 对象")
    file_id = payload.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        raise RecordParseError("清单缺少合法 file_id")
    root = payload.get("root", "/")
    if not isinstance(root, str) or not root.strip():
        raise RecordParseError("清单 root 必须为非空字符串")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise RecordParseError("清单 records 必须为数组")

    records: list[Record] = []
    rejections: list[RejectedRecord] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_records):
        if not isinstance(item, dict):
            rejections.append(RejectedRecord(index, E_PARSE, "记录必须为 JSON 对象"))
            continue
        try:
            record = parse_record(item, index)
        except RecordParseError as exc:
            rejections.append(
                RejectedRecord(index, E_PARSE, str(exc), {"field": item})
            )
            continue
        if record.record_id in seen_ids:
            rejections.append(
                RejectedRecord(
                    index,
                    E_PARSE,
                    f"主体/对象标识 {record.record_id!r} 在文件内不唯一",
                    {"id": record.record_id},
                )
            )
            continue
        seen_ids.add(record.record_id)
        records.append(record)

    metadata = {"file_id": file_id, "root": root, "policy_hint": payload.get("policy")}
    return records, rejections, metadata


def load_manifest_file(path: str) -> tuple[list[Record], list[RejectedRecord], dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordParseError(f"清单文件读取/解码失败: {exc}") from exc
    return parse_manifest(payload)
