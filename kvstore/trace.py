"""确定性读写轨迹（JSON Lines）的加载、校验与顺序推进。

每行一个操作，字段：``op_seq``（连续唯一，从 1 起）、``op``、``clock``
（整数逻辑时钟，单调不减），PUT 必须携带非空字符串 ``key``/``value`` 与
非负整数 ``expire_ms``。任何违规首错即抛 E_TRACE 并报告首个出错操作序号。
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from . import codec
from .errors import TraceError

VALID_OPS = ("PUT", "GET", "DEL", "PURGE", "REBUILD_INDEX")


def load_trace(path: str) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    seen_seq: set[int] = set()
    expected = 1
    last_clock = -1
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                op = codec.canonical_json  # noqa: F841 - 保持导入对称
                import json

                operation = json.loads(line)
            except Exception as exc:
                raise TraceError(
                    "轨迹行不是合法 JSON",
                    details={"op_seq": expected, "line_no": line_no, "reason": "bad_json",
                             "error": str(exc)},
                ) from exc
            _validate_operation(operation, line_no, seen_seq, expected, last_clock)
            last_clock = operation["clock"]
            seen_seq.add(operation["op_seq"])
            expected += 1
            operations.append(operation)
    if not operations:
        raise TraceError("轨迹为空", details={"op_seq": 0, "reason": "empty_trace"})
    return operations


def _validate_operation(
    op: Any,
    line_no: int,
    seen_seq: set[int],
    expected: int,
    last_clock: int,
) -> None:
    if not isinstance(op, dict):
        raise TraceError("操作必须为对象",
                         details={"op_seq": expected, "line_no": line_no,
                                  "reason": "not_an_object"})
    op_seq = op.get("op_seq")
    if not isinstance(op_seq, int) or isinstance(op_seq, bool):
        raise TraceError("操作序号必须为整数",
                         details={"op_seq": op_seq, "line_no": line_no,
                                  "reason": "bad_op_seq"})
    if op_seq in seen_seq:
        raise TraceError("操作序号重复",
                         details={"op_seq": op_seq, "line_no": line_no,
                                  "reason": "duplicate_op_seq"})
    if op_seq != expected:
        raise TraceError("操作序号不连续",
                         details={"op_seq": op_seq, "line_no": line_no,
                                  "reason": "non_contiguous", "expected": expected})
    name = op.get("op")
    if name not in VALID_OPS:
        raise TraceError("未知操作类型",
                         details={"op_seq": op_seq, "line_no": line_no,
                                  "reason": "bad_op", "actual": name})
    clock = op.get("clock")
    if not isinstance(clock, int) or isinstance(clock, bool) or clock < 0:
        raise TraceError("逻辑时钟必须为非负整数",
                         details={"op_seq": op_seq, "line_no": line_no,
                                  "reason": "bad_clock"})
    if clock < last_clock:
        raise TraceError("逻辑时钟必须单调不减",
                         details={"op_seq": op_seq, "line_no": line_no,
                                  "reason": "clock_regressed",
                                  "previous": last_clock, "actual": clock})
    if name == "PUT":
        for field in ("key", "value", "expire_ms"):
            if field not in op:
                raise TraceError("PUT 缺少必填字段",
                                 details={"op_seq": op_seq, "line_no": line_no,
                                          "reason": "missing_field", "field": field})
        try:
            codec.validate_key(op["key"])
        except Exception as exc:
            raise TraceError("PUT 键非法",
                             details={"op_seq": op_seq, "line_no": line_no,
                                      "reason": "bad_key", "error": exc.details}) from exc
        if not isinstance(op["value"], str):
            raise TraceError("PUT 值必须为字符串",
                             details={"op_seq": op_seq, "line_no": line_no,
                                      "reason": "bad_value"})
        expire = op["expire_ms"]
        if not isinstance(expire, int) or isinstance(expire, bool) or expire < 0:
            raise TraceError("PUT 过期戳必须为非负整数",
                             details={"op_seq": op_seq, "line_no": line_no,
                                      "reason": "bad_expire_ms"})
    if name in ("GET", "DEL"):
        try:
            codec.validate_key(op.get("key"))
        except Exception as exc:
            raise TraceError(f"{name} 键非法",
                             details={"op_seq": op_seq, "line_no": line_no,
                                      "reason": "bad_key",
                                      "error": getattr(exc, "details", None)}) from exc


def operations_digest(operations: list[dict[str, Any]]) -> dict[str, Any]:
    """对轨迹计算确定性摘要；同一轨迹多次加载逐字节一致。"""
    lines = [codec.canonical_json(op) for op in operations]
    joined = b"\n".join(lines)
    return {
        "op_count": len(operations),
        "sha256": hashlib.sha256(joined).hexdigest(),
        "first_op_seq": operations[0]["op_seq"],
        "last_op_seq": operations[-1]["op_seq"],
        "last_clock": operations[-1]["clock"],
    }


def write_trace(path: str, operations: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for op in operations:
            handle.write(codec.canonical_json(op).decode("utf-8") + "\n")
