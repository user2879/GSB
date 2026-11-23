"""交互命令流加载与应用（替换/删除对、调权重/阈值、切档位、切批次、配额）。

命令只负责声明式地产生“下一个状态”，真正的增量重算由 pipeline 完成，
因此任意命令序列的最终状态都等价于一次性全量重建。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from .config import AppConfig
from .errors import (
    E_COMMAND_INVALID, E_PAIR_NOT_FOUND, FailureEntry, FailureLog,
)
from .pairs import PreferencePair, _validate_record

SUPPORTED_OPS = (
    "replace_pair", "delete_pair", "set_weight", "set_boundary_threshold",
    "set_quant_mode", "set_decimal_places", "set_tokenizer_version",
    "set_batch", "set_per_batch_quota", "set_global_budget",
)


@dataclass
class Command:
    seq: int
    op: str
    payload: dict[str, Any]


def load_commands(path: str, failures: FailureLog) -> list[Command]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        failures.add(FailureEntry(
            sort_key="command-file", code=E_COMMAND_INVALID,
            stage="load_command", message=f"cannot parse json: {exc}",
        ))
        return []

    if not isinstance(raw, list):
        failures.add(FailureEntry(
            sort_key="command-file", code=E_COMMAND_INVALID,
            stage="load_command", message="command stream must be a list",
        ))
        return []

    commands: list[Command] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict) or "op" not in item:
            failures.add(FailureEntry(
                sort_key=f"cmd-{idx:04d}", code=E_COMMAND_INVALID,
                stage="load_command", message="command must be object with op",
                context=f"index={idx}",
            ))
            continue
        op = item["op"]
        if op not in SUPPORTED_OPS:
            failures.add(FailureEntry(
                sort_key=f"cmd-{idx:04d}", code=E_COMMAND_INVALID,
                stage="load_command", message=f"unsupported op {op!r}",
                context=f"index={idx}",
            ))
            continue
        commands.append(Command(seq=idx, op=op, payload=item.get("payload", {})))
    return commands


def apply_command(command: Command, pairs: dict[str, PreferencePair],
                  cfg: AppConfig, normalize: bool,
                  failures: FailureLog) -> tuple[dict[str, PreferencePair], AppConfig, set[str]]:
    """返回 (新 pairs, 新 cfg, 受影响对集合)。失败写入 failures 且状态不变。"""
    p = command.payload
    changed: set[str] = set()

    if command.op == "replace_pair":
        record = p.get("record")
        if not isinstance(record, dict):
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="replace_pair needs record",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        pair, problem = _validate_record(record, normalize)
        if pair is None:
            failures.add(FailureEntry(
                sort_key=str(record.get("pair_id", f"cmd-{command.seq:04d}")),
                code=E_COMMAND_INVALID, stage="apply_command",
                message=f"invalid replacement record: {problem}",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        pairs = dict(pairs)
        pairs[pair.pair_id] = pair
        changed.add(pair.pair_id)
        return pairs, cfg, changed

    if command.op == "delete_pair":
        pid = p.get("pair_id")
        if pid not in pairs:
            failures.add(FailureEntry(
                sort_key=str(pid), code=E_PAIR_NOT_FOUND,
                stage="apply_command", message="delete target does not exist",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        pairs = dict(pairs)
        pairs.pop(pid)
        changed.add(pid)
        return pairs, cfg, changed

    if command.op == "set_weight":
        name = p.get("name")
        value = p.get("value")
        if name not in ("prompt", "chosen", "rejected") or not isinstance(value, (int, float)) \
                or isinstance(value, bool) or not _finite(value):
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid set_weight payload",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        weights = replace(cfg.weights, **{name: float(value)})
        cfg = replace(cfg, weights=weights)
        changed.update(pairs.keys())
        return pairs, cfg, changed

    if command.op == "set_boundary_threshold":
        value = p.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not _finite(value) \
                or value < 0:
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid boundary threshold",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        cfg = replace(cfg, boundary_threshold=float(value))
        changed.update(pairs.keys())
        return pairs, cfg, changed

    if command.op == "set_quant_mode":
        value = p.get("value")
        if value not in ("trunc", "half_even", "half_up"):
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid quant_mode",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        cfg = replace(cfg, quant_mode=value)
        changed.update(pairs.keys())
        return pairs, cfg, changed

    if command.op == "set_decimal_places":
        value = p.get("value")
        if value not in (4, 6, 8):
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid decimal_places",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        cfg = replace(cfg, decimal_places=value)
        changed.update(pairs.keys())
        return pairs, cfg, changed

    if command.op == "set_tokenizer_version":
        value = p.get("value")
        if value not in ("v1", "v2"):
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid tokenizer_version",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        cfg = replace(cfg, tokenizer_version=value)
        changed.update(pairs.keys())
        return pairs, cfg, changed

    if command.op == "set_batch":
        pid = p.get("pair_id")
        batch_id = p.get("batch_id")
        if pid not in pairs:
            failures.add(FailureEntry(
                sort_key=str(pid), code=E_PAIR_NOT_FOUND,
                stage="apply_command", message="set_batch target does not exist",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        if not isinstance(batch_id, str) or not batch_id:
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid batch_id",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        pairs = dict(pairs)
        pairs[pid] = replace(pairs[pid], batch_id=batch_id)
        changed.add(pid)
        return pairs, cfg, changed

    if command.op == "set_per_batch_quota":
        value = p.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid per_batch_quota",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        cfg = replace(cfg, per_batch_quota=value)
        return pairs, cfg, set()

    if command.op == "set_global_budget":
        value = p.get("value")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            failures.add(FailureEntry(
                sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
                stage="apply_command", message="invalid global budget",
                context=f"seq={command.seq}",
            ))
            return pairs, cfg, changed
        cfg = replace(cfg, global_pair_budget=value)
        return pairs, cfg, set()

    failures.add(FailureEntry(
        sort_key=f"cmd-{command.seq:04d}", code=E_COMMAND_INVALID,
        stage="apply_command", message=f"unhandled op {command.op!r}",
        context=f"seq={command.seq}",
    ))
    return pairs, cfg, changed


def _finite(value: float) -> bool:
    import math
    return math.isfinite(value)
