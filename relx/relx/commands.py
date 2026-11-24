"""编辑命令流 JSON 的加载与校验。

支持命令：
  edit_tokens   修改单句用词（全量替换 token 序列）
  add_entity    新增实体
  remove_entity 删除实体（同时删除引用该实体的关系标注）
  set_template_weight 调整模板权重并推进模板版本号
  set_threshold 调整关系得分阈值
  set_quant     切换定点位宽与量化档位
非法命令抛出 E_COMMAND_INVALID，由上层决定是否记入失败清单。
"""

import json
from dataclasses import dataclass

from .errors import E_COMMAND_INVALID, RelxError
from .fixedpoint import ALLOWED_Q_LEVELS, ALLOWED_WIDTHS

VALID_COMMANDS = (
    "edit_tokens",
    "add_entity",
    "remove_entity",
    "set_template_weight",
    "set_threshold",
    "set_quant",
)


@dataclass(frozen=True)
class Command:
    seq: int
    kind: str
    payload: dict
    comment: str

    def fingerprint_material(self):
        return {"seq": self.seq, "kind": self.kind, "payload": self.payload}


def _require(payload, key, kind):
    if key not in payload:
        raise RelxError(E_COMMAND_INVALID, f"{kind} 命令缺少字段 {key}")
    return payload[key]


def load_commands(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise RelxError(E_COMMAND_INVALID, f"命令流文件不存在: {path}")
    except json.JSONDecodeError as exc:
        raise RelxError(E_COMMAND_INVALID, f"命令流 JSON 解析失败: {exc}")

    if not isinstance(data, dict) or not isinstance(data.get("commands"), list):
        raise RelxError(E_COMMAND_INVALID, "命令流根对象必须包含 commands 列表")

    commands = []
    for index, raw in enumerate(data["commands"]):
        if not isinstance(raw, dict) or raw.get("kind") not in VALID_COMMANDS:
            raise RelxError(E_COMMAND_INVALID,
                            f"第 {index} 条命令非法: {raw!r}")
        kind = raw["kind"]
        payload = raw.get("payload", {})
        if not isinstance(payload, dict):
            raise RelxError(E_COMMAND_INVALID, f"{kind} 的 payload 必须为对象")

        if kind == "edit_tokens":
            tokens = _require(payload, "tokens", kind)
            if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) for t in tokens):
                raise RelxError(E_COMMAND_INVALID, "edit_tokens.tokens 必须为非空字符串列表")
        elif kind == "add_entity":
            for key in ("entity_id", "start", "end", "type"):
                _require(payload, key, kind)
            if not isinstance(payload["entity_id"], str):
                raise RelxError(E_COMMAND_INVALID, "add_entity.entity_id 必须为字符串")
            if not isinstance(payload["start"], int) or not isinstance(payload["end"], int):
                raise RelxError(E_COMMAND_INVALID, "add_entity 边界必须为整数")
        elif kind == "remove_entity":
            _require(payload, "entity_id", kind)
        elif kind == "set_template_weight":
            _require(payload, "template_id", kind)
            weight = _require(payload, "weight", kind)
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise RelxError(E_COMMAND_INVALID, "模板权重必须为数值")
        elif kind == "set_threshold":
            threshold = _require(payload, "threshold", kind)
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
                raise RelxError(E_COMMAND_INVALID, "阈值必须为数值")
        elif kind == "set_quant":
            width = _require(payload, "fixedpoint_width", kind)
            quant = _require(payload, "quant_level", kind)
            if width not in ALLOWED_WIDTHS or quant not in ALLOWED_Q_LEVELS:
                raise RelxError(E_COMMAND_INVALID,
                                f"位宽/档位非法: width={width}, quant={quant}")

        commands.append(Command(
            seq=raw.get("seq", index),
            kind=kind,
            payload=dict(payload),
            comment=str(raw.get("comment", "")),
        ))

    seqs = [cmd.seq for cmd in commands]
    if len(set(seqs)) != len(seqs):
        raise RelxError(E_COMMAND_INVALID, "命令 seq 必须唯一")
    commands.sort(key=lambda cmd: cmd.seq)
    return commands
