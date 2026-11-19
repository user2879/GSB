"""参数化管线状态、版本化操作序列与失效闭包。

每帧的处理状态是一个 :class:`PipelineState`（目标色域、曝光 stops、
色调映射算子及参数、输出传递函数）。操作（operation）以带版本号的
不可变序列记录，每条操作携带父状态 id，形成可重放的状态历史。

状态 id 由状态参数的规范化 JSON 做 SHA-256 得到——同样参数必然同 id，
且与到达该状态的操作粒度（逐帧 / 区间）无关。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field

from .errors import E_STATE_VERSION, E_TONEMAP_PARAM, WorkbenchError
from .pipeline import VALID_COLORSPACES, VALID_OETF, VALID_TONEMAPS, validate_state_params

STATE_SCHEMA_VERSION = 3
DEFAULT_STATE = {
    "colorspace": "bt2020",
    "exposure_stops": 0.0,
    "tonemap": "aces",
    "tonemap_param": 1.0,
    "oetf": "bt709",
}


def canonical_json(obj) -> bytes:
    """确定性 JSON 序列化：键排序、无空白、固定分隔符（UTF-8）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class PipelineState:
    colorspace: str = "bt2020"
    exposure_stops: float = 0.0
    tonemap: str = "aces"
    tonemap_param: float = 1.0
    oetf: str = "bt709"

    def validate(self) -> None:
        validate_state_params(self.colorspace, float(self.exposure_stops),
                              self.tonemap, float(self.tonemap_param), self.oetf)

    def to_dict(self) -> dict:
        return {
            "colorspace": self.colorspace,
            "exposure_stops": float(self.exposure_stops),
            "tonemap": self.tonemap,
            "tonemap_param": float(self.tonemap_param),
            "oetf": self.oetf,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineState":
        try:
            state = cls(
                colorspace=str(data["colorspace"]),
                exposure_stops=float(data["exposure_stops"]),
                tonemap=str(data["tonemap"]),
                tonemap_param=float(data["tonemap_param"]),
                oetf=str(data["oetf"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkbenchError(E_STATE_VERSION, f"状态字段缺失或类型错误: {exc}") from exc
        state.validate()
        return state

    def state_id(self) -> str:
        return hashlib.sha256(canonical_json(
            {"schema": STATE_SCHEMA_VERSION, "state": self.to_dict()})).hexdigest()


def default_state() -> PipelineState:
    return PipelineState(**DEFAULT_STATE)


class StateTable:
    """维护每帧当前状态与完整版本化操作序列。"""

    def __init__(self, frame_count: int, base: PipelineState | None = None):
        self.frame_count = int(frame_count)
        base = base or default_state()
        base.validate()
        self.states: list[PipelineState] = [copy.deepcopy(base) for _ in range(self.frame_count)]
        self.operations: list[dict] = []
        self.schema_version = STATE_SCHEMA_VERSION
        self._append_operation("init", list(range(self.frame_count)),
                               base.to_dict(), parent_ids=[], note="initial baseline")

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "frame_count": self.frame_count,
            "states": [s.to_dict() for s in self.states],
            "operations": self.operations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StateTable":
        if int(data.get("schema_version", -1)) != STATE_SCHEMA_VERSION:
            raise WorkbenchError(
                E_STATE_VERSION,
                f"状态版本冲突: 期望 {STATE_SCHEMA_VERSION}, 实际 {data.get('schema_version')}",
                details={"expected": STATE_SCHEMA_VERSION,
                         "actual": data.get("schema_version")})
        n = int(data["frame_count"])
        table = cls.__new__(cls)
        table.frame_count = n
        table.schema_version = int(data["schema_version"])
        table.states = [PipelineState.from_dict(d) for d in data["states"]]
        table.operations = list(data["operations"])
        if len(table.states) != n:
            raise WorkbenchError(E_STATE_VERSION, "状态表帧数与 frame_count 不一致")
        return table

    # -- 操作 -------------------------------------------------------------
    def _append_operation(self, kind: str, frames: list[int], params: dict,
                          parent_ids: list[str], note: str = "") -> dict:
        version = len(self.operations)
        op = {
            "version": version,
            "kind": kind,
            "frames": [int(i) for i in frames],
            "params": params,
            "parent_state_ids": parent_ids,
            "note": note,
        }
        self.operations.append(op)
        return op

    def apply(self, frames: list[int], changes: dict, *, kind: str = "adjust",
              note: str = "") -> dict:
        """对给定帧集合应用参数变更，返回操作记录与失效闭包。"""
        targets = self._normalize_frames(frames)
        closure = self.compute_closure(targets, changes)
        parent_ids = [self.states[i].state_id() for i in targets]
        new_states: dict[int, PipelineState] = {}
        for index in targets:
            merged = dict(self.states[index].to_dict())
            for key, value in changes.items():
                if key not in DEFAULT_STATE:
                    raise WorkbenchError(E_TONEMAP_PARAM, f"未知管线参数: {key}")
                merged[key] = value
            new_state = PipelineState.from_dict(merged)
            new_states[index] = new_state
        for index, new_state in new_states.items():
            self.states[index] = new_state
        op = self._append_operation(kind, targets, changes, parent_ids, note)
        op["closure"] = closure
        return op

    def _normalize_frames(self, frames: list[int]) -> list[int]:
        seen = set()
        ordered = []
        for raw in frames:
            index = int(raw)
            if not (0 <= index < self.frame_count):
                raise WorkbenchError(E_STATE_VERSION,
                                     f"帧号越界: {index} (共 {self.frame_count} 帧)")
            if index not in seen:
                seen.add(index)
                ordered.append(index)
        return ordered

    def compute_closure(self, targets: list[int], changes: dict) -> list[int]:
        """精确失效闭包：仅当新参数确实改变某帧状态时才纳入闭包。"""
        for key in changes:
            if key not in DEFAULT_STATE:
                raise WorkbenchError(E_TONEMAP_PARAM, f"未知管线参数: {key}")
        closure = []
        for index in targets:
            current = self.states[index].to_dict()
            changed = any(current[key] != changes[key] for key in changes)
            if changed:
                closure.append(index)
        return closure

    def state_ids(self) -> list[str]:
        return [s.state_id() for s in self.states]

    def replay_to(self, version: int) -> "StateTable":
        """从操作序列重放到指定版本（含），返回独立的状态表副本。"""
        if not (0 <= version < len(self.operations)):
            raise WorkbenchError(E_STATE_VERSION, f"重放版本越界: {version}")
        base_params = self.operations[0]["params"]
        table = StateTable.__new__(StateTable)
        table.frame_count = self.frame_count
        table.schema_version = self.schema_version
        base = PipelineState.from_dict(base_params)
        table.states = [copy.deepcopy(base) for _ in range(self.frame_count)]
        table.operations = [copy.deepcopy(self.operations[0])]
        for op in self.operations[1:version + 1]:
            table.apply(op["frames"], op["params"], kind=op["kind"], note=op.get("note", ""))
            # 保持历史 version 编号、父状态 id 与闭包与原始一致
            table.operations[-1]["version"] = op["version"]
            table.operations[-1]["parent_state_ids"] = list(op["parent_state_ids"])
            if "closure" in op:
                table.operations[-1]["closure"] = list(op["closure"])
        return table
