"""历史版本丢包事件清单（v1/v2/v3）的读取、校验与归一。

清单文件为 JSON，顶层信封::

    {"format": "loss-event-list", "version": "v1|v2|v3", "events": [...]}

各版本字段：
- v1：事件仅含 ``event_id`` 与 ``loss_count``；流归属放在信封顶层 ``flow``
  （可选 ``direction``），符合“v1 只含事件序号与丢包计数”的历史定义。
- v2：事件增加 ``link_id`` 与毫秒时间窗 ``window_ms=[start,end]``，
  并在事件内携带 ``flow``（可选 ``direction``）。
- v3：事件再增加 ``retransmitted`` 与 ``ooo_depth``。

任何校验失败都以 E_FORMAT 整体拒绝，错误上下文带出错字段路径，
例如 ``events[3].loss_count``。归一后每条事件保留 ``source_version``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .constants import SUPPORTED_EVENT_VERSIONS
from .errors import EventFormatError


@dataclass
class LossEvent:
    """归一后的丢包事件。"""

    event_id: int
    flow: str
    loss_count: int
    source_version: str
    direction: Optional[str] = None
    link_id: Optional[str] = None
    window_ms: Optional[Tuple[int, int]] = None
    retransmitted: Optional[bool] = None
    ooo_depth: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "flow": self.flow,
            "direction": self.direction,
            "loss_count": self.loss_count,
            "link_id": self.link_id,
            "window_ms": list(self.window_ms) if self.window_ms else None,
            "retransmitted": self.retransmitted,
            "ooo_depth": self.ooo_depth,
            "source_version": self.source_version,
        }


@dataclass
class NormalizedEventLog:
    """整份清单的归一结果与信封级元数据。"""

    version: str
    events: List[LossEvent] = field(default_factory=list)
    coverage: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def versions(self) -> List[str]:
        return sorted({e.source_version for e in self.events})


def _reject(path: str, reason: str, value: Any = None) -> None:
    ctx: Dict[str, Any] = {"field_path": path, "reason": reason}
    if value is not None:
        ctx["value"] = value
    raise EventFormatError(f"事件清单格式错误 @ {path}: {reason}", ctx)


def _is_int(value: Any) -> bool:
    # bool 是 int 的子类，清单中必须显式拒绝
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(mapping: Dict[str, Any], key: str, path: str,
                 allow_negative: bool = False) -> int:
    if key not in mapping:
        _reject(f"{path}.{key}", "必需字段缺失")
    value = mapping[key]
    if not _is_int(value):
        _reject(f"{path}.{key}", "字段必须为整数", value)
    if not allow_negative and value < 0:
        _reject(f"{path}.{key}", "计数值不得为负", value)
    return value


def _validate_coverage(raw: Dict[str, Any]) -> Dict[str, Dict[str, List[int]]]:
    coverage = raw.get("coverage")
    if coverage is None:
        return {}
    if not isinstance(coverage, dict):
        _reject("coverage", "字段必须为对象")
    normalized: Dict[str, Dict[str, List[int]]] = {}
    for flow, dirs in coverage.items():
        if not isinstance(dirs, dict):
            _reject(f"coverage.{flow}", "字段必须为对象")
        normalized[flow] = {}
        for direction, interval in dirs.items():
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or not all(_is_int(v) for v in interval)):
                _reject(f"coverage.{flow}.{direction}",
                        "必须为 [min_seq, max_seq] 整数区间", interval)
            lo, hi = interval
            if lo > hi:
                _reject(f"coverage.{flow}.{direction}",
                        "区间下界不得大于上界", interval)
            normalized[flow][direction] = [lo, hi]
    return normalized


def parse_event_log(raw: Any) -> NormalizedEventLog:
    """解析并归一 JSON 内容（已反序列化的 Python 对象）。"""
    if not isinstance(raw, dict):
        _reject("$", "顶层必须为 JSON 对象")
    version = raw.get("version")
    if version not in SUPPORTED_EVENT_VERSIONS:
        _reject("version", f"未知版本 {version!r}，支持 {SUPPORTED_EVENT_VERSIONS}",
                version)
    if raw.get("format") != "loss-event-list":
        _reject("format", "format 必须为 'loss-event-list'", raw.get("format"))
    raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        _reject("events", "必需字段缺失或必须为数组")

    envelope_flow = raw.get("flow")
    envelope_direction = raw.get("direction")
    if version == "v1":
        if not isinstance(envelope_flow, str) or not envelope_flow:
            _reject("flow", "v1 清单必须在信封顶层声明非空 flow")
    elif envelope_flow is not None and not isinstance(envelope_flow, str):
        _reject("flow", "flow 必须为字符串")

    events: List[LossEvent] = []
    seen_ids = set()
    for idx, item in enumerate(raw_events):
        path = f"events[{idx}]"
        if not isinstance(item, dict):
            _reject(path, "事件必须为对象")

        event_id = _require_int(item, "event_id", path)
        if event_id in seen_ids:
            _reject(f"{path}.event_id", "事件序号重复", event_id)
        seen_ids.add(event_id)
        loss_count = _require_int(item, "loss_count", path)
        if loss_count == 0:
            _reject(f"{path}.loss_count", "丢包计数必须为正整数", loss_count)

        flow = envelope_flow
        direction = envelope_direction
        link_id: Optional[str] = None
        window_ms: Optional[Tuple[int, int]] = None
        retransmitted: Optional[bool] = None
        ooo_depth: Optional[int] = None

        if version in ("v2", "v3"):
            flow_value = item.get("flow")
            if not isinstance(flow_value, str) or not flow_value:
                _reject(f"{path}.flow", "v2/v3 事件必须声明非空 flow")
            flow = flow_value
            if "direction" in item and item["direction"] is not None:
                if not isinstance(item["direction"], str):
                    _reject(f"{path}.direction", "direction 必须为字符串")
                direction = item["direction"]
            link_value = item.get("link_id")
            if not isinstance(link_value, str) or not link_value:
                _reject(f"{path}.link_id", "v2/v3 事件必须声明非空 link_id")
            link_id = link_value
            window = item.get("window_ms")
            if not isinstance(window, list) or len(window) != 2 or not all(
                    _is_int(v) for v in window):
                _reject(f"{path}.window_ms",
                        "必须为 [start_ms, end_ms] 整数区间", window)
            start_ms, end_ms = window
            if start_ms < 0 or end_ms < start_ms:
                _reject(f"{path}.window_ms",
                        "时间窗必须满足 0 <= start <= end", window)
            window_ms = (start_ms, end_ms)

        if version == "v3":
            if "retransmitted" not in item:
                _reject(f"{path}.retransmitted", "v3 必需字段缺失")
            if not isinstance(item["retransmitted"], bool):
                _reject(f"{path}.retransmitted", "必须为布尔值",
                        item["retransmitted"])
            retransmitted = item["retransmitted"]
            ooo_depth = _require_int(item, "ooo_depth", path)

        events.append(LossEvent(
            event_id=event_id,
            flow=flow,
            direction=direction,
            loss_count=loss_count,
            link_id=link_id,
            window_ms=window_ms,
            retransmitted=retransmitted,
            ooo_depth=ooo_depth,
            source_version=version,
        ))

    coverage = _validate_coverage(raw)
    return NormalizedEventLog(version=version, events=events,
                              coverage=coverage, raw=raw)


def load_event_log(path: str | Path) -> NormalizedEventLog:
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EventFormatError(
            f"事件清单 JSON 解析失败: {exc}",
            {"field_path": "$", "reason": "invalid_json",
             "line": exc.lineno, "column": exc.colno}) from exc
    return parse_event_log(raw)


def load_event_logs(paths: List[str | Path]) -> List[NormalizedEventLog]:
    return [load_event_log(path) for path in paths]
