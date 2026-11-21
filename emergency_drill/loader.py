"""本地 JSON 加载与强校验入口。"""

import json
import os

from .errors import PlanValidationError
from .models import EventSet, Plan, validate_event_list, validate_plan_dict


def _read_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def load_plan(path):
    if not os.path.exists(path):
        raise PlanValidationError(f"预案文件不存在: {path}", location=path)
    try:
        raw = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise PlanValidationError(
            f"预案 JSON 损坏无法解析: {exc.msg}",
            location=f"{path}:line {exc.lineno} col {exc.colno}",
        ) from exc
    validate_plan_dict(raw)
    return Plan.from_dict(raw)


def load_events(path, plan):
    if not os.path.exists(path):
        raise PlanValidationError(f"注入事件文件不存在: {path}", location=path)
    try:
        raw = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise PlanValidationError(
            f"注入事件 JSON 损坏无法解析: {exc.msg}",
            location=f"{path}:line {exc.lineno} col {exc.colno}",
        ) from exc
    if isinstance(raw, dict) and "events" in raw:
        raw_events = raw["events"]
    elif isinstance(raw, list):
        raw_events = raw
    else:
        raise PlanValidationError("注入事件清单必须是数组或含 events 字段的对象", location=path)
    validate_event_list(raw_events, set(plan.stage_map))
    return EventSet.from_list(raw_events)


def load_bundle(plan_path, events_path):
    plan = load_plan(plan_path)
    events = load_events(events_path, plan)
    return plan, events
