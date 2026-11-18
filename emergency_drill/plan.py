"""Strict, all-or-nothing validation of drill plans and injected events."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .errors import E_CYCLE, E_PLAN, PlanValidationError
from .models import (
    EVENT_TYPES,
    FAILURE_POLICIES,
    EventDefinition,
    Plan,
    StageDefinition,
)


def _is_int(value: Any) -> bool:
    return type(value) is int


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _error(
    errors: list[dict[str, Any]],
    message: str,
    path: str,
    stage_seq: int | None = None,
) -> None:
    errors.append({"message": message, "field_path": path, "stage_seq": stage_seq})


def _find_cycle(
    stage_map: dict[int, dict[str, Any]], errors: list[dict[str, Any]]
) -> bool:
    state: dict[int, int] = {seq: 0 for seq in stage_map}
    stack: list[int] = []
    stage_index = {seq: index for index, seq in enumerate(stage_map)}
    found = False

    def visit(seq: int) -> None:
        nonlocal found
        if found:
            return
        state[seq] = 1
        stack.append(seq)
        raw = stage_map[seq]
        dependencies = raw.get("depends_on", [])
        if not isinstance(dependencies, list):
            dependencies = []
        for position, dependency in enumerate(dependencies):
            if not _is_int(dependency) or dependency not in stage_map:
                continue
            if state[dependency] == 1:
                cycle_start = stack.index(dependency)
                cycle = stack[cycle_start:] + [dependency]
                path = f"stages[{stage_index[seq]}].depends_on[{position}]"
                _error(
                    errors,
                    "阶段前置引用成环: " + " -> ".join(str(item) for item in cycle),
                    path,
                    seq,
                )
                found = True
                return
            if state[dependency] == 0:
                visit(dependency)
                if found:
                    return
        stack.pop()
        state[seq] = 2

    for seq in sorted(stage_map):
        if state[seq] == 0:
            visit(seq)
        if found:
            break
    return found


def validate_plan_object(raw: Any) -> Plan:
    """Validate a decoded JSON object and return a Plan only when complete.

    Validation errors are accumulated so callers can inspect every bad field,
    but no Plan is returned unless the whole document is valid. A dependency
    cycle is reported as E_CYCLE; other document failures are E_PLAN.
    """

    if not isinstance(raw, dict):
        raise PlanValidationError(
            "预案根节点必须是 JSON 对象",
            code=E_PLAN,
            stage_seq=None,
            field_path="$",
            errors=[{"message": "预案根节点必须是 JSON 对象", "field_path": "$", "stage_seq": None}],
        )

    errors: list[dict[str, Any]] = []
    plan_id = raw.get("plan_id")
    name = raw.get("name")
    version = raw.get("version")
    failure_policy = raw.get("failure_policy", "continue_downstream")

    if not _nonempty_string(plan_id):
        _error(errors, "plan_id 必须是非空字符串", "plan_id")
    if not _nonempty_string(name):
        _error(errors, "name 必须是非空字符串", "name")
    if not _is_int(version) or version <= 0:
        _error(errors, "version 必须是正整数", "version")
    if failure_policy not in FAILURE_POLICIES:
        _error(
            errors,
            "failure_policy 必须是 skip_downstream 或 continue_downstream",
            "failure_policy",
        )

    raw_stages = raw.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        _error(errors, "stages 必须是非空数组", "stages")
        raw_stages = []

    parsed_stages: list[StageDefinition] = []
    stage_map: dict[int, dict[str, Any]] = {}
    stage_positions: dict[int, int] = {}
    seen_seq: dict[int, int] = {}

    for index, item in enumerate(raw_stages):
        path = f"stages[{index}]"
        if not isinstance(item, dict):
            _error(errors, "阶段必须是 JSON 对象", path)
            continue
        seq = item.get("seq")
        stage_seq_for_error = seq if _is_int(seq) else None
        if not _is_int(seq) or seq <= 0:
            _error(errors, "seq 必须是正整数", f"{path}.seq")
        elif seq in seen_seq:
            _error(
                errors,
                f"阶段序号重复（首次出现在 stages[{seen_seq[seq]}]）",
                f"{path}.seq",
                seq,
            )
        else:
            seen_seq[seq] = index
            stage_map[seq] = item
            stage_positions[seq] = index

        stage_name = item.get("name")
        estimated_seconds = item.get("estimated_seconds")
        depends_on = item.get("depends_on")
        actions = item.get("actions")
        success_criteria = item.get("success_criteria")
        stage_policy = item.get("failure_policy", failure_policy)

        if not _nonempty_string(stage_name):
            _error(errors, "name 必须是非空字符串", f"{path}.name", stage_seq_for_error)
        if not _is_int(estimated_seconds) or estimated_seconds < 0:
            _error(
                errors,
                "estimated_seconds 必须是非负整数",
                f"{path}.estimated_seconds",
                stage_seq_for_error,
            )
        if "depends_on" not in item:
            _error(
                errors,
                "depends_on 字段缺失（无前置阶段时使用空数组）",
                f"{path}.depends_on",
                stage_seq_for_error,
            )
        elif not isinstance(depends_on, list):
            _error(errors, "depends_on 必须是整数数组", f"{path}.depends_on", stage_seq_for_error)
        else:
            for dep_index, dependency in enumerate(depends_on):
                if not _is_int(dependency) or dependency <= 0:
                    _error(
                        errors,
                        "前置阶段引用必须是正整数",
                        f"{path}.depends_on[{dep_index}]",
                        stage_seq_for_error,
                    )
        if not isinstance(actions, list) or not actions:
            _error(errors, "actions 必须是非空字符串数组", f"{path}.actions", stage_seq_for_error)
        else:
            for action_index, action in enumerate(actions):
                if not _nonempty_string(action):
                    _error(
                        errors,
                        "动作必须是非空字符串",
                        f"{path}.actions[{action_index}]",
                        stage_seq_for_error,
                    )
        if not isinstance(success_criteria, list) or not success_criteria:
            _error(
                errors,
                "success_criteria 必须是非空字符串数组",
                f"{path}.success_criteria",
                stage_seq_for_error,
            )
        else:
            for criterion_index, criterion in enumerate(success_criteria):
                if not _nonempty_string(criterion):
                    _error(
                        errors,
                        "成功判据必须是非空字符串",
                        f"{path}.success_criteria[{criterion_index}]",
                        stage_seq_for_error,
                    )
        if stage_policy not in FAILURE_POLICIES:
            _error(
                errors,
                "failure_policy 必须是 skip_downstream 或 continue_downstream",
                f"{path}.failure_policy",
                stage_seq_for_error,
            )

        if (
            _is_int(seq)
            and seq > 0
            and seq not in [existing.seq for existing in parsed_stages]
            and _nonempty_string(stage_name)
            and _is_int(estimated_seconds)
            and estimated_seconds >= 0
            and isinstance(depends_on, list)
            and all(_is_int(dep) and dep > 0 for dep in depends_on)
            and isinstance(actions, list)
            and actions
            and all(_nonempty_string(action) for action in actions)
            and isinstance(success_criteria, list)
            and success_criteria
            and all(_nonempty_string(criterion) for criterion in success_criteria)
            and stage_policy in FAILURE_POLICIES
        ):
            parsed_stages.append(
                StageDefinition(
                    seq=seq,
                    name=stage_name,
                    estimated_seconds=estimated_seconds,
                    depends_on=tuple(depends_on),
                    actions=tuple(actions),
                    success_criteria=tuple(success_criteria),
                    failure_policy=stage_policy,
                )
            )

    if raw_stages:
        expected = list(range(1, len(raw_stages) + 1))
        actual = sorted(seen_seq)
        if not errors and actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            if missing:
                _error(errors, f"阶段序号不连续，缺少 {missing}", "stages.seq", missing[0])
            if extra:
                _error(errors, f"阶段序号不连续，越界 {extra}", "stages.seq", extra[0])

    all_stage_seqs = set(stage_map)
    for seq in sorted(stage_map):
        item = stage_map[seq]
        dependencies = item.get("depends_on", [])
        if not isinstance(dependencies, list):
            continue
        index = stage_positions[seq]
        for position, dependency in enumerate(dependencies):
            if _is_int(dependency) and dependency > 0 and dependency not in all_stage_seqs:
                _error(
                    errors,
                    f"前置阶段 {dependency} 不存在",
                    f"stages[{index}].depends_on[{position}]",
                    seq,
                )

    cycle_found = _find_cycle(stage_map, errors) if stage_map else False

    if "events" not in raw:
        _error(errors, "events 字段缺失（无注入事件时使用空数组）", "events")
        raw_events = []
    else:
        raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        _error(errors, "events 必须是数组", "events")
        raw_events = []
    parsed_events: list[EventDefinition] = []
    seen_event_ids: dict[str, int] = {}
    for index, item in enumerate(raw_events):
        path = f"events[{index}]"
        if not isinstance(item, dict):
            _error(errors, "注入事件必须是 JSON 对象", path)
            continue
        event_id = item.get("event_id")
        event_type = item.get("type")
        offset_ms = item.get("offset_ms")
        event_stage_seq = item.get("stage_seq")
        stage_seq_for_error = event_stage_seq if _is_int(event_stage_seq) else None
        valid = True
        if not _nonempty_string(event_id):
            _error(errors, "event_id 必须是非空字符串", f"{path}.event_id", stage_seq_for_error)
            valid = False
        elif event_id in seen_event_ids:
            _error(
                errors,
                f"event_id 重复（首次出现在 events[{seen_event_ids[event_id]}]）",
                f"{path}.event_id",
                stage_seq_for_error,
            )
            valid = False
        else:
            seen_event_ids[event_id] = index
        if event_type not in EVENT_TYPES:
            _error(errors, "type 必须是受支持的注入事件类型", f"{path}.type", stage_seq_for_error)
            valid = False
        if not _is_int(offset_ms) or offset_ms < 0:
            _error(
                errors,
                "offset_ms 必须是非负整数",
                f"{path}.offset_ms",
                stage_seq_for_error,
            )
            valid = False
        if not _is_int(event_stage_seq) or event_stage_seq not in all_stage_seqs:
            _error(
                errors,
                "stage_seq 必须是已存在阶段的序号",
                f"{path}.stage_seq",
                stage_seq_for_error,
            )
            valid = False
        if valid:
            parsed_events.append(
                EventDefinition(
                    event_id=event_id,
                    type=event_type,
                    offset_ms=offset_ms,
                    stage_seq=event_stage_seq,
                )
            )

    if errors:
        first = errors[0]
        cycle_error = next((item for item in errors if "成环" in item["message"]), None)
        selected = cycle_error or first
        raise PlanValidationError(
            "预案整体拒绝加载；未产生任何部分生效",
            code=E_CYCLE if cycle_found else E_PLAN,
            stage_seq=selected.get("stage_seq"),
            field_path=selected.get("field_path"),
            errors=errors,
        )

    if actual != expected:
        raise PlanValidationError(
            "阶段序号必须从 1 开始连续唯一",
            code=E_PLAN,
            field_path="stages.seq",
            errors=errors,
        )

    return Plan(
        plan_id=plan_id,
        name=name,
        version=version,
        failure_policy=failure_policy,
        stages=tuple(sorted(parsed_stages, key=lambda item: item.seq)),
        events=tuple(parsed_events),
    )


def load_plan(path: str | Path) -> Plan:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanValidationError(
            f"无法读取预案文件: {exc}",
            code=E_PLAN,
            field_path="$file",
            errors=[{"message": str(exc), "field_path": "$file", "stage_seq": None}],
        ) from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanValidationError(
            f"预案 JSON 无法解析: {exc}",
            code=E_PLAN,
            field_path=f"$line:{exc.lineno}",
            errors=[
                {
                    "message": f"JSON 解析失败: {exc.msg}",
                    "field_path": f"$line:{exc.lineno}",
                    "stage_seq": None,
                }
            ],
        ) from exc
    return validate_plan_object(raw)


def with_plan_version(raw: Any, version: int) -> dict[str, Any]:
    """Return a deep copy of a plan document with a new version."""

    if not isinstance(raw, dict):
        return raw
    updated = copy.deepcopy(raw)
    updated["version"] = version
    return updated


def validate_event_object(raw: Any, plan: Plan) -> EventDefinition:
    """Validate one user-injected event against the active plan."""

    errors: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        raise PlanValidationError(
            "注入事件必须是 JSON 对象",
            code=E_PLAN,
            field_path="event",
            errors=[{"message": "注入事件必须是 JSON 对象", "field_path": "event", "stage_seq": None}],
        )
    event_id = raw.get("event_id")
    event_type = raw.get("type")
    offset_ms = raw.get("offset_ms")
    stage_seq = raw.get("stage_seq")
    stage_locator = stage_seq if _is_int(stage_seq) else None
    if not _nonempty_string(event_id):
        _error(errors, "event_id 必须是非空字符串", "event.event_id", stage_locator)
    if event_type not in EVENT_TYPES:
        _error(errors, "type 必须是受支持的注入事件类型", "event.type", stage_locator)
    if not _is_int(offset_ms) or offset_ms < 0:
        _error(errors, "offset_ms 必须是非负整数", "event.offset_ms", stage_locator)
    if not _is_int(stage_seq) or stage_seq not in plan.stage_seqs:
        _error(errors, "stage_seq 必须是已存在阶段的序号", "event.stage_seq", stage_locator)
    if errors:
        first = errors[0]
        raise PlanValidationError(
            "注入事件整体拒绝",
            code=E_PLAN,
            stage_seq=first.get("stage_seq"),
            field_path=first.get("field_path"),
            errors=errors,
        )
    return EventDefinition(
        event_id=event_id,
        type=event_type,
        offset_ms=offset_ms,
        stage_seq=stage_seq,
    )
