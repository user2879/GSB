"""演练预案与注入事件的不可变数据模型。

所有需要参与内容摘要或逐字节比对的结构都通过 :func:`canonical_json`
以固定键序、无多余空白、UTF-8 无 BOM 的方式序列化，保证跨进程一致。
"""

from dataclasses import dataclass
import hashlib
import json

from .errors import PlanValidationError

STAGE_STATES = ("pending", "running", "succeeded", "failed", "skipped")
INJECTION_KINDS = ("node_lost", "switch_fail", "check_timeout")
CRITERION_MODES = ("all_of", "any_of")


def canonical_bytes(obj):
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_json(obj):
    return canonical_bytes(obj).decode("utf-8")


def digest_of(obj):
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


@dataclass(frozen=True)
class Stage:
    stage_id: str
    order: int
    duration_ms: int
    prerequisites: tuple
    actions: tuple
    criterion: dict

    def to_dict(self):
        return {
            "stage_id": self.stage_id,
            "order": self.order,
            "duration_ms": self.duration_ms,
            "prerequisites": list(self.prerequisites),
            "actions": [dict(action) for action in self.actions],
            "criterion": dict(self.criterion),
        }

    @classmethod
    def from_dict(cls, raw):
        return cls(
            stage_id=raw["stage_id"],
            order=raw["order"],
            duration_ms=raw["duration_ms"],
            prerequisites=tuple(raw.get("prerequisites", [])),
            actions=tuple(dict(action) for action in raw.get("actions", [])),
            criterion=dict(raw.get("criterion", {})),
        )


@dataclass(frozen=True)
class InjectionEvent:
    event_id: str
    kind: str
    offset_ms: int
    target_stage: str

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "offset_ms": self.offset_ms,
            "target_stage": self.target_stage,
        }

    @classmethod
    def from_dict(cls, raw):
        return cls(
            event_id=raw["event_id"],
            kind=raw["kind"],
            offset_ms=raw["offset_ms"],
            target_stage=raw["target_stage"],
        )


@dataclass(frozen=True)
class Plan:
    plan_id: str
    stages: tuple
    version: int = 1

    @property
    def stage_map(self):
        return {stage.stage_id: stage for stage in self.stages}

    def stage_ids_topo(self):
        from .graph import topological_order

        return topological_order(self.stages)

    def digest(self):
        return digest_of(
            {
                "plan_id": self.plan_id,
                "version": self.version,
                "stages": [stage.to_dict() for stage in self.stages],
            }
        )

    def with_version(self, stages, version):
        return Plan(plan_id=self.plan_id, stages=tuple(stages), version=version)

    @classmethod
    def from_dict(cls, raw, version=None):
        stages = tuple(Stage.from_dict(item) for item in raw["stages"])
        return cls(
            plan_id=raw.get("plan_id", "plan"),
            stages=stages,
            version=raw.get("version", 1) if version is None else version,
        )


@dataclass(frozen=True)
class EventSet:
    events: tuple

    @property
    def event_map(self):
        return {event.event_id: event for event in self.events}

    def digest(self):
        return digest_of([event.to_dict() for event in self.events])

    @classmethod
    def from_list(cls, raw_events):
        return cls(tuple(InjectionEvent.from_dict(item) for item in raw_events))


def _is_int(value):
    # bool 是 int 的子类，业务上必须显式拒绝。
    return isinstance(value, int) and not isinstance(value, bool)


def _require_type(cond, message, location):
    if not cond:
        raise PlanValidationError(message, location=location)


def validate_plan_dict(raw):
    """对预案原始 JSON 做强校验，首个违规以 E_PLAN 整体拒绝。"""
    _require_type(isinstance(raw, dict), "预案根节点必须是对象", "$")
    stages = raw.get("stages")
    _require_type(isinstance(stages, list), "stages 必须为数组", "$.stages")

    seen_ids = set()
    orders = []
    for index, raw_stage in enumerate(stages):
        base = f"$.stages[{index}]"
        _require_type(isinstance(raw_stage, dict), "阶段必须是对象", base)
        stage_id = raw_stage.get("stage_id")
        _require_type(isinstance(stage_id, str) and stage_id, "stage_id 必须为非空字符串", f"{base}.stage_id")
        if stage_id in seen_ids:
            raise PlanValidationError(f"阶段标识重复: {stage_id}", location=f"{base}.stage_id")
        seen_ids.add(stage_id)

        order = raw_stage.get("order")
        _require_type(_is_int(order), "order 必须为整数", f"{base}.order")
        orders.append((order, index, stage_id))

        duration = raw_stage.get("duration_ms")
        _require_type(_is_int(duration) and duration >= 0, "duration_ms 必须为非负整数", f"{base}.duration_ms")

        prereqs = raw_stage.get("prerequisites", [])
        _require_type(isinstance(prereqs, list), "prerequisites 必须为数组", f"{base}.prerequisites")
        for prereq_index, prereq in enumerate(prereqs):
            _require_type(isinstance(prereq, str) and prereq, "前置阶段引用必须为非空字符串",
                          f"{base}.prerequisites[{prereq_index}]")

        actions = raw_stage.get("actions", [])
        _require_type(isinstance(actions, list), "actions 必须为数组", f"{base}.actions")
        for action_index, action in enumerate(actions):
            action_base = f"{base}.actions[{action_index}]"
            _require_type(isinstance(action, dict), "动作必须是对象", action_base)
            action_id = action.get("action_id")
            _require_type(isinstance(action_id, str) and action_id, "action_id 必须为非空字符串",
                          f"{action_base}.action_id")

        criterion = raw_stage.get("criterion", {})
        _require_type(isinstance(criterion, dict), "criterion 必须为对象", f"{base}.criterion")
        mode = criterion.get("mode", "all_of")
        _require_type(mode in CRITERION_MODES, f"判据模式必须是 {CRITERION_MODES} 之一",
                      f"{base}.criterion.mode")
        required = criterion.get("required_actions")
        if required is not None:
            _require_type(isinstance(required, list) and all(isinstance(item, str) for item in required),
                          "required_actions 必须为字符串数组", f"{base}.criterion.required_actions")

    sorted_orders = sorted(orders)
    expected = 1
    for order, index, _stage_id in sorted_orders:
        if order != expected:
            raise PlanValidationError(
                f"序号必须从 1 开始连续且无重复，期望 {expected} 实际 {order}",
                location=f"$.stages[{index}].order",
            )
        expected += 1

    stage_ids = {stage["stage_id"] for stage in stages}
    for index, raw_stage in enumerate(stages):
        for prereq_index, prereq in enumerate(raw_stage.get("prerequisites", [])):
            if prereq not in stage_ids:
                raise PlanValidationError(
                    f"前置阶段引用不存在: {prereq}",
                    location=f"$.stages[{index}].prerequisites[{prereq_index}]",
                )

    # 成环单独给出 E_CYCLE，由 graph 层检测并抛出。
    from .graph import ensure_acyclic

    ensure_acyclic(tuple(Stage.from_dict(stage) for stage in stages))
    return True


def validate_event_list(raw_events, stage_ids):
    _require_type(isinstance(raw_events, list), "注入事件清单必须为数组", "$.events")
    seen = set()
    for index, raw_event in enumerate(raw_events):
        base = f"$.events[{index}]"
        _require_type(isinstance(raw_event, dict), "注入事件必须是对象", base)
        event_id = raw_event.get("event_id")
        _require_type(isinstance(event_id, str) and event_id, "event_id 必须为非空字符串", f"{base}.event_id")
        if event_id in seen:
            raise PlanValidationError(f"注入事件标识重复: {event_id}", location=f"{base}.event_id")
        seen.add(event_id)
        kind = raw_event.get("kind")
        _require_type(kind in INJECTION_KINDS, f"kind 必须是 {INJECTION_KINDS} 之一", f"{base}.kind")
        offset = raw_event.get("offset_ms")
        _require_type(_is_int(offset) and offset >= 0, "offset_ms 必须为非负整数", f"{base}.offset_ms")
        target = raw_event.get("target_stage")
        _require_type(isinstance(target, str) and target, "target_stage 必须为非空字符串",
                      f"{base}.target_stage")
        if target not in stage_ids:
            raise PlanValidationError(
                f"注入事件作用阶段不存在: {target}", location=f"{base}.target_stage"
            )
    return True


def apply_criterion(criterion, action_ids, failed_actions):
    mode = criterion.get("mode", "all_of")
    required = criterion.get("required_actions")
    failed_set = set(failed_actions)
    if mode == "all_of":
        if required is None:
            return not failed_set
        return not failed_set.intersection(required)
    candidate_actions = list(action_ids) if required is None else required
    return any(action_id not in failed_set for action_id in candidate_actions)
