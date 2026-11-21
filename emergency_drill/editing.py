"""预案增量编辑：变更集应用、E_DEPEND 引用保护与失效传播。"""

from dataclasses import replace

from .errors import DependencyError, PlanValidationError
from .graph import propagation_trace, referrers_of, topological_order, transitive_downstream
from .models import Stage, canonical_json, validate_plan_dict


def _stage_dicts(stages):
    return [stage.to_dict() for stage in stages]


def _rebuild_plan(plan, stage_dicts, new_version):
    validate_plan_dict({"plan_id": plan.plan_id, "stages": stage_dicts})
    return plan.with_version(tuple(Stage.from_dict(item) for item in stage_dicts), new_version)


def _guard_delete(stages, target_id):
    referrers = referrers_of(stages, target_id)
    if referrers:
        raise DependencyError(
            f"阶段 {target_id} 仍被其他阶段引用，禁止删除",
            location=f"changeset.delete_stage.{target_id}",
            stage_id=target_id,
            referrers=referrers,
        )


def apply_changeset(plan, changeset, new_version=None):
    """应用变更集并返回新预案、失效传播依据与失效集合（稳定拓扑序）。"""
    stages = list(plan.stages)
    mapping = {stage.stage_id: stage for stage in stages}
    changed_roots = []
    next_version = plan.version + 1 if new_version is None else new_version

    operations = changeset.get("operations", [])
    if not isinstance(operations, list):
        raise PlanValidationError("changeset.operations 必须为数组", location="changeset.operations")

    for index, operation in enumerate(operations):
        op = operation.get("op")
        base = f"changeset.operations[{index}]"
        if op == "add_stage":
            raw_stage = operation["stage"]
            if raw_stage["stage_id"] in mapping:
                raise PlanValidationError(
                    f"新增阶段标识已存在: {raw_stage['stage_id']}",
                    location=f"{base}.stage.stage_id",
                )
            stages.append(Stage.from_dict(raw_stage))
            mapping[raw_stage["stage_id"]] = stages[-1]
            changed_roots.append(raw_stage["stage_id"])
        elif op == "update_duration":
            target = operation["stage_id"]
            if target not in mapping:
                raise PlanValidationError(f"待修改阶段不存在: {target}", location=f"{base}.stage_id")
            duration = operation["duration_ms"]
            if not isinstance(duration, int) or isinstance(duration, bool) or duration < 0:
                raise PlanValidationError("duration_ms 必须为非负整数", location=f"{base}.duration_ms")
            mapping[target] = replace(mapping[target], duration_ms=duration)
            changed_roots.append(target)
        elif op == "rewrite_criterion":
            target = operation["stage_id"]
            if target not in mapping:
                raise PlanValidationError(f"待修改阶段不存在: {target}", location=f"{base}.stage_id")
            criterion = operation["criterion"]
            if not isinstance(criterion, dict):
                raise PlanValidationError("criterion 必须为对象", location=f"{base}.criterion")
            mapping[target] = replace(mapping[target], criterion=dict(criterion))
            changed_roots.append(target)
        elif op == "delete_stage":
            target = operation["stage_id"]
            if target not in mapping:
                raise PlanValidationError(f"待删除阶段不存在: {target}", location=f"{base}.stage_id")
            _guard_delete(tuple(mapping.values()), target)
            del mapping[target]
            changed_roots.append(target)
        elif op == "adjust_prereqs":
            target = operation["stage_id"]
            if target not in mapping:
                raise PlanValidationError(f"待调整阶段不存在: {target}", location=f"{base}.stage_id")
            prereqs = operation["prerequisites"]
            if not isinstance(prereqs, list) or not all(isinstance(item, str) for item in prereqs):
                raise PlanValidationError("prerequisites 必须为字符串数组", location=f"{base}.prerequisites")
            mapping[target] = replace(mapping[target], prerequisites=tuple(prereqs))
            changed_roots.append(target)
        else:
            raise PlanValidationError(f"未知变更操作: {op}", location=f"{base}.op")

    ordered_ids = topological_order(tuple(mapping.values()))
    new_stage_list = []
    for new_order, sid in enumerate(ordered_ids, start=1):
        stage = mapping[sid]
        new_stage_list.append(replace(stage, order=new_order) if stage.order != new_order else stage)
    new_plan = _rebuild_plan(plan, _stage_dicts(new_stage_list), next_version)

    surviving_roots = [sid for sid in changed_roots if sid in new_plan.stage_map]
    downstream = transitive_downstream(new_plan.stages, surviving_roots)
    invalid_order = [sid for sid in topological_order(new_plan.stages)
                     if sid in set(surviving_roots).union(downstream)]
    trace = propagation_trace(new_plan.stages, set(surviving_roots))
    return {
        "plan": new_plan,
        "invalid_stage_ids": invalid_order,
        "propagation": trace,
        "deleted_stage_ids": sorted(sid for sid in changed_roots if sid not in new_plan.stage_map),
        "old_version": plan.version,
        "new_version": new_plan.version,
    }


def assert_frozen_identical(old_outcomes, frozen_stage_ids, partial_outcomes):
    """逐字节校验未受影响阶段结论完全保持不变。"""
    from .errors import IncrementalMismatchError

    for stage_id in sorted(frozen_stage_ids):
        before = old_outcomes[stage_id]
        after = partial_outcomes[stage_id]
        before_cmp = {key: value for key, value in before.items() if key != "plan_version"}
        after_cmp = {key: value for key, value in after.items() if key != "plan_version"}
        if canonical_json(before_cmp) != canonical_json(after_cmp):
            for field in sorted(set(before_cmp) | set(after_cmp)):
                if canonical_json(before.get(field)) != canonical_json(after.get(field)):
                    raise IncrementalMismatchError(
                        "未受影响阶段结论发生变化",
                        location=f"stage={stage_id}.field={field}",
                        stage_id=stage_id,
                        field=field,
                        incremental=after.get(field),
                        previous=before.get(field),
                    )
    return True
