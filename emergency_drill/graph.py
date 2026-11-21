"""阶段依赖图：稳定拓扑序、环检测与失效下游传播。"""

from collections import deque

from .errors import CycleError


def _stage_map(stages):
    return {stage.stage_id: stage for stage in stages}


def topological_order(stages):
    """Kahn 算法；同层按 order 再按 stage_id 排序，结果确定。

    成环时抛出 E_CYCLE，并把环上稳定排序后的阶段写入上下文。
    """
    stages = tuple(stages)
    mapping = _stage_map(stages)
    indegree = {stage.stage_id: 0 for stage in stages}
    dependents = {stage.stage_id: [] for stage in stages}
    for stage in stages:
        for prereq in stage.prerequisites:
            indegree[stage.stage_id] += 1
            dependents[prereq].append(stage.stage_id)

    def ready_key(sid):
        stage = mapping[sid]
        return stage.order, sid

    ready = deque(sorted((sid for sid, deg in indegree.items() if deg == 0), key=ready_key))
    order_result = []
    while ready:
        current = ready.popleft()
        order_result.append(current)
        for child in sorted(dependents[current], key=ready_key):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready = deque(sorted(ready, key=ready_key))

    if len(order_result) != len(mapping):
        remaining = sorted(sid for sid, deg in indegree.items() if deg > 0)
        raise CycleError(
            "阶段前置引用图存在环，无法建立拓扑序",
            location="$.stages[*].prerequisites",
            cycle=remaining,
        )
    return order_result


def ensure_acyclic(stages):
    topological_order(stages)
    return True


def downstream_map(stages):
    """返回 stage_id -> 直接下游 stage_id 列表（稳定排序）。"""
    mapping = _stage_map(stages)
    result = {stage.stage_id: [] for stage in stages}
    for stage in stages:
        for prereq in stage.prerequisites:
            result[prereq].append(stage.stage_id)
    for sid in result:
        result[sid] = sorted(result[sid], key=lambda child: (mapping[child].order, child))
    return result


def transitive_downstream(stages, roots):
    """沿依赖图求全部直接与间接下游，返回稳定拓扑序序列。"""
    topo = topological_order(stages)
    direct = downstream_map(stages)
    root_set = set(roots)
    reached = set()
    queue = deque(roots)
    while queue:
        current = queue.popleft()
        for child in direct.get(current, ()):
            if child not in reached and child not in root_set:
                reached.add(child)
                queue.append(child)
    return [sid for sid in topo if sid in reached]


def propagation_trace(stages, changed_set):
    """生成每一步失效传播依据，按拓扑序稳定输出。

    每项形如::

        {"stage_id": "S2", "reached_from": ["S1"], "relation": "downstream"}
    """
    mapping = _stage_map(stages)
    direct = downstream_map(stages)
    parents = {stage.stage_id: sorted(stage.prerequisites) for stage in stages}
    topo = topological_order(stages)
    reached = set(changed_set)
    trace = []
    for sid in topo:
        if sid in changed_set:
            trace.append({"stage_id": sid, "reached_from": [], "relation": "changed"})
            continue
        incoming = [parent for parent in parents[sid] if parent in reached]
        if incoming:
            reached.add(sid)
            trace.append(
                {
                    "stage_id": sid,
                    "reached_from": incoming,
                    "relation": "downstream",
                }
            )
    return trace


def stable_skipped_path(stages, failed_stage, target_stage):
    """失败阶段到被跳过阶段之间的稳定溯源链（含两端）。"""
    direct = downstream_map(stages)
    queue = deque([(failed_stage, [failed_stage])])
    seen = {failed_stage}
    solutions = []
    while queue:
        current, path = queue.popleft()
        if current == target_stage:
            solutions.append(path)
            continue
        for child in direct.get(current, ()):
            if child not in seen:
                seen.add(child)
                queue.append((child, path + [child]))
    if not solutions:
        return []
    return sorted(solutions, key=lambda path: (len(path), path))[0]


def referrers_of(stages, target_id):
    """稳定列出直接把 target_id 作为前置的阶段。"""
    result = [
        stage.stage_id
        for stage in sorted(stages, key=lambda item: (item.order, item.stage_id))
        if target_id in stage.prerequisites
    ]
    return result
