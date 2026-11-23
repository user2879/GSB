"""预算参数 / 节点集合演进：整体替换与陈旧结论防护（需求 6）。

规则：
  - 带宽配额、全局预算、并发上限或节点集合任一修改 => 版本号单调递增。
  - 沿 节点 -> 块 -> 批次 引用关系推导受影响的排队批次与被延误块集合。
  - 旧版本的调度轨迹、紧迫度排序、误差归因整体标记为陈旧（stale），
    不参与当前展示；当前视图整体替换而非增量合并。
  - 稳定列出新增、删除与变更节点及受影响块集合（均按标识排序，顺序无关）。
"""

from __future__ import annotations

from .spec import GroupSpec


def diff_nodes(old: GroupSpec, new: GroupSpec) -> dict:
    old_map = {n.node_id: n for n in old.nodes}
    new_map = {n.node_id: n for n in new.nodes}
    added = sorted(set(new_map) - set(old_map))
    removed = sorted(set(old_map) - set(new_map))
    changed = sorted(
        nid
        for nid in set(old_map) & set(new_map)
        if old_map[nid] != new_map[nid]
    )
    return {"added": added, "removed": removed, "changed": changed}


def params_changed(old: GroupSpec, new: GroupSpec) -> list:
    fields = []
    op, np_ = old.params, new.params
    if op.concurrency_limit != np_.concurrency_limit:
        fields.append("concurrency_limit")
    if op.global_budget_bytes != np_.global_budget_bytes:
        fields.append("global_budget_bytes")
    if op.segment_bytes != np_.segment_bytes:
        fields.append("segment_bytes")
    if op.scale_exponent != np_.scale_exponent:
        fields.append("scale_exponent")
    bw_old = {n.node_id: n.bandwidth_kbps for n in old.nodes}
    bw_new = {n.node_id: n.bandwidth_kbps for n in new.nodes}
    if bw_old != bw_new:
        fields.append("bandwidth_kbps")
    return fields


def impact_analysis(old: GroupSpec, new: GroupSpec, last_run: dict | None) -> dict:
    """沿引用关系推导受影响排队批次与被延误块集合。"""
    node_diff = diff_nodes(old, new)
    affected_node_ids = set(node_diff["removed"]) | set(node_diff["changed"])
    budget_touched = bool(params_changed(old, new))

    affected_batches: list[str] = []
    delayed_blocks: set[str] = set()
    if last_run:
        for batch in last_run.get("batches", []):
            if batch["state"] != "QUEUED":
                continue
            if batch["node_id"] in affected_node_ids or budget_touched:
                affected_batches.append(batch["batch_id"])
                delayed_blocks.add(batch["block_id"])
    # 被删除节点上的块整体延误（即使尚无批次记录）。
    removed = set(node_diff["removed"])
    for block in old.blocks:
        if block.node_id in removed:
            delayed_blocks.add(block.block_id)

    return {
        "nodes": node_diff,
        "param_fields_changed": params_changed(old, new),
        "affected_queued_batches": sorted(affected_batches),
        "delayed_blocks": sorted(delayed_blocks),
    }


class StaleArchive:
    """旧版本结论存档：整体标记陈旧，仅供审计，不参与当前展示。"""

    def __init__(self):
        self._entries: list[dict] = []

    def archive(self, version: int, view: dict) -> None:
        self._entries.append({"version": version, "stale": True, "view": view})

    @property
    def entries(self) -> list:
        return list(self._entries)

    def latest_stale_version(self) -> int | None:
        return self._entries[-1]["version"] if self._entries else None
