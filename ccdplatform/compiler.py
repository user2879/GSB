"""配置项编译器：定点量化 + 误差传播 + 增量编译 + 分发计划切片。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .fixedpoint import (
    GridSpec,
    derive_quantized,
    propagated_error_bound,
    trunc_zero,
)
from .graph import DependencyGraph
from .model import (
    CompiledItem,
    ConfigItem,
    DispatchPlan,
    ManifestVersion,
)


def item_fingerprint(item: ConfigItem) -> bytes:
    """变更检测指纹：取值、派生引用/系数、目标集合。"""
    payload = {
        "value": item.value,
        "targets": list(item.targets),
        "derivation": item.derivation.to_json() if item.derivation else None,
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


@dataclass
class CompileResult:
    version: int
    compiled: Dict[str, CompiledItem]
    graph: DependencyGraph
    plan: DispatchPlan
    recompiled: List[str]
    unchanged: List[str]
    invalidated: List[str]
    deleted: List[str]
    fingerprints: Dict[str, bytes]
    canonical: Dict[str, bytes]
    change_reasons: Dict[str, str] = field(default_factory=dict)


class Compiler:
    """带记忆的增量编译器。

    保存上一版本的指纹与编译产物字节，仅对变化项及其下游重新编译，
    未变化项原样复用旧 :class:`CompiledItem`（逐字节不变）。
    """

    def __init__(self, grid: GridSpec):
        self.grid = grid
        self._prev_fingerprints: Dict[str, bytes] = {}
        self._prev_compiled: Dict[str, CompiledItem] = {}
        self._prev_graph: Optional[DependencyGraph] = None

    # ----------------------------------------------------------- single item
    def _compile_base(self, item: ConfigItem) -> CompiledItem:
        q = self.grid.quantize(item.value, key=item.key)
        return CompiledItem(
            key=item.key,
            source_value=item.value,
            quantized=q,
            is_derived=False,
            coef_quantized=(),
            targets=tuple(item.targets),
            error_bound=1,  # 单次向零量化：1 个整数格
            contributions=(),
            propagation_path=(item.key,),
        )

    def _compile_derived(
        self,
        item: ConfigItem,
        item_map: Dict[str, ConfigItem],
        compiled: Dict[str, CompiledItem],
    ) -> CompiledItem:
        assert item.derivation is not None
        coef_q: List[Tuple[str, int]] = []
        for ref_key, coef in item.derivation.refs:
            cq = trunc_zero(int(abs(coef) * self.grid.scale), 1)
            cq = cq if coef >= 0 else -cq
            coef_q.append((ref_key, cq))
        upstream_q = {k: compiled[k].quantized for k, _ in coef_q}
        q = derive_quantized(coef_q, upstream_q, self.grid.scale)
        if q > self.grid.qmax or q < self.grid.qmin:
            from .errors import RangeError

            raise RangeError(
                f"派生项 {item.key} 量化值 {q} 超出定点范围",
                {"key": item.key, "quantized": q, "scale": self.grid.scale},
            )
        upstream_bounds = {k: compiled[k].error_bound for k, _ in coef_q}
        bound, contributions = propagated_error_bound(
            coef_q, upstream_bounds, upstream_q, self.grid.scale
        )
        # 传播路径：当前项 + 各上游路径（保序去重），形成叶 -> 当前项的完整链
        seen: set[str] = set()
        path: List[str] = []
        for ref_key, _ in coef_q:
            for node in compiled[ref_key].propagation_path:
                if node not in seen:
                    seen.add(node)
                    path.append(node)
        if item.key not in seen:
            path.append(item.key)
        return CompiledItem(
            key=item.key,
            source_value=item.value,
            quantized=q,
            is_derived=True,
            coef_quantized=tuple(coef_q),
            targets=tuple(item.targets),
            error_bound=bound,
            contributions=tuple(sorted(contributions.items())),
            propagation_path=tuple(path),
        )

    # ----------------------------------------------------------------- compile
    def compile_version(
        self, manifest: ManifestVersion
    ) -> CompileResult:
        item_map = manifest.item_map()
        graph = DependencyGraph.from_items(item_map)
        order = graph.topological_order()

        # 删除项的 E_REF 检查必须发生在编译前
        deleted = [k for k in self._prev_fingerprints if k not in item_map]
        if deleted and self._prev_graph is not None:
            self._prev_graph.assert_deletable(deleted, item_map)

        # 直接变化项（键值/引用/系数/目标变化）
        changed: List[str] = []
        reasons: Dict[str, str] = {}
        fingerprints: Dict[str, bytes] = {}
        for key in order:
            fp = item_fingerprint(item_map[key])
            fingerprints[key] = fp
            if key not in self._prev_fingerprints:
                changed.append(key)
                reasons[key] = "new"
            elif self._prev_fingerprints[key] != fp:
                changed.append(key)
                reasons[key] = "value/ref/coef/targets changed"

        # 下游失效（直接 + 间接，稳定拓扑序）
        invalidated = graph.downstream_closure(changed) if changed else []
        for key in invalidated:
            reasons.setdefault(key, "downstream invalidated")
        recompile_set = set(changed) | set(invalidated)

        compiled: Dict[str, CompiledItem] = {}
        canonical: Dict[str, bytes] = {}
        recompiled: List[str] = []
        unchanged: List[str] = []
        for key in order:
            item = item_map[key]
            if key not in recompile_set and key in self._prev_compiled:
                # 逐字节复用旧产物
                old = self._prev_compiled[key]
                compiled[key] = old
                canonical[key] = old.canonical_bytes()
                unchanged.append(key)
                continue
            if item.is_derived:
                built = self._compile_derived(item, item_map, compiled)
            else:
                built = self._compile_base(item)
            compiled[key] = built
            canonical[key] = built.canonical_bytes()
            recompiled.append(key)

        plan = self._build_plan(manifest, item_map, graph)

        result = CompileResult(
            version=manifest.version,
            compiled=compiled,
            graph=graph,
            plan=plan,
            recompiled=recompiled,
            unchanged=unchanged,
            invalidated=invalidated,
            deleted=deleted,
            fingerprints=fingerprints,
            canonical=canonical,
            change_reasons=reasons,
        )
        # 记忆推进
        self._prev_fingerprints = fingerprints
        self._prev_compiled = dict(compiled)
        self._prev_graph = graph
        return result

    def _build_plan(
        self,
        manifest: ManifestVersion,
        item_map: Dict[str, ConfigItem],
        graph: DependencyGraph,
    ) -> DispatchPlan:
        """依赖感知切片。

        节点按批独立重算派生摘要，跨批引用会让节点拿不到上游，因此：

        * 基础项（无派生）按 ``batch_size`` 正常切片；
        * 每个派生项携带其在本节点目标集合内的完整传递上游闭包，独立成批，
          批次可能超过 batch_size（正确性优先于批大小）；
        * 节点对重复收到的键以最后一次取值为准（幂等覆盖）。
        """
        batch_size = manifest.strategy.batch_size
        node_items: Dict[str, List[str]] = {}
        for key in graph.topological_order():
            for node in item_map[key].targets:
                node_items.setdefault(node, []).append(key)
        node_batches: Dict[str, List[List[str]]] = {}
        for node in sorted(node_items):
            keys = node_items[node]
            key_set = set(keys)
            closures: Dict[str, set[str]] = {}
            for key in keys:
                closure: set[str] = set()
                stack = (
                    list(item_map[key].derivation.ref_keys)
                    if item_map[key].derivation
                    else []
                )
                while stack:
                    cur = stack.pop()
                    # 只闭包该节点实际收到的上游；未下发到本节点的引用忽略
                    if cur not in key_set or cur in closure:
                        continue
                    closure.add(cur)
                    cur_item = item_map[cur]
                    if cur_item.derivation:
                        stack.extend(
                            ref
                            for ref in cur_item.derivation.ref_keys
                            if ref in key_set
                        )
                closures[key] = closure

            batches: List[List[str]] = []
            base_buffer: List[str] = []
            for key in keys:
                if closures[key]:
                    if base_buffer:
                        batches.append(base_buffer)
                        base_buffer = []
                    # 拓扑序的完整闭包 + 派生项自身，依赖在前
                    batches.append(
                        [d for d in keys if d in closures[key]] + [key]
                    )
                else:
                    base_buffer.append(key)
                    if len(base_buffer) >= batch_size:
                        batches.append(base_buffer)
                        base_buffer = []
            if base_buffer:
                batches.append(base_buffer)
            node_batches[node] = batches
        return DispatchPlan(
            version=manifest.version,
            node_batches=node_batches,
            strategy=manifest.strategy,
        )

    # ------------------------------------------------------------- restore API
    def restore_state(
        self,
        fingerprints: Dict[str, bytes],
        compiled: Dict[str, CompiledItem],
        graph: DependencyGraph,
    ) -> None:
        """从快照整体替换编译器内存状态（非增量合并）。"""
        self._prev_fingerprints = dict(fingerprints)
        self._prev_compiled = dict(compiled)
        self._prev_graph = graph


def recompile_from_manifest(
    manifest: ManifestVersion, grid: GridSpec
) -> CompileResult:
    """无记忆全量重编译：用于恢复后的重放校验。"""
    fresh = Compiler(grid)
    return fresh.compile_version(manifest)
