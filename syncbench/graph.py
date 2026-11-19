"""从采集参数到派生节点的依赖图。

节点（每通道一套）：
- drift:c       漂移曲线/分段拟合（含延迟估计）
- gapfill:c     音视频空洞填补
- map:c         统一时间轴映射
- aalign:c      音频对齐片段
- valign:c      视频对齐片段
- tile:c        可视化投影瓦片

边携带触发失效的参数键（drift_model / delay_compensation /
gap_policy / reference_channel）。单参数修改只重算失效闭包，
闭包外节点复用并记录 reuse_source；引擎保证闭包清单与实际重算集合一致，
请求已失效节点会抛 E_DEPENDENCY_STALE。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    PARAM_DELAY_COMP,
    PARAM_DRIFT_MODEL,
    PARAM_GAP_POLICY,
    PARAM_REFERENCE,
)
from .errors import E_DEPENDENCY_STALE, BusinessError

NODE_DRIFT = "drift"
NODE_GAPFILL = "gapfill"
NODE_MAP = "map"
NODE_AALIGN = "aalign"
NODE_VALIGN = "valign"
NODE_TILE = "tile"

NODE_ORDER = (NODE_DRIFT, NODE_GAPFILL, NODE_MAP, NODE_AALIGN,
              NODE_VALIGN, NODE_TILE)


def node_id(kind: str, channel: int) -> str:
    return f"{kind}:{channel}"


def parse_node(nid: str) -> tuple[str, int]:
    kind, ch = nid.split(":")
    return kind, int(ch)


# 依赖边：节点 -> [(上游节点, 触发参数键集合)]
# drift/gapfill 不依赖参考通道；map 及对齐/瓦片依赖参考通道。
EDGES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    NODE_DRIFT: (),
    NODE_GAPFILL: (
        (NODE_DRIFT, (PARAM_DRIFT_MODEL,)),
    ),
    NODE_MAP: (
        (NODE_DRIFT, (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP)),
    ),
    NODE_AALIGN: (
        (NODE_MAP, (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP, PARAM_REFERENCE)),
    ),
    NODE_VALIGN: (
        (NODE_MAP, (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP, PARAM_REFERENCE)),
    ),
    NODE_TILE: (
        (NODE_AALIGN, (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP,
                       PARAM_GAP_POLICY, PARAM_REFERENCE)),
        (NODE_VALIGN, (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP,
                       PARAM_GAP_POLICY, PARAM_REFERENCE)),
        (NODE_GAPFILL, (PARAM_GAP_POLICY,)),
    ),
}

# 节点自身直接绑定的参数键（参数变化即令该通道节点失效，参考键为全局）
NODE_PARAM_BINDINGS: dict[str, tuple[str, ...]] = {
    NODE_DRIFT: (PARAM_DRIFT_MODEL,),
    NODE_GAPFILL: (PARAM_GAP_POLICY, PARAM_DRIFT_MODEL),
    NODE_MAP: (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP, PARAM_REFERENCE),
    NODE_AALIGN: (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP, PARAM_REFERENCE),
    NODE_VALIGN: (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP, PARAM_REFERENCE),
    NODE_TILE: (PARAM_DRIFT_MODEL, PARAM_DELAY_COMP, PARAM_GAP_POLICY,
                PARAM_REFERENCE),
}


@dataclass
class NodeState:
    nid: str
    kind: str
    channel: int
    version: int = 0
    content_hash: str | None = None
    param_version: str | None = None
    blob_hashes: tuple[str, ...] = ()
    payload: Any = None
    status: str = "absent"          # absent | ok | stale
    reuse_source: str | None = None # 复用时记录来源节点与版本
    recompute_count: int = 0

    def to_manifest(self) -> dict[str, Any]:
        return {
            "node": self.nid,
            "kind": self.kind,
            "channel": self.channel,
            "version": self.version,
            "content_hash": self.content_hash,
            "param_version": self.param_version,
            "blob_hashes": list(self.blob_hashes),
            "status": self.status,
            "reuse_source": self.reuse_source,
            "recompute_count": self.recompute_count,
        }


@dataclass
class Change:
    """一次增量参数修改。"""

    channel: int | None      # None 表示全局（参考通道）
    key: str
    old_value: Any
    new_value: Any

    def affected_scope(self) -> tuple[set[int], bool]:
        """返回 (受影响通道集合, 是否全局参考切换)。"""
        if self.key == PARAM_REFERENCE:
            return set(range(4)), True
        assert self.channel is not None
        return {self.channel}, False


class DependencyGraph:
    """维护 4 通道全部节点状态与失效闭包。"""

    def __init__(self, n_channels: int = 4) -> None:
        self.n_channels = n_channels
        self.nodes: dict[str, NodeState] = {}
        for c in range(n_channels):
            for kind in NODE_ORDER:
                nid = node_id(kind, c)
                self.nodes[nid] = NodeState(nid, kind, c)
        self.changes: list[dict[str, Any]] = []

    # -- 闭包计算 -------------------------------------------------------------
    def closure(self, change: Change) -> set[str]:
        """精确计算单参数修改的失效闭包。"""
        channels, global_ref = change.affected_scope()
        invalid: set[str] = set()
        # 1) 直接绑定该参数键的节点失效
        for c in channels:
            for kind, keys in NODE_PARAM_BINDINGS.items():
                if change.key in keys:
                    invalid.add(node_id(kind, c))
        # 2) 沿边向下游传播（仅当边的触发键包含该参数键）
        changed = set(invalid)
        while changed:
            nxt: set[str] = set()
            for nid in changed:
                kind, ch = parse_node(nid)
                for down_kind, edge_keys in EDGES.items():
                    for up_kind, trigger_keys in edge_keys:
                        if up_kind == kind and change.key in trigger_keys:
                            down_id = node_id(down_kind, ch)
                            if down_id not in invalid:
                                nxt.add(down_id)
                                invalid.add(down_id)
            changed = nxt
        # 参考切换：参考通道自身的 drift/gapfill 不在闭包内（与参考无关）
        if global_ref:
            for c in range(self.n_channels):
                invalid.discard(node_id(NODE_DRIFT, c))
                invalid.discard(node_id(NODE_GAPFILL, c))
        return invalid

    def mark_stale(self, closure: set[str]) -> None:
        for nid in closure:
            self.nodes[nid].status = "stale"

    def record_recompute(self, nid: str, content_hash: str, param_version: str,
                         blob_hashes: tuple[str, ...], payload: Any) -> None:
        st = self.nodes[nid]
        st.version += 1
        st.recompute_count += 1
        st.content_hash = content_hash
        st.param_version = param_version
        st.blob_hashes = tuple(blob_hashes)
        st.payload = payload
        st.status = "ok"
        st.reuse_source = None

    def record_reuse(self, nid: str, source_nid: str) -> None:
        st = self.nodes[nid]
        src = self.nodes[source_nid]
        st.reuse_source = f"{source_nid}@v{src.version}"
        st.status = "ok"

    def get_live(self, nid: str) -> NodeState:
        st = self.nodes[nid]
        if st.status != "ok":
            raise BusinessError(E_DEPENDENCY_STALE,
                                f"节点 {nid} 状态为 {st.status}，不可读取",
                                channel=st.channel)
        return st

    def assert_closure_matches(self, closure: set[str],
                               recomputed: set[str]) -> None:
        """闭包清单必须与实际重算集合完全一致。"""
        if closure != recomputed:
            missing = sorted(closure - recomputed)
            extra = sorted(recomputed - closure)
            raise BusinessError(
                E_DEPENDENCY_STALE,
                "失效闭包与实际重算集合不一致",
                detail={"missing": missing, "extra": extra})

    def reusable_hashes(self) -> dict[str, str]:
        return {nid: st.content_hash for nid, st in self.nodes.items()
                if st.status == "ok" and st.content_hash}

    def manifest(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[node_id(k, c)].to_manifest()
                      for c in range(self.n_channels) for k in NODE_ORDER],
            "changes": list(self.changes),
        }
