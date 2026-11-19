"""依赖图与精确失效闭包。

节点类型
--------
- ``tile``        地形格（源节点，不被失效，只产生变更事件）；
- ``flow``        区域流场，依赖区域内每个 tile；
- ``jump``        区域边界跳点（门户）表，仅依赖边界 tile；
- ``path``        路径缓存，依赖其经过区域的 flow，并显式引用其使用的 jump。

有向边 ``source -> dependent`` 携带触发条件（tile 变更 / unit 变更）。
失效沿“flow -> path”正向传播；``jump`` 失效不传播到 flow（方向不可颠倒）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Set

from .encoding import content_hash
from .errors import ClosureMismatchError, DependencyStaleError
from .grid import Cell, TacticalGrid

TILE = "tile"
FLOW = "flow"
JUMP = "jump"
PATH = "path"

TRIGGER_TILE = "tile_change"
TRIGGER_UNIT = "unit_change"


@dataclass(frozen=True)
class DepEdge:
    source: str
    dependent: str
    trigger: str
    reason: str

    def key(self) -> str:
        return f"{self.source}->{self.dependent}#{self.trigger}"


@dataclass
class DepNode:
    node_id: str
    kind: str
    content_hash: str
    alive: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClosureResult:
    changed_tiles: List[Cell]
    invalidated_flow_regions: List[int]
    invalidated_jump_portals: List[str]
    invalidated_path_queries: List[str]
    reused_flows: List[int]
    reused_jump_portals: List[str]
    reused_path_queries: List[str]
    edges_fired: List[str]
    boundary_hits: List[Cell]
    interior_hits: List[Cell]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "changed_tiles": [list(cell) for cell in self.changed_tiles],
            "invalidated_flow_regions": self.invalidated_flow_regions,
            "invalidated_jump_portals": self.invalidated_jump_portals,
            "invalidated_path_queries": self.invalidated_path_queries,
            "reused_flows": self.reused_flows,
            "reused_jump_portals": self.reused_jump_portals,
            "reused_path_queries": self.reused_path_queries,
            "edges_fired": self.edges_fired,
            "boundary_hits": [list(cell) for cell in self.boundary_hits],
            "interior_hits": [list(cell) for cell in self.interior_hits],
        }


def tile_id(cell: Cell) -> str:
    return f"tile:{cell[0]:04d}:{cell[1]:04d}"


def flow_id(region_id: int) -> str:
    return f"flow:R{region_id:03d}"


def jump_id(portal_id: str) -> str:
    return f"jump:{portal_id}"


def path_id(query_key: str) -> str:
    return f"path:{query_key}"


class DependencyGraph:
    def __init__(self, grid: TacticalGrid) -> None:
        self.grid = grid
        self.nodes: Dict[str, DepNode] = {}
        self.edges: Dict[str, DepEdge] = {}
        self.path_routes: Dict[str, List[Cell]] = {}
        self.path_regions: Dict[str, List[int]] = {}
        self.path_portals: Dict[str, List[str]] = {}
        self.unit_anchors: Dict[str, Cell] = {}
        self.version = 0

    # ---------- 节点管理 ----------
    def upsert_node(self, node_id: str, kind: str, digest: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.kind = kind
            node.content_hash = digest
            node.alive = True
            node.meta = meta or node.meta
        else:
            self.nodes[node_id] = DepNode(node_id=node_id, kind=kind, content_hash=digest, meta=meta or {})

    def get_node(self, node_id: str) -> DepNode:
        if node_id not in self.nodes:
            raise DependencyStaleError("依赖图缺失节点", missing_node=node_id)
        return self.nodes[node_id]

    def verify_hash(self, node_id: str, expected_hash: str) -> None:
        node = self.get_node(node_id)
        if node.content_hash != expected_hash:
            raise DependencyStaleError(
                "节点内容哈希不匹配",
                node_id=node_id,
                stored=node.content_hash,
                expected=expected_hash,
            )

    def alive_node_ids(self, kind: Optional[str] = None) -> List[str]:
        return sorted(
            node_id
            for node_id, node in self.nodes.items()
            if node.alive and (kind is None or node.kind == kind)
        )

    def add_edge(self, source: str, dependent: str, trigger: str, reason: str) -> None:
        edge = DepEdge(source=source, dependent=dependent, trigger=trigger, reason=reason)
        self.edges[edge.key()] = edge

    def outgoing(self, node_id: str, trigger: Optional[str] = None) -> List[DepEdge]:
        result = [
            edge
            for edge in self.edges.values()
            if edge.source == node_id and (trigger is None or edge.trigger == trigger)
        ]
        return sorted(result, key=lambda edge: edge.key())

    # ---------- 建图 ----------
    def seed_tile_nodes(self, tile_hashes: Dict[Cell, str]) -> None:
        for cell in sorted(tile_hashes):
            self.upsert_node(tile_id(cell), TILE, tile_hashes[cell], {"cell": list(cell)})

    def register_flow(self, region_id: int, digest: str, region_cells: Sequence[Cell]) -> None:
        fid = flow_id(region_id)
        self.upsert_node(fid, FLOW, digest, {"region_id": region_id})
        for cell in region_cells:
            self.add_edge(tile_id(cell), fid, TRIGGER_TILE, f"tile in region {region_id}")

    def register_jump(self, portal_id: str, digest: str, cells: Sequence[Cell]) -> None:
        jid = jump_id(portal_id)
        self.upsert_node(jid, JUMP, digest, {"portal_id": portal_id, "cells": [list(c) for c in cells]})
        for cell in cells:
            self.add_edge(tile_id(cell), jid, TRIGGER_TILE, "boundary tile both sides")

    def register_path(
        self,
        query_key: str,
        digest: str,
        route_cells: Sequence[Cell],
        used_portals: Sequence[str],
    ) -> None:
        pid = path_id(query_key)
        regions = sorted({self.grid.region_of_cell(cell) for cell in route_cells})
        self.upsert_node(
            pid,
            PATH,
            digest,
            {"query_key": query_key, "regions": regions, "portals": list(used_portals)},
        )
        self.path_routes[query_key] = list(route_cells)
        self.path_regions[query_key] = regions
        self.path_portals[query_key] = list(used_portals)
        for region_id in regions:
            self.add_edge(flow_id(region_id), pid, TRIGGER_TILE, f"flow region {region_id} -> path")
        for portal_id in used_portals:
            self.add_edge(jump_id(portal_id), pid, TRIGGER_TILE, f"jump {portal_id} referenced by path")

    def anchor_unit(self, unit_id: str, cell: Cell) -> None:
        self.unit_anchors[unit_id] = cell

    # ---------- 失效闭包 ----------
    def tile_closure(self, changed_tiles: Sequence[Cell]) -> ClosureResult:
        changed = sorted(set(changed_tiles))
        boundary_hits = sorted(cell for cell in changed if self.grid.is_boundary_cell(cell))
        interior_hits = sorted(cell for cell in changed if not self.grid.is_boundary_cell(cell))

        fired_edges: Set[str] = set()
        invalid_flows: Set[int] = set()
        invalid_jumps: Set[str] = set()
        invalid_paths: Set[str] = set()

        for cell in changed:
            tid = tile_id(cell)
            for edge in self.outgoing(tid, TRIGGER_TILE):
                fired_edges.add(edge.key())
                dependent = self.nodes[edge.dependent]
                if not dependent.alive:
                    continue
                if dependent.kind == FLOW:
                    invalid_flows.add(int(dependent.meta["region_id"]))
                elif dependent.kind == JUMP:
                    invalid_jumps.add(str(dependent.meta["portal_id"]))

        # flow -> path 正向传播（jump 不向 flow 传播）。
        queue = sorted(flow_id(region_id) for region_id in invalid_flows)
        seen: Set[str] = set(queue)
        while queue:
            current = queue.pop(0)
            for edge in self.outgoing(current, TRIGGER_TILE):
                fired_edges.add(edge.key())
                dependent = self.nodes[edge.dependent]
                if dependent.kind == PATH and dependent.alive:
                    invalid_paths.add(str(dependent.meta["query_key"]))
                    if dependent.node_id not in seen:
                        seen.add(dependent.node_id)
                        queue.append(dependent.node_id)

        # jump 失效只标记其本身；引用 jump 的 path 单独失效，但不反向触及 flow。
        for portal_id in sorted(invalid_jumps):
            for edge in self.outgoing(jump_id(portal_id), TRIGGER_TILE):
                dependent = self.nodes[edge.dependent]
                if dependent.kind == PATH and dependent.alive:
                    fired_edges.add(edge.key())
                    invalid_paths.add(str(dependent.meta["query_key"]))

        reused_flows = sorted(
            int(node.meta["region_id"])
            for node_id, node in self.nodes.items()
            if node.kind == FLOW and node.alive and int(node.meta["region_id"]) not in invalid_flows
        )
        reused_jumps = sorted(
            str(node.meta["portal_id"])
            for node_id, node in self.nodes.items()
            if node.kind == JUMP and node.alive and str(node.meta["portal_id"]) not in invalid_jumps
        )
        reused_paths = sorted(
            str(node.meta["query_key"])
            for node_id, node in self.nodes.items()
            if node.kind == PATH and node.alive and str(node.meta["query_key"]) not in invalid_paths
        )

        return ClosureResult(
            changed_tiles=changed,
            invalidated_flow_regions=sorted(invalid_flows),
            invalidated_jump_portals=sorted(invalid_jumps),
            invalidated_path_queries=sorted(invalid_paths),
            reused_flows=reused_flows,
            reused_jump_portals=reused_jumps,
            reused_path_queries=reused_paths,
            edges_fired=sorted(fired_edges),
            boundary_hits=boundary_hits,
            interior_hits=interior_hits,
        )

    def apply_invalidation(self, closure: ClosureResult) -> None:
        for region_id in closure.invalidated_flow_regions:
            self.get_node(flow_id(region_id)).alive = False
        for portal_id in closure.invalidated_jump_portals:
            self.get_node(jump_id(portal_id)).alive = False
        for query_key in closure.invalidated_path_queries:
            self.get_node(path_id(query_key)).alive = False
        self.version += 1

    def unit_closure(self, unit_id: str, old_cell: Cell, new_cell: Cell) -> Set[str]:
        """单位移动只使以该单位为源的路径缓存失效，不触及流场/跳点。"""
        affected: Set[str] = set()
        for node_id, node in self.nodes.items():
            if node.kind == PATH and node.alive and node.meta.get("unit_id") == unit_id:
                node.alive = False
                affected.add(node_id)
        self.anchor_unit(unit_id, new_cell)
        self.version += 1
        return affected

    # ---------- 校验 ----------
    def assert_closure_precision(self, closure: ClosureResult, expected_regions: Set[int]) -> None:
        actual = set(closure.invalidated_flow_regions)
        if actual != set(expected_regions):
            raise ClosureMismatchError(
                "失效闭包区域集合与预期不符",
                expected=sorted(expected_regions),
                actual=sorted(actual),
                missing=sorted(set(expected_regions) - actual),
                extra=sorted(actual - set(expected_regions)),
            )

    def assert_no_reverse_invalidation(self, closure: ClosureResult, changed_regions: Set[int]) -> None:
        # jump 失效不应带出任何额外 flow；flow 集合必须恰好等于 tile 所在区域。
        if set(closure.invalidated_flow_regions) != changed_regions:
            raise ClosureMismatchError(
                "跳点失效反向污染了流场闭包",
                changed_regions=sorted(changed_regions),
                flow_closure=sorted(closure.invalidated_flow_regions),
            )

    # ---------- 序列化 ----------
    def to_payload(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "nodes": [self.nodes[key].to_payload() for key in sorted(self.nodes)],
            "edges": [asdict(self.edges[key]) for key in sorted(self.edges)],
            "path_routes": {key: [list(c) for c in self.path_routes[key]] for key in sorted(self.path_routes)},
            "path_regions": {key: self.path_regions[key] for key in sorted(self.path_regions)},
            "path_portals": {key: self.path_portals[key] for key in sorted(self.path_portals)},
            "unit_anchors": {key: list(value) for key, value in sorted(self.unit_anchors.items())},
        }

    def payload_hash(self) -> str:
        return content_hash(self.to_payload())

    @classmethod
    def from_payload(cls, grid: TacticalGrid, payload: Dict[str, Any]) -> "DependencyGraph":
        graph = cls(grid)
        graph.version = int(payload["version"])
        for node_payload in payload["nodes"]:
            graph.nodes[node_payload["node_id"]] = DepNode(**node_payload)
        for edge_payload in payload["edges"]:
            edge = DepEdge(**edge_payload)
            graph.edges[edge.key()] = edge
        for key, cells in payload["path_routes"].items():
            graph.path_routes[key] = [tuple(cell) for cell in cells]
        graph.path_regions = {key: list(value) for key, value in payload["path_regions"].items()}
        graph.path_portals = {key: list(value) for key, value in payload["path_portals"].items()}
        graph.unit_anchors = {key: tuple(value) for key, value in payload["unit_anchors"].items()}
        return graph
