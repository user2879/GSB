"""战术寻路诊断工作台门面。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .astar import AStarSolver, Route
from .budget import Budget, LRUCache, Scheduler
from .consistency import ConsistencyReport, compare_sets
from .dependency import (
    DependencyGraph,
    ClosureResult,
    flow_id,
    jump_id,
    path_id,
    tile_id,
)
from .encoding import content_hash
from .errors import (
    DependencyStaleError,
    TerrainRangeError,
    UnitInvalidError,
)
from .flowfield import (
    FlowFieldBuilder,
    FlowFieldSolver,
    JumpPointTable,
    Portal,
    RegionFlowField,
)
from .generate import Query, Sample, Unit, generate_sample
from .grid import Cell, TacticalGrid
from .persistence import PersistenceStore


def tile_content_hash(grid: TacticalGrid, cell: Cell) -> str:
    return content_hash(
        [
            list(cell),
            grid.cost_of(cell),
            int(grid.is_passable(cell)),
            int(grid.has_cover(cell)),
            grid.region_of_cell(cell),
        ]
    )


@dataclass
class ChangeRecord:
    sequence: int
    kind: str
    cell: Optional[Cell]
    old_cost: Optional[int]
    new_cost: Optional[int]
    closure: Dict[str, Any]
    map_hash_before: str
    map_hash_after: str

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "sequence": self.sequence,
            "kind": self.kind,
            "cell": list(self.cell) if cell_valid(self.cell) else None,
            "old_cost": self.old_cost,
            "new_cost": self.new_cost,
            "closure": self.closure,
            "map_hash_before": self.map_hash_before,
            "map_hash_after": self.map_hash_after,
        }
        return payload


def cell_valid(cell: Optional[Cell]) -> bool:
    return cell is not None


class PathfindingWorkbench:
    def __init__(
        self,
        sample: Sample,
        budget: Optional[Budget] = None,
        heuristic_name: str = "manhattan",
        cost_tolerance: int = 0,
    ) -> None:
        self.sample = sample
        self.grid: TacticalGrid = sample.grid
        self.budget = budget or Budget()
        self.scheduler = Scheduler(self.budget)
        self.heuristic_name = heuristic_name
        self.cost_tolerance = cost_tolerance
        self.astar = AStarSolver(self.grid, heuristic_name=heuristic_name)
        self.path_cache: LRUCache[str, Route] = LRUCache(self.budget.memory_budget)
        self.change_log: List[ChangeRecord] = []
        self.map_version = 0
        self.reuse_sources: Dict[str, str] = {}
        self._build_full_state()

    # ---------- 全量状态构建 ----------
    def _build_full_state(self) -> None:
        builder = FlowFieldBuilder(self.grid)
        self.jump_table: JumpPointTable = builder.build_jump_points()
        self.flows: Dict[int, RegionFlowField] = self._build_flows_parallel(
            self.grid.region_ids(), self.jump_table
        )
        self.graph = DependencyGraph(self.grid)
        tile_hashes = {cell: tile_content_hash(self.grid, cell) for cell in self._all_cells()}
        self.graph.seed_tile_nodes(tile_hashes)
        for region_id in self.grid.region_ids():
            flow = self.flows[region_id]
            self.graph.register_flow(
                region_id, flow.content_hash(), self.grid.cells_of_region(region_id)
            )
        for portal in self.jump_table.portals:
            self.graph.register_jump(
                portal.portal_id,
                self._portal_hash(portal),
                list(portal.cells()),
            )
        for unit in self.sample.units:
            self.graph.anchor_unit(unit.unit_id, unit.cell)
        self.flow_solver = FlowFieldSolver(self.grid, self.jump_table, self.flows)
        self.map_version = 1

    def _all_cells(self) -> List[Cell]:
        return [(r, c) for r in range(self.grid.height) for c in range(self.grid.width)]

    def _build_flows_parallel(
        self, region_ids: Sequence[int], jump_table: JumpPointTable
    ) -> Dict[int, RegionFlowField]:
        builder = FlowFieldBuilder(self.grid)
        items = [(rid, (lambda rid=rid: builder.build_region_flow(rid, jump_table))) for rid in region_ids]
        return self.scheduler.run_regions(items)

    def _portal_hash(self, portal: Portal) -> str:
        a, b = portal.a, portal.b
        return content_hash(
            [
                portal.portal_id,
                [list(a), self.grid.cost_of(a), int(self.grid.is_passable(a))],
                [list(b), self.grid.cost_of(b), int(self.grid.is_passable(b))],
            ]
        )

    # ---------- 增量地形修改 ----------
    def modify_terrain(self, cell: Cell, new_cost: int) -> ChangeRecord:
        if not (1 <= int(new_cost) <= 10):
            raise TerrainRangeError(
                "非法地形代价",
                cell=list(cell),
                cost=int(new_cost),
            )
        old_cost = self.grid.cost_of(cell)
        hash_before = self.grid.content_hash()
        closure = self.graph.tile_closure([cell])
        expected_regions = {self.grid.region_of_cell(cell)}
        self.graph.assert_closure_precision(closure, expected_regions)
        self.graph.assert_no_reverse_invalidation(closure, expected_regions)

        self.graph.apply_invalidation(closure)
        self.grid.set_terrain_cost(cell, new_cost)
        self._refresh_tile_hash(cell)
        self._rebuild_invalidated(closure)
        self._drop_invalid_cache_entries(closure)

        hash_after = self.grid.content_hash()
        self.map_version += 1
        record = ChangeRecord(
            sequence=self.map_version,
            kind="terrain_cost",
            cell=cell,
            old_cost=old_cost,
            new_cost=new_cost,
            closure=closure.to_payload(),
            map_hash_before=hash_before,
            map_hash_after=hash_after,
        )
        self.change_log.append(record)
        return record

    def modify_passable(self, cell: Cell, passable: bool) -> ChangeRecord:
        hash_before = self.grid.content_hash()
        closure = self.graph.tile_closure([cell])
        self.graph.apply_invalidation(closure)
        self.grid.set_passable(cell, passable)
        self._refresh_tile_hash(cell)
        self._rebuild_invalidated(closure)
        self._drop_invalid_cache_entries(closure)
        hash_after = self.grid.content_hash()
        self.map_version += 1
        record = ChangeRecord(
            sequence=self.map_version,
            kind="passable",
            cell=cell,
            old_cost=None,
            new_cost=int(passable),
            closure=closure.to_payload(),
            map_hash_before=hash_before,
            map_hash_after=hash_after,
        )
        self.change_log.append(record)
        return record

    def _refresh_tile_hash(self, cell: Cell) -> None:
        node = self.graph.get_node(tile_id(cell))
        node.content_hash = tile_content_hash(self.grid, cell)

    def _rebuild_invalidated(self, closure: ClosureResult) -> None:
        if closure.invalidated_jump_portals:
            new_table = FlowFieldBuilder(self.grid).build_jump_points()
            self.jump_table = new_table
            for portal in new_table.portals:
                if portal.portal_id in set(closure.invalidated_jump_portals):
                    node = self.graph.get_node(jump_id(portal.portal_id))
                    node.content_hash = self._portal_hash(portal)
                    node.alive = True

        rebuilt_regions = closure.invalidated_flow_regions
        if rebuilt_regions:
            new_flows = self._build_flows_parallel(rebuilt_regions, self.jump_table)
            for region_id, flow in new_flows.items():
                self.flows[region_id] = flow
                node = self.graph.get_node(flow_id(region_id))
                node.content_hash = flow.content_hash()
                node.alive = True

        self.flow_solver = FlowFieldSolver(self.grid, self.jump_table, self.flows)

    def _drop_invalid_cache_entries(self, closure: ClosureResult) -> None:
        invalid_queries = set(closure.invalidated_path_queries)
        for query_key in list(self.path_cache.keys()):
            if query_key in invalid_queries:
                self.path_cache._store.pop(query_key, None)

    # ---------- 单位移动 ----------
    def move_unit(self, unit_id: str, new_cell: Cell) -> ChangeRecord:
        unit = self._unit(unit_id)
        if not self.grid.in_bounds(*new_cell) or not self.grid.is_passable(new_cell):
            raise UnitInvalidError(
                "单位移动目标不可通行",
                unit_id=unit_id,
                cell=list(new_cell),
            )
        old_cell = unit.cell
        affected = self.graph.unit_closure(unit_id, old_cell, new_cell)
        replacement = Unit(
            unit_id=unit.unit_id,
            faction=unit.faction,
            cell=new_cell,
            movement_budget=unit.movement_budget,
        )
        new_units = [
            replacement if candidate.unit_id == unit_id else candidate
            for candidate in self.sample.units
        ]
        new_queries = [
            Query(query.query_id, query.unit_id, new_cell, query.target)
            if query.unit_id == unit_id
            else query
            for query in self.sample.queries
        ]
        self.sample = Sample(
            seed=self.sample.seed,
            grid=self.grid,
            units=new_units,
            queries=new_queries,
        )
        for query_key in list(self.path_cache.keys()):
            if path_id(query_key) in affected:
                self.path_cache._store.pop(query_key, None)
        self.map_version += 1
        record = ChangeRecord(
            sequence=self.map_version,
            kind="unit_move",
            cell=new_cell,
            old_cost=None,
            new_cost=None,
            closure={"invalidated_paths": sorted(affected)},
            map_hash_before=self.grid.content_hash(),
            map_hash_after=self.grid.content_hash(),
        )
        self.change_log.append(record)
        return record

    def _unit(self, unit_id: str) -> Unit:
        for unit in self.sample.units:
            if unit.unit_id == unit_id:
                return unit
        raise UnitInvalidError("未知单位", unit_id=unit_id)

    # ---------- 双求解 ----------
    def solve_query(self, query: Query, prefer: str = "flowfield") -> Route:
        cached = self.path_cache.get(query.key())
        if cached is not None:
            self.reuse_sources[query.key()] = "path_cache:hit"
            return cached

        if prefer == "astar":
            unit = self._unit(query.unit_id)
            route = self.astar.solve(
                query.source, query.target, unit.movement_budget, cache_source="cache:miss"
            )
        else:
            unit = self._unit(query.unit_id)
            route = self.flow_solver.solve(
                query.source, query.target, unit.movement_budget, cache_source="cache:miss"
            )
        self.scheduler.account_expansion(route.expanded)
        self._store_path(query, route)
        self.reuse_sources[query.key()] = "path_cache:miss"
        return route

    def solve_dual(self, query: Query) -> Dict[str, Route]:
        unit = self._unit(query.unit_id)
        astar_route = self.astar.solve(
            query.source, query.target, unit.movement_budget, cache_source="astar:direct"
        )
        flow_route = self.flow_solver.solve(
            query.source, query.target, unit.movement_budget, cache_source="flowfield:direct"
        )
        return {"astar": astar_route, "flowfield": flow_route}

    def _store_path(self, query: Query, route: Route) -> None:
        used_portals = self._portals_along_route(route)
        self.graph.register_path(
            query.key(),
            route.path_hash(),
            route.cells,
            used_portals,
        )
        node = self.graph.get_node(path_id(query.key()))
        node.meta["unit_id"] = query.unit_id
        evicted = self.path_cache.put(query.key(), route)
        if evicted:
            self.scheduler.enforce_memory(self.path_cache)

    def _portals_along_route(self, route: Route) -> List[str]:
        cell_set = set(route.cells)
        used: List[str] = []
        for portal in self.jump_table.portals:
            if portal.a in cell_set and portal.b in cell_set:
                adjacent = (
                    abs(portal.a[0] - portal.b[0]) + abs(portal.a[1] - portal.b[1]) == 1
                )
                if adjacent:
                    used.append(portal.portal_id)
        return sorted(used)

    def solve_batch(
        self,
        queries: Optional[Sequence[Query]] = None,
        prefer: str = "flowfield",
    ) -> Dict[str, Route]:
        targets = list(queries) if queries is not None else list(self.sample.queries)
        items = [
            (query.key(), (lambda query=query: self.solve_query(query, prefer=prefer)))
            for query in targets
        ]
        return self.scheduler.run_queries(items)

    # ---------- 全量重建对照 ----------
    def full_rebuild_routes(self, queries: Sequence[Query]) -> Dict[str, Route]:
        fresh_sample = Sample(
            seed=self.sample.seed,
            grid=TacticalGrid.from_payload(self.grid.to_payload()),
            units=self.sample.units,
            queries=self.sample.queries,
        )
        fresh = PathfindingWorkbench(
            fresh_sample,
            budget=Budget(thread_budget=1, memory_budget=self.budget.memory_budget),
            heuristic_name=self.heuristic_name,
            cost_tolerance=self.cost_tolerance,
        )
        return fresh.solve_batch(queries, prefer="flowfield")

    def consistency_check(
        self, queries: Optional[Sequence[Query]] = None
    ) -> ConsistencyReport:
        targets = list(queries) if queries is not None else list(self.sample.queries)
        incremental = self.solve_batch(targets, prefer="flowfield")
        rebuilt = self.full_rebuild_routes(targets)
        edges_by_query: Dict[str, List[str]] = {}
        for query in targets:
            node_id = path_id(query.key())
            edges = [edge.key() for edge in self.graph.edges.values() if edge.dependent == node_id]
            edges_by_query[query.key()] = sorted(edges)
        report = compare_sets(incremental, rebuilt, edges_by_query, self.cost_tolerance)
        return report

    # ---------- 报告 ----------
    def closure_history(self) -> List[Dict[str, Any]]:
        return [record.to_payload() for record in self.change_log]

    def path_hash_report(self) -> Dict[str, Any]:
        rows: Dict[str, Any] = {}
        for key in sorted(self.path_cache.keys()):
            route = self.path_cache.get(key)
            assert route is not None
            rows[key] = {
                "path_hash": route.path_hash(),
                "cost": route.cost,
                "expanded": route.expanded,
                "solver": route.solver,
                "cache_source": self.reuse_sources.get(key, route.cache_source),
                "within_budget": route.within_budget,
            }
        return rows

    def state_payload(self) -> Dict[str, Any]:
        return {
            "map_version": self.map_version,
            "seed": self.sample.seed,
            "grid": self.grid.to_payload(),
            "units": [unit.to_payload() for unit in self.sample.units],
            "queries": [query.to_payload() for query in self.sample.queries],
            "dependency_graph": self.graph.to_payload(),
            "jump_hash": self.jump_table.content_hash(),
            "flow_hashes": {
                str(region_id): flow.content_hash() for region_id, flow in sorted(self.flows.items())
            },
            "path_hashes": self.path_hash_report(),
            "changes": self.closure_history(),
            "scheduler": self.scheduler.report(),
            "reuse_sources": dict(sorted(self.reuse_sources.items())),
        }

    # ---------- 持久化集成 ----------
    def persist(self, store: PersistenceStore) -> str:
        return store.commit_manifest(self.state_payload(), self.map_version)

    @classmethod
    def restore(
        cls,
        store: PersistenceStore,
        budget: Optional[Budget] = None,
        heuristic_name: str = "manhattan",
    ) -> "PathfindingWorkbench":
        payload = store.load_manifest()
        grid = TacticalGrid.from_payload(payload["grid"])
        units = [
            Unit(
                unit_id=item["unit_id"],
                faction=item["faction"],
                cell=tuple(item["cell"]),
                movement_budget=item["movement_budget"],
            )
            for item in payload["units"]
        ]
        queries = [
            Query(
                query_id=item["query_id"],
                unit_id=item["unit_id"],
                source=tuple(item["source"]),
                target=tuple(item["target"]),
            )
            for item in payload["queries"]
        ]
        sample = Sample(seed=payload.get("seed", 0), grid=grid, units=units, queries=queries)
        workbench = cls(sample, budget=budget, heuristic_name=heuristic_name)
        stored_graph = DependencyGraph.from_payload(grid, payload["dependency_graph"])
        _verify_restored_hashes(workbench, stored_graph)
        workbench.graph = stored_graph
        workbench.map_version = int(payload["map_version"])
        return workbench


def _verify_restored_hashes(
    workbench: PathfindingWorkbench, stored_graph: DependencyGraph
) -> None:
    current = workbench.graph
    for node_id, stored_node in stored_graph.nodes.items():
        if node_id not in current.nodes:
            continue
        current_node = current.nodes[node_id]
        if stored_node.kind == "path":
            continue
        if stored_node.content_hash != current_node.content_hash:
            raise DependencyStaleError(
                "恢复后节点哈希与重算结果不一致",
                node_id=node_id,
                stored=stored_node.content_hash,
                rebuilt=current_node.content_hash,
            )


def build_workbench(
    seed: int = 20261119,
    budget: Optional[Budget] = None,
    heuristic_name: str = "manhattan",
) -> PathfindingWorkbench:
    sample = generate_sample(seed=seed)
    return PathfindingWorkbench(sample, budget=budget, heuristic_name=heuristic_name)
