"""Cached, concurrent tier solver with exact incremental invalidation."""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .canonical import sha256_json
from .config import AppConfig
from .dependency import DependencyGraph, DependencyNode
from .errors import ErrorCode, TacticalError
from .models import (
    Budgets,
    MapBundle,
    MapData,
    Query,
    Route,
    StrategyDecision,
    region_content_hash,
)
from .solvers.astar import AStarSolver
from .solvers.chunkjump import ChunkJumpSolver, JumpData, JumpDataBuilder
from .solvers.flowfield import FlowField, FlowFieldBuilder, flow_key


MIB = 1024 * 1024


@dataclass(frozen=True)
class MapEdit:
    cell: int
    terrain: int | None = None
    passable: bool | None = None


@dataclass
class FlowEntry:
    field: FlowField
    node_id: str
    deps_versions: dict[str, str]


@dataclass
class JumpEntry:
    data: JumpData
    node_id: str
    deps_versions: dict[str, str]


@dataclass
class PathEntry:
    query_id: str
    strategy: str
    route: Route
    node_id: str
    deps_versions: dict[str, str]
    route_hash: str


def region_node_id(tier: str, region_id: int) -> str:
    return f"region:{tier}:{region_id}"


def chunk_node_id(tier: str, region_id: int) -> str:
    return f"chunk:{tier}:{region_id}"


def flow_node_id(tier: str, key: str) -> str:
    return f"flow:{tier}:{key}"


def jump_node_id(tier: str, key: str) -> str:
    return f"jump:{tier}:{key}"


def path_node_id(tier: str, query_id: str) -> str:
    return f"path:{tier}:{query_id}"


class TierEngine:
    def __init__(
        self,
        bundle: MapBundle,
        config: AppConfig,
        *,
        graph: DependencyGraph | None = None,
    ) -> None:
        self.bundle = bundle
        self.config = config
        self.map = bundle.map_data
        self.tier = self.map.tier
        self.astar = AStarSolver(self.map)
        self.flow_builder = FlowFieldBuilder(self.map)
        self.jump_builder = JumpDataBuilder(self.map)
        self.jump_solver = ChunkJumpSolver(self.map, self.jump_builder)
        self.flow_cache: dict[str, FlowEntry] = {}
        self.jump_cache: dict[str, JumpEntry] = {}
        self.path_cache: dict[str, PathEntry] = {}
        self._lock = threading.RLock()
        self._flow_locks: dict[str, threading.RLock] = {}
        self._batch_reuse_events: list[dict[str, Any]] = []
        self.region_versions: dict[int, int] = {
            region_id: 1 for region_id in range(self.map.region_count)
        }
        self.chunk_versions: dict[int, int] = {
            region_id: 1 for region_id in range(self.map.region_count)
        }
        if graph is None:
            self.graph = DependencyGraph(self.tier)
            self._initialize_graph()
        else:
            self.graph = graph

    def _initialize_graph(self) -> None:
        map_node = DependencyNode(
            id=f"map:{self.tier}",
            kind="map",
            content_hash=self.map.content_hash,
            version=self.map.map_version,
            deps=(),
            metadata={"tier": self.tier},
        )
        self.graph.add(map_node)
        for region_id in range(self.map.region_count):
            region_hash = region_content_hash(self.map, region_id)
            region_version = f"v{self.region_versions[region_id]}:{region_hash[:16]}"
            chunk_version = f"v{self.chunk_versions[region_id]}:{region_hash[:16]}"
            self.graph.add(
                DependencyNode(
                    id=region_node_id(self.tier, region_id),
                    kind="region",
                    content_hash=region_hash,
                    version=region_version,
                    deps=(),
                    metadata={"region_id": region_id},
                )
            )
            self.graph.add(
                DependencyNode(
                    id=chunk_node_id(self.tier, region_id),
                    kind="chunk",
                    content_hash=region_hash,
                    version=chunk_version,
                    deps=(),
                    metadata={"region_id": region_id},
                )
            )

    def _version_for_hash(self, kind: str, content_hash: str) -> str:
        return f"{kind}:{content_hash[:20]}"

    def _get_flow_lock(self, key: str) -> threading.RLock:
        with self._lock:
            return self._flow_locks.setdefault(key, threading.RLock())

    def _ensure_flow(
        self,
        region_id: int,
        target: int,
        mobility: str,
        *,
        force: bool = False,
    ) -> FlowEntry:
        key = flow_key(region_id, target, mobility)
        lock = self._get_flow_lock(key)
        with lock:
            with self._lock:
                existing = self.flow_cache.get(key)
                if existing is not None and not force:
                    return existing
                if existing is not None:
                    self.flow_cache.pop(key, None)
                    self.graph.remove(existing.node_id)
            field = self.flow_builder.build(region_id, target, mobility)
            node_id = flow_node_id(self.tier, key)
            dependency = region_node_id(self.tier, region_id)
            with self._lock:
                dependency_version = self.graph.get(dependency).version
                node = DependencyNode(
                    id=node_id,
                    kind="flow",
                    content_hash=field.content_hash,
                    version=self._version_for_hash("flow", field.content_hash),
                    deps=(dependency,),
                    metadata={
                        "key": key,
                        "region_id": region_id,
                        "target": target,
                        "mobility": mobility,
                    },
                )
                self.graph.add(node)
                entry = FlowEntry(
                    field=field,
                    node_id=node_id,
                    deps_versions={dependency: dependency_version},
                )
                self.flow_cache[key] = entry
                return entry

    def _register_jump_data(self, data: JumpData) -> JumpEntry:
        key = data.key
        node_id = jump_node_id(self.tier, key)
        dependency = chunk_node_id(self.tier, data.region_id)
        with self._lock:
            existing = self.jump_cache.get(key)
            if existing is not None and existing.data.content_hash == data.content_hash:
                return existing
            dependency_version = self.graph.get(dependency).version
            node = DependencyNode(
                id=node_id,
                kind="jump",
                content_hash=data.content_hash,
                version=self._version_for_hash("jump", data.content_hash),
                deps=(dependency,),
                metadata={
                    "key": key,
                    "region_id": data.region_id,
                    "mobility": data.mobility,
                },
            )
            self.graph.add(node)
            entry = JumpEntry(
                data=data,
                node_id=node_id,
                deps_versions={dependency: dependency_version},
            )
            self.jump_cache[key] = entry
            return entry

    def _register_cached_jumps(self) -> None:
        for data in self.jump_builder.all_cached():
            self._register_jump_data(data)

    def _flow_entry_for(self, query: Query) -> FlowEntry:
        region_id = self.map.region_of_cell(query.start)
        if region_id != self.map.region_of_cell(query.target):
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "flow strategy requires source and target in one region",
                context={"query_id": query.id},
            )
        return self._ensure_flow(region_id, query.target, query.mobility)

    def _register_path(
        self,
        query: Query,
        decision: StrategyDecision,
        route: Route,
    ) -> PathEntry:
        node_id = path_node_id(self.tier, query.id)
        dependencies: list[str] = []
        for region_id in route.route_regions:
            dependencies.append(region_node_id(self.tier, region_id))
        if route.flow_key is not None:
            dependencies.append(flow_node_id(self.tier, route.flow_key))
        for region_id in route.jump_regions:
            dependencies.append(f"jump:{self.tier}:jump:{region_id}:{query.mobility}")
        dependencies = sorted(set(dependencies))
        route_hash = route.route_hash(self.map.width)
        with self._lock:
            for dependency in dependencies:
                self.graph.get(dependency)
            dep_versions = {
                dependency: self.graph.get(dependency).version
                for dependency in dependencies
            }
            node = DependencyNode(
                id=node_id,
                kind="path",
                content_hash=route_hash,
                version=self._version_for_hash("path", route_hash),
                deps=tuple(dependencies),
                metadata={
                    "query_id": query.id,
                    "strategy": decision.strategy,
                    "cost": route.cost,
                    "flow_key": route.flow_key,
                    "jump_regions": list(route.jump_regions),
                    "route_regions": list(route.route_regions),
                },
            )
            self.graph.add(node)
            entry = PathEntry(
                query_id=query.id,
                strategy=decision.strategy,
                route=route,
                node_id=node_id,
                deps_versions=dep_versions,
                route_hash=route_hash,
            )
            self.path_cache[query.id] = entry
            return entry

    def _try_reuse_path(self, query: Query, decision: StrategyDecision) -> PathEntry | None:
        with self._lock:
            entry = self.path_cache.get(query.id)
            if entry is None or entry.strategy != decision.strategy:
                return None
            try:
                self.graph.assert_dependencies_current(entry.node_id, entry.deps_versions)
            except TacticalError as exc:
                if exc.code != ErrorCode.DEPENDENCY_STALE:
                    raise
                self.path_cache.pop(query.id, None)
                self.graph.remove(entry.node_id)
                return None
            self._batch_reuse_events.append(
                {
                    "query_id": query.id,
                    "source": "path_cache",
                    "node_id": entry.node_id,
                    "version": self.graph.get(entry.node_id).version,
                    "content_hash": entry.route_hash,
                }
            )
            return entry

    def _solve_one(
        self,
        query: Query,
        decision: StrategyDecision,
        *,
        force: bool = False,
    ) -> Route:
        if not force:
            reused = self._try_reuse_path(query, decision)
            if reused is not None:
                return reused.route
        if decision.strategy == "region_flow":
            entry = self._flow_entry_for(query)
            route = entry.field.solve(query, self.map.width)
        elif decision.strategy == "dense_astar":
            route = self.astar.solve(query, strategy="dense_astar")
        elif decision.strategy == "chunk_jump":
            route = self.jump_solver.solve(query)
            self._register_cached_jumps()
        else:
            raise TacticalError(
                ErrorCode.BUDGET_DENIED,
                "unknown strategy selected",
                context={"query_id": query.id, "strategy": decision.strategy},
            )
        self._register_path(query, decision, route)
        return route

    def _cache_bytes(self) -> int:
        with self._lock:
            return sum(entry.field.estimated_bytes for entry in self.flow_cache.values()) + sum(
                entry.data.estimated_bytes for entry in self.jump_cache.values()
            ) + sum(
                entry.route.estimated_peak_bytes for entry in self.path_cache.values()
            )

    def _evict_all_cache(self) -> list[str]:
        evicted: list[str] = []
        with self._lock:
            for query_id in sorted(self.path_cache):
                entry = self.path_cache[query_id]
                evicted.append(entry.node_id)
                self.graph.remove(entry.node_id)
            self.path_cache.clear()
            for key in sorted(self.flow_cache):
                entry = self.flow_cache[key]
                evicted.append(entry.node_id)
                self.graph.remove(entry.node_id)
            self.flow_cache.clear()
            for key in sorted(self.jump_cache):
                entry = self.jump_cache[key]
                evicted.append(entry.node_id)
                self.graph.remove(entry.node_id)
            self.jump_cache.clear()
            self.jump_builder.invalidate(None)
        return evicted

    def solve_batch(
        self,
        queries: list[Query],
        decisions: dict[str, StrategyDecision],
        budget: Budgets,
        *,
        force: bool = False,
        memory_budget_mib: float | None = None,
    ) -> dict[str, Any]:
        budget.validate()
        memory_budget = (
            budget.total_memory_mib if memory_budget_mib is None else float(memory_budget_mib)
        )
        if memory_budget <= 0:
            raise TacticalError(
                ErrorCode.BUDGET_DENIED,
                "total solver memory budget must be positive",
                context={"tier": self.tier, "memory_budget_mib": memory_budget},
            )
        ordered = sorted(queries, key=lambda item: item.id)
        for query in ordered:
            if query.id not in decisions:
                raise TacticalError(
                    ErrorCode.BUDGET_DENIED,
                    "missing strategy decision for query",
                    context={"query_id": query.id},
                )
        with self._lock:
            self._batch_reuse_events = []
        workers = max(1, min(budget.max_threads, len(ordered) or 1))
        estimated_query_peak = max(
            [decisions[query.id].estimated_bytes for query in ordered] or [1]
        )
        parallel_estimate = estimated_query_peak * workers + self._cache_bytes()
        downgrades: list[dict[str, Any]] = []
        cache_evicted: list[str] = []
        limit = int(memory_budget * MIB)
        if workers > 1 and parallel_estimate > limit:
            before = {
                "workers": workers,
                "estimated_peak_bytes": parallel_estimate,
                "cache_entries": len(self.path_cache) + len(self.flow_cache) + len(self.jump_cache),
            }
            workers = 1
            estimated_query_peak = max(
                [decisions[query.id].estimated_bytes for query in ordered] or [1]
            )
            parallel_estimate = estimated_query_peak + self._cache_bytes()
            downgrades.append(
                {
                    "stage": "serial",
                    "priority": 1,
                    "reason": "estimated parallel memory exceeds total budget",
                    "before": before,
                    "after": {
                        "workers": workers,
                        "estimated_peak_bytes": parallel_estimate,
                        "cache_entries": len(self.path_cache) + len(self.flow_cache) + len(self.jump_cache),
                    },
                }
            )
        if parallel_estimate > limit:
            before_cache = self._cache_bytes()
            before_count = len(self.path_cache) + len(self.flow_cache) + len(self.jump_cache)
            cache_evicted = self._evict_all_cache()
            parallel_estimate = estimated_query_peak
            downgrades.append(
                {
                    "stage": "cache_eviction",
                    "priority": 2,
                    "reason": "serial peak still exceeds total budget; all reusable caches evicted",
                    "before": {
                        "workers": workers,
                        "cache_bytes": before_cache,
                        "cache_entries": before_count,
                        "estimated_peak_bytes": before_cache + estimated_query_peak,
                    },
                    "after": {
                        "workers": workers,
                        "cache_bytes": 0,
                        "cache_entries": 0,
                        "estimated_peak_bytes": parallel_estimate,
                    },
                }
            )
        if parallel_estimate > limit:
            downgrades.append(
                {
                    "stage": "minimal_memory",
                    "priority": 3,
                    "reason": "minimum single-query estimate still exceeds budget; solving in minimal mode without dropping queries",
                    "before": {
                        "workers": workers,
                        "estimated_peak_bytes": parallel_estimate,
                    },
                    "after": {
                        "workers": 1,
                        "estimated_peak_bytes": parallel_estimate,
                    },
                }
            )
            workers = 1

        runnable = ordered
        flow_jobs: dict[str, tuple[int, int, str]] = {}
        for query in runnable:
            decision = decisions[query.id]
            if decision.strategy != "region_flow":
                continue
            region_id = self.map.region_of_cell(query.start)
            key = flow_key(region_id, query.target, query.mobility)
            with self._lock:
                cached = self.flow_cache.get(key)
            if cached is None or force:
                flow_jobs[key] = (region_id, query.target, query.mobility)

        if flow_jobs:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"flow-{self.tier}") as executor:
                futures = {
                    key: executor.submit(
                        self._ensure_flow,
                        region_id,
                        target,
                        mobility,
                        force=force,
                    )
                    for key, (region_id, target, mobility) in sorted(flow_jobs.items())
                }
                for key in sorted(futures):
                    futures[key].result()

        routes: dict[str, Route] = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"query-{self.tier}") as executor:
            futures = {
                query.id: executor.submit(self._solve_one, query, decisions[query.id], force=force)
                for query in runnable
            }
            for query_id in sorted(futures):
                routes[query_id] = futures[query_id].result()
        with self._lock:
            reuse_events = sorted(
                self._batch_reuse_events,
                key=lambda item: (item["query_id"], item["node_id"]),
            )
        return {
            "routes": routes,
            "workers_used": workers,
            "downgrades": downgrades,
            "cache_evicted": sorted(cache_evicted),
            "estimated_peak_bytes": parallel_estimate,
            "reuse_events": reuse_events,
            "queries": [
                routes[query_id].to_dict(self.map.width)
                for query_id in sorted(routes)
            ],
        }

    def _update_solver_map(self, new_map: MapData) -> None:
        self.map = new_map
        self.bundle.map_data = new_map
        self.astar.map = new_map
        self.astar._allowed.clear()
        self.astar._min_cost.clear()
        self.flow_builder.map = new_map
        self.flow_builder._allowed.clear()
        self.jump_builder.map = new_map
        self.jump_builder._allowed.clear()
        self.jump_solver.map = new_map
        self.jump_solver._allowed.clear()
        self.jump_solver._min_enter.clear()

    def _node_metadata(self, node_ids: Iterable[str]) -> dict[str, DependencyNode]:
        return {node_id: self.graph.get(node_id) for node_id in sorted(node_ids)}

    def apply_edits(
        self,
        edits: list[MapEdit],
        decisions: dict[str, StrategyDecision],
    ) -> dict[str, Any]:
        if not edits:
            return {
                "changed_cells": [],
                "affected_regions": [],
                "closure": [],
                "recomputed": [],
                "reused_nodes": [],
                "new_materialized_nodes": [],
                "routes": {
                    query_id: entry.route.to_dict(self.map.width)
                    for query_id, entry in sorted(self.path_cache.items())
                },
            }
        terrain = self.map.terrain.copy()
        passable = self.map.passable.copy()
        changed_cells: set[int] = set()
        for edit in edits:
            if not 0 <= edit.cell < self.map.cells:
                raise TacticalError(
                    ErrorCode.MAP_SHAPE,
                    "edit cell is outside the map",
                    context={"tier": self.tier, "cell": edit.cell},
                )
            if edit.terrain is None and edit.passable is None:
                raise TacticalError(
                    ErrorCode.TERRAIN_RANGE,
                    "edit must change terrain or passability",
                    context={"cell": edit.cell},
                )
            y = edit.cell // self.map.width
            x = edit.cell % self.map.width
            old_terrain = int(terrain[y, x])
            old_passable = bool(passable[y, x])
            if edit.terrain is not None:
                if not 1 <= int(edit.terrain) <= 9:
                    raise TacticalError(
                        ErrorCode.TERRAIN_RANGE,
                        "terrain edit is outside [1, 9]",
                        context={"cell": edit.cell, "terrain": edit.terrain},
                    )
                terrain[y, x] = np.uint8(edit.terrain)
            if edit.passable is not None:
                passable[y, x] = np.uint8(bool(edit.passable))
            if int(terrain[y, x]) != old_terrain or bool(passable[y, x]) != old_passable:
                changed_cells.add(edit.cell)
        if not changed_cells:
            return {
                "changed_cells": [],
                "affected_regions": [],
                "closure": [],
                "recomputed": [],
                "reused_nodes": [],
                "new_materialized_nodes": [],
                "routes": {
                    query_id: entry.route.to_dict(self.map.width)
                    for query_id, entry in sorted(self.path_cache.items())
                },
            }
        affected_regions = sorted(
            {self.map.region_of_cell(cell) for cell in changed_cells}
        )
        seeds = [region_node_id(self.tier, region_id) for region_id in affected_regions]
        seeds.extend(chunk_node_id(self.tier, region_id) for region_id in affected_regions)
        closure = self.graph.cache_closure(seeds)
        closure_nodes = self._node_metadata(closure)
        reused_nodes = [
            {
                "node_id": node_id,
                "kind": self.graph.nodes[node_id].kind,
                "version": self.graph.nodes[node_id].version,
                "content_hash": self.graph.nodes[node_id].content_hash,
            }
            for node_id in sorted(self.graph.nodes)
            if self.graph.nodes[node_id].kind in {"flow", "jump", "path"}
            and node_id not in closure_nodes
        ]
        existing_nodes = set(self.graph.nodes)
        new_map = self.map.clone_with(terrain=terrain, passable=passable)
        for query in self.bundle.queries:
            if not bool(new_map.allowed_mask(query.mobility).flat[query.start]) or not bool(
                new_map.allowed_mask(query.mobility).flat[query.target]
            ):
                raise TacticalError(
                    ErrorCode.UNIT_INVALID,
                    "edit invalidates a query endpoint",
                    context={"query_id": query.id, "regions": affected_regions},
                )
        self._update_solver_map(new_map)
        with self._lock:
            map_node = self.graph.get(f"map:{self.tier}")
            map_node.content_hash = self.map.content_hash
            map_node.version = self.map.map_version
            for region_id in affected_regions:
                self.region_versions[region_id] += 1
                self.chunk_versions[region_id] += 1
                region_hash = region_content_hash(self.map, region_id)
                region_node = self.graph.get(region_node_id(self.tier, region_id))
                region_node.content_hash = region_hash
                region_node.version = f"v{self.region_versions[region_id]}:{region_hash[:16]}"
                chunk_node = self.graph.get(chunk_node_id(self.tier, region_id))
                chunk_node.content_hash = region_hash
                chunk_node.version = f"v{self.chunk_versions[region_id]}:{region_hash[:16]}"

            flow_records: list[dict[str, Any]] = []
            jump_records: list[dict[str, Any]] = []
            path_records: list[dict[str, Any]] = []
            for node_id in closure:
                node = self.graph.nodes[node_id]
                if node.kind == "flow":
                    flow_records.append(dict(node.metadata))
                    key = str(node.metadata["key"])
                    self.flow_cache.pop(key, None)
                elif node.kind == "jump":
                    jump_records.append(dict(node.metadata))
                    key = str(node.metadata["key"])
                    self.jump_cache.pop(key, None)
                    self.jump_builder.invalidate_key(
                        int(node.metadata["region_id"]), str(node.metadata["mobility"])
                    )
                elif node.kind == "path":
                    path_records.append(dict(node.metadata))
                    self.path_cache.pop(str(node.metadata["query_id"]), None)
                self.graph.remove(node_id)

        actual_recomputed: set[str] = set()
        for record in sorted(flow_records, key=lambda item: item["key"]):
            entry = self._ensure_flow(
                int(record["region_id"]),
                int(record["target"]),
                str(record["mobility"]),
                force=True,
            )
            actual_recomputed.add(entry.node_id)
        for record in sorted(jump_records, key=lambda item: item["key"]):
            data = self.jump_builder.get(
                int(record["region_id"]), str(record["mobility"])
            )
            entry = self._register_jump_data(data)
            actual_recomputed.add(entry.node_id)
        query_lookup = self.bundle.query_by_id()
        for record in sorted(path_records, key=lambda item: item["query_id"]):
            query_id = str(record["query_id"])
            query = query_lookup[query_id]
            if query_id not in decisions:
                raise TacticalError(
                    ErrorCode.BUDGET_DENIED,
                    "incremental query lacks strategy decision",
                    context={"query_id": query_id},
                )
            route = self._solve_one(query, decisions[query_id], force=True)
            actual_recomputed.add(path_node_id(self.tier, query_id))

        expected = set(closure)
        if actual_recomputed != expected:
            raise TacticalError(
                ErrorCode.CLOSURE_MISMATCH,
                "actual recompute set differs from dependency closure",
                context={
                    "tier": self.tier,
                    "missing": sorted(expected - actual_recomputed),
                    "extra": sorted(actual_recomputed - expected),
                },
            )
        new_materialized = sorted(
            node_id
            for node_id in self.graph.nodes
            if node_id not in existing_nodes and node_id not in expected
        )
        routes = {
            query_id: self.path_cache[query_id].route
            for query_id in sorted(self.path_cache)
        }
        return {
            "changed_cells": sorted(changed_cells),
            "affected_regions": affected_regions,
            "closure": closure,
            "recomputed": sorted(actual_recomputed),
            "reused_nodes": reused_nodes,
            "new_materialized_nodes": new_materialized,
            "routes": routes,
        }

    def route_hashes(self) -> dict[str, str]:
        return {
            query_id: entry.route_hash
            for query_id, entry in sorted(self.path_cache.items())
        }

    def snapshot_payload(
        self,
        *,
        snapshot_id: str,
        decisions: dict[str, StrategyDecision] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": "tactical-snapshot-v1",
            "snapshot_id": snapshot_id,
            "tier": self.tier,
            "map": self.map.map_manifest(),
            "dependency_graph": self.graph.to_dict(),
            "region_versions": {
                str(key): value for key, value in sorted(self.region_versions.items())
            },
            "chunk_versions": {
                str(key): value for key, value in sorted(self.chunk_versions.items())
            },
            "flow_cache": [
                entry.field.manifest() for entry in sorted(self.flow_cache.values(), key=lambda item: item.field.key)
            ],
            "jump_cache": [
                entry.data.manifest() for entry in sorted(self.jump_cache.values(), key=lambda item: item.data.key)
            ],
            "path_cache": [
                {
                    "query_id": entry.query_id,
                    "strategy": entry.strategy,
                    "node_id": entry.node_id,
                    "route": entry.route.to_dict(self.map.width),
                    "route_hash": entry.route_hash,
                    "deps_versions": dict(sorted(entry.deps_versions.items())),
                }
                for entry in sorted(self.path_cache.values(), key=lambda item: item.query_id)
            ],
            "decisions": [
                decisions[key].to_dict() for key in sorted(decisions)
            ]
            if decisions is not None
            else [],
        }
