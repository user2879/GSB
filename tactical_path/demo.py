"""One-command reproducible experiment orchestration."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_bytes,
    file_sha256,
    sha256_bytes,
    sha256_json,
)
from .config import AppConfig, load_config
from .consistency import compare_incremental_with_full
from .engine import MapEdit, TierEngine
from .errors import ErrorCode, TacticalError
from .mapgen import generate_bundle, save_bundle
from .models import Budgets, Query, Route
from .persistence import StateStore, verify_recovered_hashes
from .solvers.astar import AStarSolver
from .solvers.flowfield import FlowFieldBuilder, flow_eligible, flow_key
from .strategy import StrategyPlanner


def _reset_directory(path: Path) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved == cwd or resolved in cwd.parents:
        raise TacticalError(
            ErrorCode.MAP_SHAPE,
            "refusing to reset the project directory or one of its parents",
            context={"workspace": str(resolved)},
        )
    if path.exists():
        if any(path.iterdir()) and not (path / ".demo-root").exists():
            raise TacticalError(
                ErrorCode.MAP_SHAPE,
                "refusing to reset a non-empty directory without the demo marker",
                context={"workspace": str(resolved)},
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path / ".demo-root", {"schema": "demo-root-v1"})


def _tree_files(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)).replace(os.sep, "/"): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".demo-root"
    }


def _tree_hash(files: dict[str, bytes]) -> str:
    digest = __import__("hashlib").sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[name])
    return digest.hexdigest()


def _generation_verification(
    cfg: AppConfig,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundles: dict[str, Any] = {}
    tiers: list[dict[str, Any]] = []
    for name, spec in cfg.tiers.items():
        first = generate_bundle(spec, seed=cfg.seed, mobility_rules=cfg.mobility)
        save_bundle(first, root / "samples" / name)
        second = generate_bundle(spec, seed=cfg.seed, mobility_rules=cfg.mobility)
        save_bundle(second, root / "repro" / name)
        first_files = _tree_files(root / "samples" / name)
        second_files = _tree_files(root / "repro" / name)
        same_names = sorted(first_files) == sorted(second_files)
        differing = [
            filename
            for filename in sorted(first_files)
            if second_files.get(filename) != first_files[filename]
        ]
        if not same_names or differing:
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "deterministic generation produced different bytes",
                context={"tier": name, "differing_files": differing},
            )
        bundles[name] = first
        tiers.append(
            {
                "tier": name,
                "size": spec.size,
                "byte_identical": True,
                "tree_sha256": _tree_hash(first_files),
                "map_content_hash": first.map_data.content_hash,
                "region_count": first.map_data.region_count,
                "unit_count": len(first.units),
                "query_count": len(first.queries),
            }
        )
    verification = {
        "schema": "generation-verification-v1",
        "seed": cfg.seed,
        "all_byte_identical": all(item["byte_identical"] for item in tiers),
        "tiers": tiers,
    }
    atomic_write_json(root / "generation_verification.json", verification)
    return verification, bundles


def _route_summary(route: Route, width: int) -> dict[str, Any]:
    return {
        "query_id": route.query_id,
        "strategy": route.strategy,
        "route_hash": route.route_hash(width),
        "cost": route.cost,
        "expansions": route.expansions,
        "visited": route.visited,
        "peak_frontier": route.peak_frontier,
        "estimated_peak_bytes": route.estimated_peak_bytes,
        "work_units": route.work_units,
        "flow_key": route.flow_key,
        "jump_regions": list(route.jump_regions),
        "route_regions": list(route.route_regions),
    }


def _solver_crosscheck(bundle: Any, cfg: AppConfig) -> dict[str, Any]:
    astar = AStarSolver(bundle.map_data)
    flow_builder = FlowFieldBuilder(bundle.map_data)
    field_cache: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    max_deviation = 0
    deterministic = True
    for query in sorted(bundle.queries, key=lambda item: item.id):
        if not flow_eligible(bundle.map_data, query):
            continue
        astar_route = astar.solve(query)
        astar_again = astar.solve(query)
        region_id = bundle.map_data.region_of_cell(query.start)
        key = flow_key(region_id, query.target, query.mobility)
        if key not in field_cache:
            field_cache[key] = flow_builder.build(region_id, query.target, query.mobility)
        flow_route = field_cache[key].solve(query, bundle.map_data.width)
        flow_again = field_cache[key].solve(query, bundle.map_data.width)
        deviation = abs(astar_route.cost - flow_route.cost)
        max_deviation = max(max_deviation, deviation)
        scale = max(1, abs(astar_route.cost))
        within = float(deviation) <= cfg.cost_tolerance * scale
        same_a = astar_route.route_hash(bundle.map_data.width) == astar_again.route_hash(bundle.map_data.width)
        same_f = flow_route.route_hash(bundle.map_data.width) == flow_again.route_hash(bundle.map_data.width)
        deterministic = deterministic and same_a and same_f
        comparisons.append(
            {
                "query_id": query.id,
                "astar_cost": astar_route.cost,
                "flow_cost": flow_route.cost,
                "cost_deviation": deviation,
                "within_tolerance": within,
                "astar_repeat_hash_equal": same_a,
                "flow_repeat_hash_equal": same_f,
                "astar_route_hash": astar_route.route_hash(bundle.map_data.width),
                "flow_route_hash": flow_route.route_hash(bundle.map_data.width),
            }
        )
    return {
        "eligible_queries": len(comparisons),
        "all_within_tolerance": all(item["within_tolerance"] for item in comparisons),
        "max_cost_deviation": max_deviation,
        "repeat_solves_byte_identical": deterministic,
        "comparisons": comparisons,
    }


def _strategy_replan_demo(
    cfg: AppConfig,
    bundle: Any,
    budget: Budgets,
) -> dict[str, Any]:
    planner = StrategyPlanner(cfg)
    baseline = planner.plan(bundle.map_data, bundle.queries, budget)
    same = planner.plan(bundle.map_data, bundle.queries, budget)
    first = bundle.queries[0]
    extra = Query(
        id=f"q-{bundle.map_data.tier}-extra",
        tier=bundle.map_data.tier,
        unit_id=first.unit_id,
        start=first.start,
        target=first.target,
        mobility=first.mobility,
    )
    changed_queries = planner.plan(bundle.map_data, bundle.queries + [extra], budget)
    changed_budget = Budgets(
        tier=budget.tier,
        query_time_ms=budget.query_time_ms,
        query_memory_mib=budget.query_memory_mib,
        total_memory_mib=budget.total_memory_mib,
        max_threads=max(1, budget.max_threads - 1),
        precompute_time_ms=budget.precompute_time_ms,
    )
    budget_plan = planner.plan(bundle.map_data, bundle.queries, changed_budget)
    return {
        "baseline_recomputed": list(baseline.recomputed),
        "same_input_recomputed_count": len(same.recomputed),
        "same_input_reused_count": len(same.reused),
        "query_distribution_recomputed": list(changed_queries.recomputed),
        "query_distribution_reused_count": len(changed_queries.reused),
        "budget_change_recomputed_count": len(budget_plan.recomputed),
        "budget_change_reused_count": len(budget_plan.reused),
        "deterministic_decisions": all(
            baseline.decisions[key].to_dict() == same.decisions[key].to_dict()
            for key in baseline.decisions
        ),
    }


def _choose_edit_cell(bundle: Any, routes: dict[str, Route]) -> tuple[int, int]:
    endpoints = {query.start for query in bundle.queries} | {
        query.target for query in bundle.queries
    }
    counter: Counter[int] = Counter()
    for route in routes.values():
        for cell in route.path[1:-1]:
            if cell not in endpoints and bool(bundle.map_data.passable.flat[cell]):
                counter[int(cell)] += 1
    if not counter:
        for route in routes.values():
            for cell in route.path[1:]:
                if cell not in endpoints and bool(bundle.map_data.passable.flat[cell]):
                    counter[int(cell)] += 1
    if not counter:
        raise TacticalError(
            ErrorCode.MAP_SHAPE,
            "unable to select an interior edit cell",
        )
    cell = sorted(counter, key=lambda item: (-counter[item], item))[0]
    old = int(bundle.map_data.terrain.flat[cell])
    new = old + 1 if old < 9 else old - 1
    return cell, new


def _write_tier_artifacts(
    root: Path,
    tier: str,
    *,
    plan: Any,
    solve: dict[str, Any],
    crosscheck: dict[str, Any],
    incremental: dict[str, Any],
    consistency: dict[str, Any],
    budget: dict[str, Any] | None,
) -> None:
    tier_root = root / "tiers" / tier
    tier_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(tier_root / "strategy_plan.json", plan.to_dict())
    atomic_write_json(
        tier_root / "resource_ledger.json",
        {
            "workers_used": solve["workers_used"],
            "estimated_peak_bytes": solve["estimated_peak_bytes"],
            "downgrades": solve["downgrades"],
            "cache_evicted": solve["cache_evicted"],
            "reuse_events": solve["reuse_events"],
            "queries": solve["queries"],
        },
    )
    atomic_write_json(tier_root / "solver_crosscheck.json", crosscheck)
    atomic_write_json(
        tier_root / "incremental_closure.json",
        {
            "changed_cells": incremental["changed_cells"],
            "affected_regions": incremental["affected_regions"],
            "closure": incremental["closure"],
            "recomputed": incremental["recomputed"],
            "reused_nodes": incremental["reused_nodes"],
            "new_materialized_nodes": incremental["new_materialized_nodes"],
        },
    )
    atomic_write_json(tier_root / "consistency.json", consistency)
    if budget is not None:
        atomic_write_json(tier_root / "budget_downgrade.json", budget)


def _run_crash_and_recovery(store: StateStore, expected_hashes: dict[str, str]) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tactical_path.crash_worker",
            "--workspace",
            str(store.workspace),
        ],
        cwd=str(project_root),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 91:
        raise TacticalError(
            ErrorCode.SNAPSHOT_VERSION,
            "crash simulation did not terminate with the expected status",
            context={
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
    recovery = store.recover()
    verification = verify_recovered_hashes(recovery, expected_hashes)
    if not verification["all_match"]:
        raise TacticalError(
            ErrorCode.DEPENDENCY_STALE,
            "recovered routes differ from last committed state",
            context=verification,
        )
    result = {
        "forced_exit_code": completed.returncode,
        "recovery": recovery,
        "comparison_to_last_commit": verification,
    }
    atomic_write_json(store.state / "crash_recovery_demo.json", result)
    return result


def _budget_downgrade_demo(
    bundle: Any,
    cfg: AppConfig,
    decisions: dict[str, Any],
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    budget = cfg.budgets[bundle.map_data.tier]
    engine = TierEngine(bundle, cfg)
    engine.solve_batch(bundle.queries, decisions, budget)
    tight = engine.solve_batch(
        bundle.queries,
        decisions,
        budget,
        force=True,
        memory_budget_mib=cfg.tight_total_memory_mib,
    )
    before = Counter(entry.strategy for entry in engine.path_cache.values())
    routes = tight["routes"]
    missing = sorted(set(expected_hashes) - set(routes))
    hash_mismatches = sorted(
        query_id
        for query_id, route in routes.items()
        if route.route_hash(bundle.map_data.width) != expected_hashes.get(query_id)
    )
    return {
        "tier": bundle.map_data.tier,
        "tight_total_memory_mib": cfg.tight_total_memory_mib,
        "downgrade_priority": ["serial", "cache_eviction", "minimal_memory"],
        "downgrades": tight["downgrades"],
        "workers_used": tight["workers_used"],
        "cache_evicted_count": len(tight["cache_evicted"]),
        "queries_expected": len(expected_hashes),
        "queries_returned": len(routes),
        "missing_queries": missing,
        "route_hash_mismatches": hash_mismatches,
        "no_query_missing": not missing,
        "results_match_pre_downgrade": not hash_mismatches,
        "strategy_after_downgrade": dict(sorted(before.items())),
    }


def run_demo(config_path: str | Path, workspace: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = Path(workspace)
    _reset_directory(root)
    verification, bundles = _generation_verification(cfg, root)

    tier_reports: list[dict[str, Any]] = []
    tier_artifacts: list[dict[str, Any]] = []
    small_store: StateStore | None = None
    small_hashes: dict[str, str] = {}
    small_engine: TierEngine | None = None
    small_decisions: dict[str, Any] = {}
    for tier, spec in cfg.tiers.items():
        bundle = bundles[tier]
        budget = cfg.budgets[tier]
        planner = StrategyPlanner(cfg)
        plan = planner.plan(bundle.map_data, bundle.queries, budget)
        same_plan = planner.plan(bundle.map_data, bundle.queries, budget)
        replan = _strategy_replan_demo(cfg, bundle, budget)
        crosscheck = _solver_crosscheck(bundle, cfg)

        engine = TierEngine(bundle, cfg)
        solve = engine.solve_batch(bundle.queries, plan.decisions, budget)
        if len(solve["routes"]) != len(bundle.queries):
            raise TacticalError(
                ErrorCode.CLOSURE_MISMATCH,
                "initial solve dropped queries",
                context={
                    "tier": tier,
                    "expected": len(bundle.queries),
                    "actual": len(solve["routes"]),
                },
            )

        store = StateStore(root / "state" / tier)
        initial_commit = store.commit(engine, decisions=plan.decisions, label="initial-solve")
        edit_cell, new_terrain = _choose_edit_cell(bundle, solve["routes"])
        old_terrain = int(bundle.map_data.terrain.flat[edit_cell])
        edits = [MapEdit(cell=edit_cell, terrain=new_terrain)]
        incremental = engine.apply_edits(edits, plan.decisions)
        consistency = compare_incremental_with_full(
            engine,
            incremental["routes"],
            plan.decisions,
            cfg,
            affected_regions=incremental["affected_regions"],
        )
        if not consistency["all_consistent"]:
            raise TacticalError(
                ErrorCode.CLOSURE_MISMATCH,
                "incremental result diverged from full rebuild",
                context={
                    "tier": tier,
                    "first_divergence": consistency["first_divergence"],
                },
            )
        bundle.metadata["map"] = bundle.map_data.map_manifest()
        bundle.metadata["map_content_hash"] = bundle.map_data.content_hash
        save_bundle(bundle, root / "maps" / tier)
        final_commit = store.commit(engine, decisions=plan.decisions, label="post-incremental")
        final_hashes = engine.route_hashes()
        if tier == "small":
            small_store = store
            small_hashes = final_hashes
            small_engine = engine
            small_decisions = plan.decisions

        _write_tier_artifacts(
            root,
            tier,
            plan=plan,
            solve=solve,
            crosscheck=crosscheck,
            incremental=incremental,
            consistency=consistency,
            budget=None,
        )
        route_summaries = [
            _route_summary(engine.path_cache[query.id].route, bundle.map_data.width)
            for query in sorted(bundle.queries, key=lambda item: item.id)
        ]
        tier_summary = {
            "tier": tier,
            "size": spec.size,
            "map": bundle.map_data.map_manifest(),
            "query_count": len(bundle.queries),
            "strategy_plan": {
                "strategies": dict(
                    sorted(Counter(item.strategy for item in plan.decisions.values()).items())
                ),
                "initial_recomputed": len(plan.recomputed),
                "same_input_recomputed": len(same_plan.recomputed),
                "same_input_reused": len(same_plan.reused),
                **replan,
            },
            "solver_crosscheck": {
                "eligible_queries": crosscheck["eligible_queries"],
                "all_within_tolerance": crosscheck["all_within_tolerance"],
                "max_cost_deviation": crosscheck["max_cost_deviation"],
                "repeat_solves_byte_identical": crosscheck["repeat_solves_byte_identical"],
            },
            "resource_ledger": {
                "workers_used": solve["workers_used"],
                "estimated_peak_bytes": solve["estimated_peak_bytes"],
                "downgrades": solve["downgrades"],
                "query_count": len(route_summaries),
                "queries": route_summaries,
            },
            "incremental": {
                "edit": {
                    "cell": edit_cell,
                    "xy": [edit_cell % bundle.map_data.width, edit_cell // bundle.map_data.width],
                    "terrain_before": old_terrain,
                    "terrain_after": new_terrain,
                },
                "changed_cells": len(incremental["changed_cells"]),
                "affected_regions": incremental["affected_regions"],
                "closure_count": len(incremental["closure"]),
                "recomputed_count": len(incremental["recomputed"]),
                "reused_nodes_count": len(incremental["reused_nodes"]),
                "new_materialized_nodes_count": len(incremental["new_materialized_nodes"]),
                "closure_equals_recomputed": set(incremental["closure"])
                == set(incremental["recomputed"]),
            },
            "consistency": {
                "queries_compared": consistency["queries_compared"],
                "all_consistent": consistency["all_consistent"],
                "max_cost_deviation": consistency["max_cost_deviation"],
                "first_divergence": consistency["first_divergence"],
                "sequence_rule": consistency["sequence_rule"],
            },
            "persistence": {
                "initial_commit": initial_commit,
                "final_commit": final_commit,
                "final_route_hashes": dict(sorted(final_hashes.items())),
            },
        }
        tier_reports.append(tier_summary)
        tier_artifacts.append(
            {
                "tier": tier,
                "route_hashes": dict(sorted(final_hashes.items())),
                "closure_sha256": sha256_json(
                    {
                        "closure": incremental["closure"],
                        "recomputed": incremental["recomputed"],
                        "reused": incremental["reused_nodes"],
                    }
                ),
            }
        )

    if small_store is None or not small_hashes:
        raise TacticalError(
            ErrorCode.SNAPSHOT_VERSION,
            "small tier was not available for crash-recovery verification",
        )
    small_budget_demo = _budget_downgrade_demo(
        bundles["small"], cfg, small_decisions, small_hashes
    )
    _write_tier_artifacts(
        root,
        "small-budget-downgrade",
        plan=StrategyPlanner(cfg).plan(
            bundles["small"].map_data,
            bundles["small"].queries,
            cfg.budgets["small"],
        ),
        solve={
            "workers_used": 1,
            "estimated_peak_bytes": 0,
            "downgrades": small_budget_demo["downgrades"],
            "cache_evicted": [],
            "reuse_events": [],
            "queries": [],
        },
        crosscheck={
            "eligible_queries": 0,
            "all_within_tolerance": True,
            "max_cost_deviation": 0,
            "repeat_solves_byte_identical": True,
            "comparisons": [],
        },
        incremental={
            "changed_cells": [],
            "affected_regions": [],
            "closure": [],
            "recomputed": [],
            "reused_nodes": [],
            "new_materialized_nodes": [],
        },
        consistency={
            "queries_compared": 0,
            "all_consistent": True,
            "max_cost_deviation": 0,
            "first_divergence": None,
            "sequence_rule": "not applicable",
        },
        budget=small_budget_demo,
    )
    recovery = _run_crash_and_recovery(small_store, small_hashes)
    aggregate = {
        "tiers": [item["tier"] for item in tier_reports],
        "queries": sum(item["query_count"] for item in tier_reports),
        "route_count": sum(
            len(item["persistence"]["final_route_hashes"]) for item in tier_reports
        ),
        "all_generation_byte_identical": verification["all_byte_identical"],
        "all_consistency_checks_passed": all(
            item["consistency"]["all_consistent"] for item in tier_reports
        ),
        "all_crosschecks_within_tolerance": all(
            item["solver_crosscheck"]["all_within_tolerance"]
            for item in tier_reports
        ),
        "all_downgrades_complete": small_budget_demo["no_query_missing"]
        and small_budget_demo["results_match_pre_downgrade"],
        "recovery_hash_match": recovery["comparison_to_last_commit"]["all_match"],
        "tier_route_hash_digests": {
            item["tier"]: sha256_json(item["route_hashes"]) for item in tier_artifacts
        },
    }
    report = {
        "schema": "tactical-demo-report-v1",
        "config_sha256": file_sha256(config_path),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyyaml": getattr(yaml, "__version__", "unknown"),
            "determinism_envelope": "logical work units and estimated bytes; wall clocks excluded",
        },
        "generation_verification": verification,
        "tiers": tier_reports,
        "budget_downgrade_demo": {
            "tier": "small",
            "downgrade_priority": ["serial", "cache_eviction", "minimal_memory"],
            **small_budget_demo,
        },
        "forced_termination_recovery": recovery,
        "aggregate": aggregate,
    }
    report_bytes = canonical_bytes(report)
    atomic_write_bytes(root / "report.json", report_bytes)
    atomic_write_json(
        root / "report.sha256",
        {
            "sha256": sha256_bytes(report_bytes),
            "bytes": len(report_bytes),
            "schema": report["schema"],
        },
    )
    return report
