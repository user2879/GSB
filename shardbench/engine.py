"""路由工作台引擎：编排一个完整路由轮次的所有环节。

轮次流程（确定性）：
1. 加载并强校验配置、推导四类版本；
2. 与上一版本比对，必要时整体替换缓存视图并推导失效集合；
3. 按稳定拓扑顺序编译路由计划，合并上游派生目标，命中版本化缓存则复用；
4. 在全局预算 + 每库配额约束下并行探查，记录任务轨迹与结局；
5. 单元结果、任务轨迹、版本、缓存/历史摘要在单事务内写入 SQLite 账本；
6. 原子写入会话快照，记录账本进度游标。
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any

from .cache import CacheEntry, RouteCache
from .config import ShardConfig, load_config
from .determinism import content_digest, stable_targets
from .invalidation import InvalidationResult, propagate, seeds_from_rule_change
from .ledger import Ledger
from .router import RoutePlan, Router
from .scheduler import ProbeTask, Scheduler, dbs_for_targets
from .snapshot import build_snapshot, save_snapshot
from .versioning import VersionState, derive_versions, diff_versions, initial_versions


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


@dataclass
class RoundResult:
    round_id: int
    versions: VersionState
    plans: dict[str, dict[str, Any]]
    invalidation: dict[str, Any]
    traces: list[dict[str, Any]]
    replacement: dict[str, Any] | None
    cache_digest: str
    history_digest: str
    scheduler_verification: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)


class Engine:
    def __init__(
        self,
        config_path: str,
        units_path: str | None,
        ledger_path: str,
        snapshot_path: str,
        logger=None,
    ) -> None:
        self.config_path = config_path
        self.units_path = units_path
        self.ledger = Ledger(ledger_path)
        self.snapshot_path = snapshot_path
        self.logger = logger
        self.cache = RouteCache()
        self._previous_versions: VersionState | None = None

    def close(self) -> None:
        self.ledger.close()

    def _log(self, event: str, **fields: Any) -> None:
        if self.logger is not None:
            self.logger.log(event, **fields)

    def _bootstrap_versions(self, config: ShardConfig) -> VersionState:
        last_id = self.ledger.last_committed_round_id()
        if last_id is None:
            return derive_versions(initial_versions(), config)
        record = self.ledger.load_round(last_id)["round"]
        return derive_versions(self._state_from_record(record), config)

    def _load_last_version_state(self) -> VersionState | None:
        last_id = self.ledger.last_committed_round_id()
        if last_id is None:
            return None
        record = self.ledger.load_round(last_id)["round"]
        return self._state_from_record(record)

    @staticmethod
    def _state_from_record(record: dict[str, Any]) -> VersionState:
        return VersionState(
            config_version=record["config_version"],
            ring_version=record["ring_version"],
            dependency_version=record["dependency_version"],
            budget_version=record["budget_version"],
            config_digest=record["config_digest"],
            ring_digest=record["ring_digest"],
            dependency_digest=record["dependency_digest"],
            budget_digest=record["budget_digest"],
        )

    def run_round(
        self,
        *,
        changed_tables: list[str] | None = None,
        fault_overrides: dict[str, dict[str, bool]] | None = None,
    ) -> RoundResult:
        config = load_config(self.config_path, self.units_path)
        prior_state = self._load_last_version_state()
        versions = self._bootstrap_versions(config)
        current_view = self.cache.view
        events: list[dict[str, Any]] = []

        replacement = None
        if current_view is None and prior_state is None:
            self.cache.initialize(
                versions.config_version,
                versions.ring_version,
                versions.dependency_version,
            )
        else:
            baseline = prior_state or self._previous_versions
            if current_view is None:
                self.cache.initialize(
                    baseline.config_version,
                    baseline.ring_version,
                    baseline.dependency_version,
                )
            changed_flags = diff_versions(baseline, versions)
            changed_rules = [
                name for name, changed in changed_flags.items() if changed
            ]
            affected_targets = self._affected_targets(config, changed_tables or [])
            replacement = self.cache.replace_view(
                config_version=versions.config_version,
                ring_version=versions.ring_version,
                dependency_version=versions.dependency_version,
                changed_rules=changed_rules,
                affected_targets=affected_targets,
            ).to_dict()
            events.append({"event": "cache_view_replaced", "payload": replacement})
            self._log("cache_view_replaced", **replacement)

        if changed_tables:
            seeds = seeds_from_rule_change(config, changed_tables)
        else:
            seeds = []
        invalidation: InvalidationResult = propagate(
            config,
            seeds,
            seed_reason="分片规则或哈希环参数演进",
        )
        invalidated_set = set(invalidation.invalidated)
        events.append(
            {"event": "invalidation_propagated", "payload": invalidation.to_dict()}
        )

        router = Router(config)
        plans: dict[str, RoutePlan] = {}
        unit_records: list[dict[str, Any]] = []
        version_tuple = (
            versions.config_version,
            versions.ring_version,
            versions.dependency_version,
        )
        for unit_id in config.topo_order:
            plan = self._compile_with_cache(router, config, unit_id, version_tuple)
            plan = self._merge_dependencies(config, unit_id, plan, plans)
            plans[unit_id] = plan
            record = plan.to_dict()
            record["plan"] = plan.to_dict()
            record["recomputed"] = unit_id in invalidated_set
            unit_records.append(record)
            entry = CacheEntry(
                entry_key=self._entry_key(unit_id, plan.route_key),
                unit_id=unit_id,
                route_key=plan.route_key,
                config_version=versions.config_version,
                ring_version=versions.ring_version,
                dependency_version=versions.dependency_version,
                targets=tuple(plan.targets),
                summary=plan.summary(),
                plan_digest=content_digest(plan.to_dict()),
            )
            self.cache.put(entry)

        traces, verification = self._schedule(config, plans, fault_overrides or {})
        events.append(
            {"event": "scheduler_verification", "payload": verification}
        )

        round_id = self.ledger.begin_round()
        cache_digest = self.cache.view.digest() if self.cache.view else ""
        history_digest = content_digest(
            [record["summary"] for record in sorted(unit_records, key=lambda item: item["unit_id"])]
        )
        config_json, units_json = self._canonical_inputs()
        payload_digest = self.ledger.commit_round(
            round_id,
            committed_at=_now_iso(),
            versions=versions.as_dict(),
            cache_digest=cache_digest,
            history_digest=history_digest,
            config_json=config_json,
            units_json=units_json,
            invalidated=invalidation.invalidated,
            unit_records=unit_records,
            trace_records=traces,
            events=events,
        )
        snapshot = build_snapshot(
            versions=versions,
            cache_digest=cache_digest,
            history_digest=history_digest,
            ledger_cursor=round_id,
            config_path=self.config_path,
            units_path=self.units_path,
        )
        save_snapshot(self.snapshot_path, snapshot)
        self._previous_versions = versions
        self._log(
            "round_committed",
            round_id=round_id,
            payload_digest=payload_digest,
            outcomes={trace["unit_id"]: trace["outcome_code"] for trace in traces},
        )
        return RoundResult(
            round_id=round_id,
            versions=versions,
            plans={unit_id: plan.to_dict() for unit_id, plan in plans.items()},
            invalidation=invalidation.to_dict(),
            traces=traces,
            replacement=replacement,
            cache_digest=cache_digest,
            history_digest=history_digest,
            scheduler_verification=verification,
            events=events,
        )

    def _entry_key(self, unit_id: str, route_key: str) -> str:
        return f"{unit_id}::{route_key}"

    def _compile_with_cache(
        self,
        router: Router,
        config: ShardConfig,
        unit_id: str,
        version_tuple: tuple[int, int, int],
    ) -> RoutePlan:
        unit = config.units[unit_id]
        probe_key = unit.predicate_text if unit.predicate_text is not None else unit_id
        entry_key = f"{unit_id}::probe::{content_digest(probe_key)[:16]}"
        cached, miss_reason = self.cache.get(
            entry_key, current_versions=version_tuple
        )
        if cached is not None:
            self._log(
                "cache_hit",
                unit_id=unit_id,
                entry_key=entry_key,
            )
        plan = router.compile(unit_id)
        if cached is not None and cached.plan_digest == content_digest(plan.to_dict()):
            plan.derivation = list(plan.derivation)
            self._log("cache_reused", unit_id=unit_id)
        elif cached is not None:
            self._log(
                "cache_version_miss",
                unit_id=unit_id,
                reason=miss_reason or "计划摘要不一致",
            )
        else:
            self._log("cache_miss", unit_id=unit_id, reason=miss_reason)
        return plan

    def _merge_dependencies(
        self,
        config: ShardConfig,
        unit_id: str,
        plan: RoutePlan,
        compiled: dict[str, RoutePlan],
    ) -> RoutePlan:
        return _merge_plan(config, unit_id, plan, compiled)

    def _schedule(
        self,
        config: ShardConfig,
        plans: dict[str, RoutePlan],
        fault_overrides: dict[str, dict[str, bool]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tasks: list[ProbeTask] = []
        readiness = self._readiness_map(config)
        for index, unit_id in enumerate(config.topo_order):
            plan = plans[unit_id]
            override = fault_overrides.get(unit_id, {})
            tasks.append(
                ProbeTask(
                    task_id=f"task-{index:03d}-{unit_id}",
                    unit_id=unit_id,
                    dbs=tuple(dbs_for_targets(plan.targets)),
                    ready=readiness[unit_id] and not override.get("force_dependency_not_ready", False),
                    config_present=not override.get("force_config_missing", False),
                    order_index=index,
                )
            )
        scheduler = Scheduler(config)
        traces = scheduler.run(tasks)
        verification = scheduler.verify_never_exceeded()
        return [trace.to_dict() for trace in traces], verification

    def _readiness_map(self, config: ShardConfig) -> dict[str, bool]:
        ready: dict[str, bool] = {}
        for unit_id in config.topo_order:
            unit = config.units[unit_id]
            ready[unit_id] = all(ready[dep] for dep in unit.depends_on)
        return ready

    def _affected_targets(
        self, config: ShardConfig, changed_tables: list[str]
    ) -> list[str]:
        targets: set[str] = set()
        for table_name in changed_tables:
            table = config.tables.get(table_name)
            if table is None:
                continue
            from .determinism import all_physical_targets
            targets.update(
                all_physical_targets(table.db_count, table.tables_per_db)
            )
        return sorted(targets)

    def _canonical_inputs(self) -> tuple[str, str]:
        with open(self.config_path, "r", encoding="utf-8") as handle:
            config_json = _canonical_text(handle.read())
        units_json = "{}"
        if self.units_path:
            with open(self.units_path, "r", encoding="utf-8") as handle:
                units_json = _canonical_text(handle.read())
        return config_json, units_json


def _canonical_text(raw: str) -> str:
    import json
    from .determinism import canonical_json
    return canonical_json(json.loads(raw))


def full_recompute(
    config: ShardConfig,
    *,
    previous_plans: dict[str, RoutePlan] | None = None,
) -> dict[str, dict[str, Any]]:
    """全量重算所有路由单元，供与增量结果逐项比较。"""
    router = Router(config)
    compiled: dict[str, RoutePlan] = {}
    result: dict[str, dict[str, Any]] = {}
    for unit_id in config.topo_order:
        plan = router.compile(unit_id)
        plan = _merge_plan(config, unit_id, plan, compiled)
        compiled[unit_id] = plan
        result[unit_id] = plan.to_dict()
    return result


def compile_all(config: ShardConfig) -> dict[str, dict[str, Any]]:
    """按稳定拓扑编译全部单元并完成依赖合并（增量侧的确定性基线）。"""
    return full_recompute(config)


def _merge_plan(
    config: ShardConfig,
    unit_id: str,
    plan: RoutePlan,
    compiled: dict[str, RoutePlan],
) -> RoutePlan:
    unit = config.units[unit_id]
    if not unit.depends_on:
        return plan
    merged = set(plan.targets)
    derivation = list(plan.derivation)
    for dep in unit.depends_on:
        upstream = compiled[dep]
        merged.update(upstream.targets)
        derivation.append(
            f"派生自 {dep}：并入目标 {','.join(upstream.targets) or '∅'}"
        )
        if upstream.is_broadcast:
            plan.is_broadcast = True
            reason = f"上游 {dep} 为广播，派生单元同步广播"
            plan.broadcast_reason = (
                reason
                if not plan.broadcast_reason
                else plan.broadcast_reason + "；" + reason
            )
    table = config.table_of(unit_id)
    if plan.is_broadcast:
        from .algorithms import broadcast_decision
        decision = broadcast_decision(
            plan.broadcast_reason or "派生广播",
            table.algorithm,
            table.params,
            table.db_count,
            table.tables_per_db,
            config.rings.get(table.name).ring_version
            if table.name in config.rings
            else None,
        )
        plan.decisions = [decision.to_dict()]
        merged = set(decision.targets)
    plan.targets = stable_targets(merged)
    plan.derivation = derivation
    return plan
