"""工作台门面：串联规则加载、变更、编译、账本、整形、快照与恢复的完整业务流。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .applog import StructuredLogger
from .canonical import short_hash
from .changeset import ChangeSet, apply_changeset
from .compiler import Plan, PlanCompiler
from .graph import DependencyGraph
from .ledger import Ledger, RecoveryReport
from .models import Rule
from .replay import replay_snapshot
from .ruleset import build_graph, load_rules, ruleset_version
from .shaper import (SimulationResult, deterministic_packets,
                     run_virtual, verify_throughput)
from .snapshot import Snapshot, diff_versions, load_snapshot, save_snapshot


class Workbench:
    def __init__(self, db_path: str | Path = "data/qos_ledger.db",
                 log_path: str | Path = "logs/qos.jsonl"):
        self.ledger = Ledger(db_path)
        self.logger = StructuredLogger(log_path)
        self.compiler = PlanCompiler()
        self.rules: list[Rule] = []
        self.graph: DependencyGraph | None = None
        self.ruleset_ver: str | None = None
        self.graph_ver: str | None = None
        self.plan: Plan | None = None
        self.last_simulation: SimulationResult | None = None
        self.last_throughput_report: dict[str, Any] | None = None
        cached = self.ledger.load_compiler_cache()
        if cached:
            self.compiler.load_cache(cached)

    # ---- 加载与首次编译 -------------------------------------------------
    def load(self, rule_path: str | Path) -> dict[str, Any]:
        rules = load_rules(rule_path)  # E_RULE / E_DEPEND
        graph = build_graph(rules)
        self.rules = rules
        self.graph = graph
        self.ruleset_ver = ruleset_version(rules)
        self.graph_ver = short_hash(graph.topo_order())

        self.ledger.save_ruleset(self.ruleset_ver, rules)
        self.ledger.save_graph(self.graph_ver, self.ruleset_ver, graph.topo_order())

        op_id = self.ledger.begin_operation("compile", [r.rule_id for r in rules])
        plan_version = self.ledger.next_plan_version()
        plan = self.compiler.compile(rules, graph,
                                     invalid_set=[r.rule_id for r in rules],
                                     next_plan_version=plan_version)
        self.ledger.save_plan(plan, self.graph_ver, self.ruleset_ver)
        self.ledger.commit_operation(op_id)
        self.ledger.save_compiler_cache(self.compiler.snapshot_cache())
        self.plan = plan
        self.logger.info("load", ruleset_version=self.ruleset_ver,
                         plan_version=plan.plan_version, rule_count=len(rules))
        return {"ruleset_version": self.ruleset_ver,
                "graph_version": self.graph_ver,
                "plan": plan.to_dict()}

    # ---- 增量变更 -------------------------------------------------------
    def apply_change(self, changeset: ChangeSet) -> dict[str, Any]:
        if self.graph is None:
            raise RuntimeError("请先 load 规则集")
        applied = apply_changeset(self.rules, changeset)  # E_REF/E_RULE/E_DEPEND
        self.rules = applied.rules
        self.graph = applied.graph
        self.ruleset_ver = ruleset_version(self.rules)
        self.graph_ver = short_hash(self.graph.topo_order())
        self.ledger.save_ruleset(self.ruleset_ver, self.rules)
        self.ledger.save_graph(self.graph_ver, self.ruleset_ver,
                               self.graph.topo_order())

        op_id = self.ledger.begin_operation(
            "compile", applied.invalidation_sequence)
        plan_version = self.ledger.next_plan_version()
        plan = self.compiler.compile(
            self.rules, self.graph,
            invalid_set=applied.invalidation_sequence,
            next_plan_version=plan_version)
        self.ledger.save_plan(plan, self.graph_ver, self.ruleset_ver)
        self.ledger.save_invalidation(plan_version,
                                      applied.invalidation_sequence,
                                      applied.trace, applied.change_summary)
        self.ledger.commit_operation(op_id)
        self.ledger.save_compiler_cache(self.compiler.snapshot_cache())
        self.plan = plan
        self.logger.info(
            "apply_change", plan_version=plan.plan_version,
            invalidation=applied.invalidation_sequence,
            recompiled=plan.recompiled, reused=plan.reused,
            summary=applied.change_summary)
        return {
            "change_summary": applied.change_summary,
            "invalidation_sequence": applied.invalidation_sequence,
            "trace": [t.to_dict() for t in applied.trace],
            "plan": plan.to_dict(),
        }

    def verify_incremental(self) -> dict[str, Any]:
        assert self.graph is not None and self.plan is not None
        result = PlanCompiler.assert_incremental_matches_full(
            self.plan, self.rules, self.graph)  # E_INCR
        self.logger.info("verify_incremental", **{k: v for k, v in result.items()
                                                  if k != "plan_hash"})
        return result

    # ---- 整形模拟 -------------------------------------------------------
    def simulate(self, count: int = 40, size: int = 1024,
                 dst_ports: list[int] | None = None,
                 arrival_gap_us: int = 50_000) -> dict[str, Any]:
        assert self.plan is not None and self.graph is not None
        ports = dst_ports or sorted({
            p for r in self.rules for p in r.port_prefixes})
        packets = deterministic_packets(count, size, ports)
        op_id = self.ledger.begin_operation(
            "shape", sorted({r.rule_id for r in self.rules}))
        result = run_virtual(self.plan.entries, packets,
                             self.plan.plan_version, arrival_gap_us)
        report = verify_throughput(result, self.plan.entries)  # E_SHAPE
        self.ledger.save_observation(self.plan.plan_version,
                                     result.digest(), result.records())
        self.ledger.commit_operation(op_id)
        self.last_simulation = result
        self.last_throughput_report = report
        self.logger.info("simulate", plan_version=self.plan.plan_version,
                         digest=result.digest())
        return {"digest": result.digest(), "throughput": report,
                "records": result.records()}

    # ---- 崩溃与恢复 -----------------------------------------------------
    def simulate_crash_during_compile(self) -> int:
        """测试辅助：制造一条永不 COMMIT 的 compile 操作后“强制终止”。"""
        assert self.graph is not None
        op_id = self.ledger.begin_operation(
            "compile", [r.rule_id for r in self.rules])
        self.ledger.crash_without_commit(op_id)
        self.logger.error("forced_termination", "E_UNKNOWN",
                          op_id=op_id, op_type="compile",
                          note="模拟强制终止，OPEN 操作未提交")
        return op_id

    def recover(self) -> dict[str, Any]:
        report: RecoveryReport = self.ledger.recover()
        self.logger.info("recover", **report.to_dict())
        return report.to_dict()

    def resolve_unknown(self) -> int:
        """未知规则重新编译/重跑后，登记 OPEN 操作已解决。"""
        return self.ledger.mark_open_operations_resolved()

    # ---- 从账本重建内存态（恢复后继续业务） --------------------------------
    def hydrate_from_ledger(self) -> dict[str, Any]:
        """依据最近一次完整提交的规则集/图/计划记录重建内存状态。"""
        latest = self.ledger.latest_plan()
        if latest is None:
            raise RuntimeError("账本中没有已提交计划，无法重建内存态")
        import json as _json
        rules = parse_snapshot_rules_type(_json.loads(
            self.ledger.conn.execute(
                "SELECT rules_json FROM rulesets WHERE ruleset_version=?",
                (latest["ruleset_version"],)).fetchone()["rules_json"]))
        graph = build_graph(rules)
        self.rules = rules
        self.graph = graph
        self.ruleset_ver = latest["ruleset_version"]
        self.graph_ver = latest["graph_version"]
        cached = self.ledger.load_compiler_cache()
        if cached:
            self.compiler.load_cache(cached)
        # 以全量编译恢复计划对象，再对齐已提交的全局版本号。
        plan = PlanCompiler.full_recompile(rules, graph)
        self.plan = Plan(
            plan_version=latest["plan_version"], entries=plan.entries,
            artifact_hashes=plan.artifact_hashes,
            artifact_versions=plan.artifact_versions,
            plan_hash=latest["plan_hash"],
            recompiled=[r.rule_id for r in rules], reused=[])
        self.compiler.plan_version = latest["plan_version"]
        return {"plan_version": self.plan.plan_version,
                "rule_count": len(rules)}

    # ---- 快照 -----------------------------------------------------------
    def create_snapshot(self, path: str | Path) -> dict[str, Any]:
        assert self.plan is not None and self.graph is not None
        latest_obs = self.ledger.latest_observation(self.plan.plan_version)
        obs_digest = (json_safe_loads(latest_obs["digest_json"])
                      if latest_obs else {})
        invalidation = self.ledger.execute(
            "SELECT sequence_json, trace_json, summary_json FROM invalidations "
            "ORDER BY invalidation_id DESC LIMIT 1").fetchone()
        if invalidation is not None:
            sequence = json_safe_loads(invalidation["sequence_json"])
            trace = json_safe_loads(invalidation["trace_json"])
            change_summary = json_safe_loads(invalidation["summary_json"])
        else:
            sequence, trace, change_summary = [], [], {}
        snap = Snapshot(
            ruleset_version=self.ruleset_ver,  # type: ignore[arg-type]
            graph_version=self.graph_ver,  # type: ignore[arg-type]
            plan_version=self.plan.plan_version,
            rules=[r.to_dict() for r in self.rules],
            topo=self.graph.topo_order(),
            artifact_hashes=dict(self.plan.artifact_hashes),
            artifact_versions=dict(self.plan.artifact_versions),
            plan_hash=self.plan.plan_hash,
            invalidation_sequence=sequence,
            invalidation_trace=trace,
            observation_digest=obs_digest,
            change_summary=change_summary,
        )
        save_snapshot(path, snap)
        self.ledger.register_snapshot(str(path), snap.to_dict())
        self.logger.info("snapshot_create", path=str(path),
                         plan_version=self.plan.plan_version)
        return snap.to_dict()

    def restore_snapshot(self, path: str | Path,
                         require_versions: bool = True) -> dict[str, Any]:
        snap = load_snapshot(path, require_versions=require_versions)  # E_SNAP
        diff = diff_versions(snap, self.ruleset_ver or "",
                             self.graph_ver or "", self.rules)
        if not diff["version_match"]:
            # 版本不一致：整体替换当前规则集、依赖图与计划状态，禁止增量合并。
            rules = parse_snapshot_rules(snap)
            graph = build_graph(rules)
            self.rules = rules
            self.graph = graph
            self.ruleset_ver = snap.ruleset_version
            self.graph_ver = snap.graph_version
            # 计划整体替换：以快照条目重建编译器缓存，版本号采用快照值。
            self.compiler = PlanCompiler()
            plan = self._rebuild_plan_from_snapshot(snap, graph)
            self.plan = plan
            self.ledger.save_ruleset(self.ruleset_ver, rules)
            self.ledger.save_graph(self.graph_ver, self.ruleset_ver,
                                   graph.topo_order())
            self.ledger.save_plan(plan, self.graph_ver, self.ruleset_ver)
        self.logger.info("snapshot_restore", path=str(path), **diff)
        return {"diff": diff, "snapshot": snap.to_dict()}

    def replay(self, path: str | Path,
               baseline_rules: list[Rule] | None = None) -> dict[str, Any]:
        snap = load_snapshot(path)  # E_SNAP
        result = replay_snapshot(snap, baseline_rules)  # E_REPLAY
        self.logger.info("replay", path=str(path), **result)
        return result

    def _rebuild_plan_from_snapshot(self, snap: Snapshot,
                                    graph: DependencyGraph) -> Plan:
        # 用全量编译恢复缓存，再把全局计划版本对齐到快照版本（整体替换）。
        plan = PlanCompiler.full_recompile(self.rules, graph)
        aligned = Plan(
            plan_version=snap.plan_version,
            entries=plan.entries,
            artifact_hashes=plan.artifact_hashes,
            artifact_versions=plan.artifact_versions,
            plan_hash=snap.plan_hash,
            recompiled=[r.rule_id for r in self.rules],
            reused=[],
        )
        if aligned.plan_hash != plan.plan_hash:
            # 快照内容与重编译不一致时不静默处理，交由 replay 命令定位。
            self.logger.error("snapshot_plan_mismatch", "E_REPLAY",
                              snapshot_hash=snap.plan_hash,
                              replay_hash=plan.plan_hash)
        self.compiler.plan_version = snap.plan_version
        self.ledger.save_compiler_cache(self.compiler.snapshot_cache())
        return aligned

    def close(self) -> None:
        self.ledger.close()


def parse_snapshot_rules(snap: Snapshot) -> list[Rule]:
    from .models import parse_rules
    return parse_rules(snap.rules)


def parse_snapshot_rules_type(rule_dicts: list[dict]) -> list[Rule]:
    from .models import parse_rules
    return parse_rules(rule_dicts)


def json_safe_loads(text: str) -> Any:
    import json
    return json.loads(text)
