"""规则 -> 整形计划的编译器。

关键性质：
- 计划条目按确定性优先级排序，序列化不依赖 dict/set 遍历顺序；
- 每次编译产生单调递增的计划版本号，计划整体替换；
- 增量编译仅重算失效子集，未失效规则产物逐字节复用、产物版本号不变；
- 同一规则集合全量/增量编译的计划摘要必须一致，否则 E_INCR。

编译器缓存可通过 snapshot_cache/load_cache 持久化，使跨进程（CLI 各子命令、
强制终止重启）的增量编译依然严格复用未失效产物。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import canonical_bytes, sha256_hex, short_hash
from .errors import IncrementalError
from .graph import DependencyGraph
from .models import Rule


def _priority_key(rule: Rule) -> tuple:
    return (
        -max(len(str(p)) for p in rule.port_prefixes),
        tuple(sorted(rule.port_prefixes)),
        rule.dst_addr,
        rule.rule_id,
    )


def build_artifact(rule: Rule) -> dict[str, Any]:
    """单规则的规范化编译产物（不含版本号，保证可逐字节复用）。"""
    return {
        "artifact_version": 1,
        "rule_id": rule.rule_id,
        "match": {
            "dst_addr": rule.dst_addr,
            "port_prefixes": sorted(rule.port_prefixes),
        },
        "shaping": {
            "rate_kbps": rule.rate_kbps,
            "burst_bytes": rule.burst_bytes,
            "queue_limit_packets": rule.queue_limit_packets,
        },
        "upstream": sorted(rule.upstream),
        "actions": sorted(rule.actions),
        "priority": {
            "longest_port_prefix": max(rule.port_prefixes),
            "max_prefix_len": max(len(str(p)) for p in rule.port_prefixes),
        },
    }


def artifact_hash(artifact: dict[str, Any]) -> str:
    return sha256_hex(artifact)


@dataclass
class Plan:
    plan_version: int
    entries: list[dict[str, Any]]
    artifact_hashes: dict[str, str]
    artifact_versions: dict[str, int]
    plan_hash: str
    recompiled: list[str]
    reused: list[str]

    def payload(self) -> dict[str, Any]:
        return {
            "entries": self.entries,
            "artifact_hashes": {rid: self.artifact_hashes[rid]
                                for rid in sorted(self.artifact_hashes)},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "plan_hash": self.plan_hash,
            "entry_count": len(self.entries),
            "recompiled": self.recompiled,
            "reused": self.reused,
            "payload": self.payload(),
        }


@dataclass
class _CachedRule:
    rule_fingerprint: str
    artifact: dict[str, Any]
    artifact_hash: str
    artifact_version: int


class PlanCompiler:
    def __init__(self) -> None:
        self._cache: dict[str, _CachedRule] = {}
        self.plan_version = 0

    @staticmethod
    def _fingerprint(rule: Rule) -> str:
        return short_hash(rule.to_dict())

    # ---- 缓存持久化 -----------------------------------------------------
    def snapshot_cache(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "rules": {
                rid: {
                    "rule_fingerprint": item.rule_fingerprint,
                    "artifact": item.artifact,
                    "artifact_hash": item.artifact_hash,
                    "artifact_version": item.artifact_version,
                }
                for rid, item in self._cache.items()
            },
        }

    def load_cache(self, data: dict[str, Any]) -> None:
        self.plan_version = int(data.get("plan_version", 0))
        self._cache = {
            rid: _CachedRule(
                rule_fingerprint=item["rule_fingerprint"],
                artifact=item["artifact"],
                artifact_hash=item["artifact_hash"],
                artifact_version=int(item["artifact_version"]),
            )
            for rid, item in data.get("rules", {}).items()
        }

    # ---- 编译 -----------------------------------------------------------
    def compile(self, rules: list[Rule], graph: DependencyGraph,
                invalid_set: list[str] | None = None,
                next_plan_version: int | None = None) -> Plan:
        invalid = set(invalid_set or [r.rule_id for r in rules])
        current_ids = {r.rule_id for r in rules}
        for stale in set(self._cache) - current_ids:
            self._cache.pop(stale)

        recompiled: list[str] = []
        reused: list[str] = []
        new_cache: dict[str, _CachedRule] = {}
        for rule in rules:
            rid = rule.rule_id
            fingerprint = self._fingerprint(rule)
            cached = self._cache.get(rid)
            if rid not in invalid and cached and cached.rule_fingerprint == fingerprint:
                new_cache[rid] = cached
                reused.append(rid)
                continue
            artifact = build_artifact(rule)
            version = cached.artifact_version + 1 if cached else 1
            new_cache[rid] = _CachedRule(fingerprint, artifact,
                                         artifact_hash(artifact), version)
            recompiled.append(rid)
        self._cache = new_cache

        ordered = sorted(rules, key=_priority_key)
        entries = [new_cache[r.rule_id].artifact for r in ordered]
        hashes = {rid: item.artifact_hash for rid, item in sorted(new_cache.items())}
        artifact_versions = {rid: item.artifact_version
                             for rid, item in sorted(new_cache.items())}
        payload = {
            "entries": entries,
            "artifact_hashes": {rid: hashes[rid] for rid in sorted(hashes)},
        }
        plan_hash = sha256_hex(payload)

        if next_plan_version is None:
            self.plan_version += 1
            plan_version = self.plan_version
        else:
            if next_plan_version <= self.plan_version:
                raise ValueError(
                    f"计划版本号必须单调递增：现有 {self.plan_version}，"
                    f"请求 {next_plan_version}")
            self.plan_version = next_plan_version
            plan_version = next_plan_version

        return Plan(plan_version, entries, hashes, artifact_versions,
                    plan_hash, sorted(recompiled), sorted(reused))

    # ---- 一致性 ---------------------------------------------------------
    @staticmethod
    def full_recompile(rules: list[Rule], graph: DependencyGraph) -> Plan:
        fresh = PlanCompiler()
        return fresh.compile(rules, graph)

    @staticmethod
    def assert_incremental_matches_full(incremental: Plan, rules: list[Rule],
                                        graph: DependencyGraph) -> dict[str, Any]:
        full = PlanCompiler.full_recompile(rules, graph)
        all_ids = sorted(set(incremental.artifact_hashes) | set(full.artifact_hashes))
        for rid in all_ids:
            inc = incremental.artifact_hashes.get(rid)
            ful = full.artifact_hashes.get(rid)
            if inc is None:
                raise IncrementalError(
                    "增量结果缺少规则产物",
                    details={"first_diff_rule": rid, "field": "artifact_hashes",
                             "incremental": None, "full": ful})
            if ful is None:
                raise IncrementalError(
                    "增量结果多出未知规则产物",
                    details={"first_diff_rule": rid, "field": "artifact_hashes",
                             "incremental": inc, "full": None})
            if inc != ful:
                field_name = _first_artifact_field_diff(
                    incremental.entries, full.entries, rid)
                raise IncrementalError(
                    "增量编译与全量编译产物不一致",
                    details={"first_diff_rule": rid, "field": field_name,
                             "incremental_hash": inc, "full_hash": ful})
        if incremental.plan_hash != full.plan_hash:
            raise IncrementalError(
                "计划摘要不一致",
                details={"first_diff_rule": all_ids[0] if all_ids else "<plan>",
                         "field": "plan_hash",
                         "incremental_hash": incremental.plan_hash,
                         "full_hash": full.plan_hash})
        return {
            "ok": True,
            "plan_hash": full.plan_hash,
            "recompiled_count": len(incremental.recompiled),
            "reused_count": len(incremental.reused),
            "recompiled": incremental.recompiled,
            "reused": incremental.reused,
        }


def _first_artifact_field_diff(inc_entries: list[dict[str, Any]],
                               full_entries: list[dict[str, Any]],
                               rule_id: str) -> str:
    inc = next((e for e in inc_entries if e["rule_id"] == rule_id), None)
    ful = next((e for e in full_entries if e["rule_id"] == rule_id), None)
    if inc is None or ful is None:
        return "entries"
    for key in sorted(set(inc) | set(ful)):
        if canonical_bytes(inc.get(key)) != canonical_bytes(ful.get(key)):
            return key
    return "artifact"
