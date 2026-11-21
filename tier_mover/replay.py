"""恢复后的一致性重放校验。

重放以账本中已完整提交的迁移记录为权威注入序列，从初始全热层放置出发，
按记录顺序逐条应用并核对：
- 对象键、策略版本、内容摘要、迁移方向、源/目标层、字节数
- 应用过程中的归属层与层容量不变量
- 最终归属、各层占用、对象数与成本同账本逐项比较
任何分歧立即以 E_REPLAY 报出首个不一致对象键与字段；
策略版本/摘要不一致在前置校验报 E_VERSION，同样定位到首个分歧点。
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import ReplayError, VersionConflictError
from .ledger import Ledger
from .manifest import Manifest


@dataclass
class ReplayResult:
    steps_compared: int
    batches_compared: int
    final_occupancy: dict[str, int]
    final_cost: dict[str, int]
    final_placement: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "steps_compared": self.steps_compared,
            "batches_compared": self.batches_compared,
            "final_occupancy": self.final_occupancy,
            "final_cost": self.final_cost,
            "final_placement": self.final_placement,
        }


class ReplayService:
    def __init__(self, ledger: Ledger, manifest: Manifest):
        self.ledger = ledger
        self.manifest = manifest

    def _check_spec(self) -> None:
        version = self.ledger.active_spec_version()
        blob = self.ledger.spec_blob(version)
        if blob.get("spec_digest") != self.manifest.spec_digest:
            raise VersionConflictError(
                "重放前置校验失败：当前清单摘要与账本活动策略版本不一致",
                field="spec_digest",
                details={"ledger_spec_digest": blob.get("spec_digest"),
                         "input_spec_digest": self.manifest.spec_digest,
                         "spec_version": version})
        if version != self.manifest.spec_version:
            raise VersionConflictError(
                f"策略版本不一致：账本 {version}，输入 {self.manifest.spec_version}",
                field="spec_version",
                details={"ledger_version": version,
                         "input_version": self.manifest.spec_version})

    def run(self) -> ReplayResult:
        self._check_spec()
        self.ledger.verify_chain()
        recorded = self.ledger.committed_migrations()

        obj_by_key = {o.key: o for o in self.manifest.objects}
        hot_id = self.manifest.hot_tier.tier_id
        placement = {o.key: hot_id for o in self.manifest.objects}
        occupancy = {t.tier_id: 0 for t in self.manifest.tiers}
        occupancy[hot_id] = sum(o.size_bytes for o in self.manifest.objects)
        counts = {t.tier_id: 0 for t in self.manifest.tiers}
        counts[hot_id] = len(self.manifest.objects)

        # rebuild 恢复会把未提交对象的归属直接改写为目标层，重放须先对齐
        # 账本当前对象表中“无任何已提交迁移记录”的对象归属。
        migrated_keys = {row["key"] for row in recorded}
        for row in self.ledger.conn.execute(
                "SELECT key, tier_id FROM object_state ORDER BY key").fetchall():
            if row["key"] not in migrated_keys:
                expected_tier = placement[row["key"]]
                if row["tier_id"] != expected_tier:
                    obj = obj_by_key[row["key"]]
                    occupancy[expected_tier] -= obj.size_bytes
                    counts[expected_tier] -= 1
                    occupancy[row["tier_id"]] += obj.size_bytes
                    counts[row["tier_id"]] += 1
                    placement[row["key"]] = row["tier_id"]

        for index, row in enumerate(recorded):
            key = row["key"]
            basis = f"ledger.migrations[{index}]"
            if key not in obj_by_key:
                raise ReplayError(
                    "账本迁移引用了清单中不存在的对象", object_key=key,
                    mismatch_field="key", basis=basis,
                    expected="object in manifest", actual=key)
            obj = obj_by_key[key]
            if row["spec_version"] != self.manifest.spec_version:
                raise ReplayError(
                    "策略版本分歧导致不可复现", object_key=key,
                    mismatch_field="spec_version", basis=basis,
                    expected=row["spec_version"],
                    actual=self.manifest.spec_version)
            if row["content_digest"] != obj.content_digest:
                raise ReplayError(
                    "内容摘要分歧导致不可复现", object_key=key,
                    mismatch_field="content_digest", basis=basis,
                    expected=row["content_digest"],
                    actual=obj.content_digest)
            if row["size_bytes"] != obj.size_bytes:
                raise ReplayError(
                    "对象字节数分歧", object_key=key,
                    mismatch_field="size_bytes", basis=basis,
                    expected=row["size_bytes"], actual=obj.size_bytes)
            if placement[key] != row["source_tier"]:
                raise ReplayError(
                    "重放归属层与迁移源层分歧", object_key=key,
                    mismatch_field="source_tier", basis=basis,
                    expected=row["source_tier"], actual=placement[key])
            known_tiers = {t.tier_id for t in self.manifest.tiers}
            if row["target_tier"] not in known_tiers:
                raise ReplayError(
                    "迁移目标层不在清单中", object_key=key,
                    mismatch_field="target_tier", basis=basis,
                    expected=sorted(known_tiers), actual=row["target_tier"])
            # 应用迁移并重算占用/归属
            placement[key] = row["target_tier"]
            occupancy[row["source_tier"]] -= row["size_bytes"]
            occupancy[row["target_tier"]] += row["size_bytes"]
            counts[row["source_tier"]] -= 1
            counts[row["target_tier"]] += 1
            tier = self.manifest.tier(row["target_tier"])
            if occupancy[row["target_tier"]] > tier.capacity_bytes:
                raise ReplayError(
                    "重放过程中层容量被突破", object_key=key,
                    mismatch_field="used_bytes", basis=basis,
                    expected=f"<= {tier.capacity_bytes}",
                    actual=occupancy[row["target_tier"]])

        # 最终视图逐项比对
        ledger_placement = self.ledger.placement()
        for key in sorted(set(placement) | set(ledger_placement)):
            if placement.get(key) != ledger_placement.get(key):
                raise ReplayError(
                    "最终归属层分歧", object_key=key,
                    mismatch_field="tier_id", basis="ledger.object_state",
                    expected=ledger_placement.get(key),
                    actual=placement.get(key))
        cost: dict[str, int] = {}
        for state in self.ledger.tier_states():
            if state.used_bytes != occupancy[state.tier_id]:
                raise ReplayError(
                    "最终层占用字节数分歧", object_key=None,
                    mismatch_field="used_bytes",
                    basis=f"ledger.tier_state[{state.tier_id}]",
                    expected=state.used_bytes,
                    actual=occupancy[state.tier_id])
            if state.object_count != counts[state.tier_id]:
                raise ReplayError(
                    "最终层对象数分歧", object_key=None,
                    mismatch_field="object_count",
                    basis=f"ledger.tier_state[{state.tier_id}]",
                    expected=state.object_count,
                    actual=counts[state.tier_id])
            expected_cost = occupancy[state.tier_id] * state.unit_cost
            cost[state.tier_id] = expected_cost
            if state.cost != expected_cost:
                raise ReplayError(
                    "最终层成本分歧", object_key=None,
                    mismatch_field="cost",
                    basis=f"ledger.tier_state[{state.tier_id}]",
                    expected=state.cost, actual=expected_cost)

        batch_count = int(self.ledger.conn.execute(
            "SELECT COUNT(*) FROM batches WHERE status='committed'").fetchone()[0])
        return ReplayResult(
            steps_compared=len(recorded), batches_compared=batch_count,
            final_occupancy=dict(occupancy), final_cost=cost,
            final_placement=dict(placement))
