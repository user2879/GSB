"""迁移执行引擎：真实文件搬运、原子切换、批次提交与目录真值校验。

崩溃模型：
- 批次以 open 状态先登记账本，再逐对象执行“临时文件 -> 校验 -> 原子替换 ->
  删源”，全部成功后单事务提交；
- crash_hook 在指定对象搬运完成后、批次提交前强制退出，模拟进程被杀死，
  此时账本无该批次提交记录，磁盘上出现归属未定对象，由 recovery 模块处置。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .applog import StructuredLogger
from .content import DIGEST_ALGO, verify_file, write_object_file
from .errors import CapacityError, IntegrityError
from .ledger import Ledger, TierState
from .manifest import Manifest
from .planner import Plan, PlanStep, derive_plan, split_batches


@dataclass(frozen=True)
class DirScan:
    tier_id: str
    path: str
    files: dict[str, int]  # 文件名(摘要) -> 字节数
    total_bytes: int


class TierEngine:
    def __init__(self, workspace: str | Path, logger: StructuredLogger | None = None,
                 crash_hook: Callable[[str], None] | None = None):
        self.root = Path(workspace)
        self.tiers_dir = self.root / "tiers"
        self.db_path = self.root / "ledger.db"
        self.logger = logger or StructuredLogger(self.root / "logs" / "events.jsonl")
        self.crash_hook = crash_hook
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)

    def tier_dir(self, tier_id: str) -> Path:
        return self.tiers_dir / tier_id

    def object_path(self, tier_id: str, digest: str) -> Path:
        return self.tier_dir(tier_id) / f"{digest}.bin"

    def prepare_dirs(self, manifest: Manifest) -> None:
        for tier in manifest.tiers:
            self.tier_dir(tier.tier_id).mkdir(parents=True, exist_ok=True)

    def seed_objects(self, manifest: Manifest) -> dict[str, str]:
        """按种子确定性生成全部对象文件到热层（已存在则跳过并校验）。"""
        hot_id = manifest.hot_tier.tier_id
        result: dict[str, str] = {}
        for obj in manifest.objects:
            path = self.object_path(hot_id, obj.content_digest)
            if path.exists():
                verify_file(path, obj.content_digest,
                            field=f"objects.{obj.key}.content_digest")
            else:
                digest = write_object_file(
                    path, manifest.seed, obj.key, obj.size_bytes,
                    manifest.backend)
                if digest != obj.content_digest:  # pragma: no cover - 防御
                    raise IntegrityError(
                        f"生成内容摘要与清单不一致: {obj.key}",
                        field=f"objects.{obj.key}.content_digest",
                        details={"expected": obj.content_digest, "actual": digest})
            result[obj.key] = str(path)
        self.logger.info("objects_seeded",
                         count=len(manifest.objects), tier=hot_id)
        return result

    # ---------- 目录扫描真值 ----------
    def scan_tier(self, tier_id: str) -> DirScan:
        directory = self.tier_dir(tier_id)
        files: dict[str, int] = {}
        if directory.exists():
            for entry in sorted(directory.iterdir()):
                if entry.is_file() and not entry.name.endswith(".tmp"):
                    files[entry.name] = entry.stat().st_size
        return DirScan(tier_id=tier_id, path=str(directory),
                       files=files, total_bytes=sum(files.values()))

    def scan_all(self, manifest: Manifest) -> dict[str, DirScan]:
        return {tier.tier_id: self.scan_tier(tier.tier_id)
                for tier in manifest.tiers}

    def reconcile_bytes(self, manifest: Manifest,
                        ledger: Ledger) -> None:
        """以目录扫描为字节数真值，比对账本层占用。"""
        scans = self.scan_all(manifest)
        for state in ledger.tier_states():
            scan = scans.get(state.tier_id)
            if scan is None:
                raise IntegrityError(
                    f"账本引用的层目录缺失: {state.tier_id}",
                    field="tier_id", details={"tier_id": state.tier_id})
            if scan.total_bytes != state.used_bytes:
                raise IntegrityError(
                    f"层 {state.tier_id} 目录字节数 {scan.total_bytes} 与账本 "
                    f"{state.used_bytes} 不一致",
                    field="used_bytes",
                    details={"tier_id": state.tier_id,
                             "scan_bytes": scan.total_bytes,
                             "ledger_bytes": state.used_bytes})
            if len(scan.files) != state.object_count:
                raise IntegrityError(
                    f"层 {state.tier_id} 文件数 {len(scan.files)} 与账本对象数 "
                    f"{state.object_count} 不一致",
                    field="object_count",
                    details={"tier_id": state.tier_id,
                             "scan_files": len(scan.files),
                             "ledger_objects": state.object_count})

    # ---------- 原子搬运 ----------
    def move_object(self, manifest: Manifest, step: PlanStep) -> int:
        digest = next(o.content_digest for o in manifest.objects
                      if o.key == step.key)
        src = self.object_path(step.source_tier, digest)
        dst_dir = self.tier_dir(step.target_tier)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = self.object_path(step.target_tier, digest)
        tmp = dst_dir / f".{digest}.{os.getpid()}.tmp"
        if not src.is_file():
            raise IntegrityError(
                f"迁移源文件缺失: {step.key}", field="source_file",
                details={"key": step.key, "path": str(src)})
        try:
            with src.open("rb") as inp, tmp.open("wb") as out:
                while True:
                    chunk = inp.read(64 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            verify_file(tmp, digest, field=f"objects.{step.key}.content_digest")
            os.replace(tmp, dst)
            if src.resolve() != dst.resolve():
                src.unlink()
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        if src.exists():
            raise IntegrityError(
                f"原子切换后源文件仍存在: {step.key}",
                field="source_file", details={"key": step.key})
        self.logger.info("object_moved", key=step.key,
                         source=step.source_tier, target=step.target_tier,
                         digest=digest, algo=DIGEST_ALGO)
        target_tier = manifest.tier(step.target_tier)
        return step.size_bytes * target_tier.rw_weight

    # ---------- 计划与批次执行 ----------
    def plan(self, ledger: Ledger, manifest: Manifest) -> Plan:
        placement = ledger.placement()
        plan = derive_plan(manifest, placement)
        plan_id = ledger.save_plan(plan, int(time.time() * 1000))
        self.logger.info("plan_derived", plan_id=plan_id,
                         steps=len(plan.steps), digest=plan.plan_digest,
                         evict=list(plan.evict_keys),
                         promote=list(plan.promote_keys),
                         retained=list(plan.retained_keys))
        return plan

    def execute(self, ledger: Ledger, manifest: Manifest,
                plan: Plan | None = None,
                crash_after_key: str | None = None) -> dict:
        if plan is None:
            pending = ledger.pending_plan()
            if pending is None:
                plan = self.plan(ledger, manifest)
            else:
                plan = derive_plan(manifest, ledger.placement())
        self._assert_capacity(ledger, manifest, plan)
        batches = split_batches(plan, manifest.rules.batch_limit)
        plan_row = ledger.pending_plan()
        plan_id = int(plan_row["plan_id"]) if plan_row else ledger.save_plan(
            plan, int(time.time() * 1000))
        progress = ledger.progress()
        start_index = int(progress["next_batch_index"])
        moved_total = 0
        executed_batches: list[int] = []
        for batch_index in range(start_index, len(batches)):
            steps = batches[batch_index]
            batch_id = ledger.open_batch(
                plan_id, manifest.spec_version, batch_index,
                int(time.time() * 1000))
            migrations: list[dict] = []
            for step in steps:
                move_cost = self.move_object(manifest, step)
                obj = next(o for o in manifest.objects if o.key == step.key)
                migrations.append({
                    "plan_id": plan_id, "spec_version": manifest.spec_version,
                    "seq": step.seq, "key": step.key,
                    "direction": step.direction,
                    "source_tier": step.source_tier,
                    "target_tier": step.target_tier,
                    "size_bytes": step.size_bytes,
                    "content_digest": obj.content_digest,
                    "move_cost": move_cost,
                })
                if self.crash_hook is not None:
                    self.crash_hook(step.key)
                if crash_after_key is not None and step.key == crash_after_key:
                    self.logger.error("crash_injected", code="E_CRASH",
                                      key=step.key, batch_id=batch_id,
                                      reason="forced_termination_after_move")
                    raise SystemExit(
                        f"模拟强制终止：对象 {step.key} 已搬运但批次未提交")
            states = self._tier_states_after(ledger, manifest, migrations)
            cursor = {"next_batch_index": batch_index + 1,
                      "last_committed_seq": steps[-1].seq}
            ledger.commit_batch(batch_id, migrations, states, cursor,
                                int(time.time() * 1000))
            executed_batches.append(batch_id)
            moved_total += len(migrations)
            self.reconcile_bytes(manifest, ledger)
        # 空计划或全部批次执行完毕，计划收尾；中途崩溃不会走到这里
        if not batches or len(executed_batches) == len(batches):
            ledger.mark_plan_done(plan_id, int(time.time() * 1000))
        conclusion = {
            "spec_version": manifest.spec_version,
            "plan_digest": plan.plan_digest,
            "batches": len(batches),
            "executed_batches": executed_batches,
            "migrations_committed": moved_total,
            "evict_keys": list(plan.evict_keys),
            "promote_keys": list(plan.promote_keys),
            "retained_keys": list(plan.retained_keys),
            "occupancy": plan.simulated_occupancy,
        }
        ledger.save_conclusion(conclusion, int(time.time() * 1000))
        self.logger.info("session_conclusion", **conclusion)
        return conclusion

    def _assert_capacity(self, ledger: Ledger, manifest: Manifest,
                         plan: Plan) -> None:
        states = {s.tier_id: s for s in ledger.tier_states()}
        occupancy = {tid: s.used_bytes for tid, s in states.items()}
        for step in plan.steps:
            occupancy[step.source_tier] -= step.size_bytes
            occupancy[step.target_tier] += step.size_bytes
            tier = manifest.tier(step.target_tier)
            if occupancy[step.target_tier] > tier.capacity_bytes:
                raise CapacityError(
                    f"执行前容量检查失败：层 {step.target_tier} 无法接纳 {step.key}",
                    tier_id=step.target_tier, rejected_keys=[step.key],
                    free_bytes=tier.capacity_bytes
                    - occupancy[step.target_tier] + step.size_bytes,
                    required_bytes=step.size_bytes,
                    field="capacity_bytes")

    def _tier_states_after(self, ledger: Ledger, manifest: Manifest,
                           migrations: list[dict]) -> list[TierState]:
        states = {s.tier_id: s for s in ledger.tier_states()}
        counts = {tid: s.object_count for tid, s in states.items()}
        used = {tid: s.used_bytes for tid, s in states.items()}
        for mig in migrations:
            used[mig["source_tier"]] -= mig["size_bytes"]
            used[mig["target_tier"]] += mig["size_bytes"]
            counts[mig["source_tier"]] -= 1
            counts[mig["target_tier"]] += 1
        result: list[TierState] = []
        for tier in manifest.tiers:
            if used[tier.tier_id] < 0 or counts[tier.tier_id] < 0:
                raise IntegrityError(
                    f"层 {tier.tier_id} 占用在批次后为负",
                    field="used_bytes", details={"used": used[tier.tier_id]})
            result.append(TierState(
                tier_id=tier.tier_id, role=tier.role,
                capacity_bytes=tier.capacity_bytes, used_bytes=used[tier.tier_id],
                free_bytes=tier.capacity_bytes - used[tier.tier_id],
                object_count=counts[tier.tier_id], unit_cost=tier.unit_cost,
                rw_weight=tier.rw_weight,
                eviction_priority=tier.eviction_priority,
                cost=used[tier.tier_id] * tier.unit_cost))
        return result
