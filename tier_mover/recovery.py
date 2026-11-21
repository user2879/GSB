"""强制终止后的恢复。

恢复只承认账本中已完整提交的迁移记录；对磁盘上层目录与账本归属不一致、
或同时出现在两层的对象标注为 unknown，并提供两种处置：
- rollback：按内容摘要回退到源层（删除目标副本，重建源副本）
- rebuild：按目标层重建归属（删除源副本，账本归属更新为目标层）
本模块不生成、也不声称能生成强制终止时刻的“退出报告”。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .applog import StructuredLogger
from .content import verify_file
from .engine import TierEngine
from .errors import IntegrityError
from .ledger import Ledger
from .manifest import Manifest


@dataclass
class RecoveryReport:
    committed_objects: int = 0
    unknown_objects: int = 0
    rolled_back_objects: int = 0
    rebuilt_objects: int = 0
    unknown_keys: list[str] = field(default_factory=list)
    resolutions: list[dict] = field(default_factory=list)
    open_batches: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "committed_objects": self.committed_objects,
            "unknown_objects": self.unknown_objects,
            "rolled_back_objects": self.rolled_back_objects,
            "rebuilt_objects": self.rebuilt_objects,
            "unknown_keys": list(self.unknown_keys),
            "resolutions": list(self.resolutions),
            "aborted_open_batches": list(self.open_batches),
        }


class RecoveryService:
    def __init__(self, engine: TierEngine, ledger: Ledger,
                 manifest: Manifest, logger: StructuredLogger):
        self.engine = engine
        self.ledger = ledger
        self.manifest = manifest
        self.logger = logger

    def _physical_locations(self) -> dict[str, list[str]]:
        """摘要 -> 实际存在该对象文件的层列表。"""
        locations: dict[str, list[str]] = {}
        for scan in self.engine.scan_all(self.manifest).values():
            for filename in scan.files:
                digest = filename[:-4] if filename.endswith(".bin") else filename
                locations.setdefault(digest, []).append(scan.tier_id)
        return locations

    def inspect(self) -> RecoveryReport:
        """只依据已提交账本核对磁盘，产出未知清单，不做修改。"""
        report = RecoveryReport()
        locations = self._physical_locations()
        rows = self.ledger.conn.execute(
            "SELECT key, tier_id, content_digest, size_bytes FROM object_state "
            "ORDER BY key").fetchall()
        open_batch_rows = self.ledger.open_batches()
        report.open_batches = [int(r["batch_id"]) for r in open_batch_rows]
        for row in rows:
            tiers_with_file = locations.get(row["content_digest"], [])
            ledger_tier = row["tier_id"]
            if tiers_with_file == [ledger_tier]:
                report.committed_objects += 1
                continue
            report.unknown_objects += 1
            report.unknown_keys.append(row["key"])
            self.logger.warn("unknown_object_detected", key=row["key"],
                             ledger_tier=ledger_tier,
                             physical_tiers=tiers_with_file,
                             digest=row["content_digest"])
        # 账本没有登记的游离文件也算未知（按摘要无法对应对象时直接报错）
        known_digests = {r["content_digest"] for r in rows}
        for digest, tier_ids in locations.items():
            if digest not in known_digests:
                raise IntegrityError(
                    "发现账本未登记的游离对象文件",
                    field="content_digest",
                    details={"digest": digest, "tiers": tier_ids})
        return report

    def recover(self, mode: str = "rollback") -> RecoveryReport:
        if mode not in ("rollback", "rebuild"):
            raise ValueError("mode 必须为 rollback 或 rebuild")
        report = self.inspect()
        ts = int(time.time() * 1000)
        obj_by_digest = {o.content_digest: o for o in self.manifest.objects}
        with self.ledger.transaction():
            for batch_id in report.open_batches:
                self.ledger.conn.execute(
                    "UPDATE batches SET status='aborted', committed_ms=? "
                    "WHERE batch_id=? AND status='open'", (ts, batch_id))
            rows = self.ledger.conn.execute(
                "SELECT key, tier_id, content_digest FROM object_state "
                "ORDER BY key").fetchall()
            locations = self._physical_locations()
            for row in rows:
                digest = row["content_digest"]
                tiers_with_file = locations.get(digest, [])
                ledger_tier = row["tier_id"]
                if tiers_with_file == [ledger_tier]:
                    continue
                obj = obj_by_digest[digest]
                resolution = self._resolve(row["key"], digest, ledger_tier,
                                           tiers_with_file, mode, ts)
                report.resolutions.append(resolution)
                if mode == "rollback":
                    report.rolled_back_objects += 1
                else:
                    report.rebuilt_objects += 1
            self._rebuild_tier_state(ts)
            # 恢复不是已提交迁移批次：仅写审计表，不改动批次摘要链，
            # 保证 verify_chain 仍能仅依据 committed 批次重算一致
            self.ledger.audit(ts, "recovery_completed",
                              details=report.to_dict())
        self.engine.reconcile_bytes(self.manifest, self.ledger)
        self.logger.info("recovery_completed", **report.to_dict(), mode=mode)
        return report

    def _resolve(self, key: str, digest: str, ledger_tier: str,
                 physical_tiers: list[str], mode: str, ts: int) -> dict:
        tier_ids = [t.tier_id for t in self.manifest.tiers]
        other = next(t for t in tier_ids if t != ledger_tier)
        ledger_path = self.engine.object_path(ledger_tier, digest)
        other_path = self.engine.object_path(other, digest)
        if mode == "rollback":
            # 归属回到账本记录的源层
            if other_path.exists():
                if not ledger_path.exists():
                    self._copy_verified(digest, other_path, ledger_path)
                other_path.unlink()
            elif not ledger_path.exists():
                raise IntegrityError(
                    f"对象 {key} 两层均无文件，无法回退",
                    field="content_digest", details={"key": key})
            verify_file(ledger_path, digest)
            target_tier = ledger_tier
        else:
            # 按目标层重建归属
            if ledger_path.exists() and other_path.exists():
                ledger_path.unlink()
            elif ledger_path.exists() and not other_path.exists():
                self._copy_verified(digest, ledger_path, other_path)
                ledger_path.unlink()
            elif not other_path.exists():
                raise IntegrityError(
                    f"对象 {key} 目标层无文件，无法重建归属",
                    field="content_digest", details={"key": key})
            verify_file(other_path, digest)
            target_tier = other
            self.ledger.conn.execute(
                "UPDATE object_state SET tier_id=?, last_event_ms=? WHERE key=?",
                (target_tier, ts, key))
        return {"key": key, "mode": mode, "resolved_tier": target_tier,
                "physical_tiers_before": list(physical_tiers)}

    @staticmethod
    def _copy_verified(digest: str, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".recover.tmp")
        with src.open("rb") as inp, tmp.open("wb") as out:
            while True:
                chunk = inp.read(64 * 1024)
                if not chunk:
                    break
                out.write(chunk)
            out.flush()
        verify_file(tmp, digest)
        import os
        os.replace(tmp, dst)

    def _rebuild_tier_state(self, ts: int) -> None:
        """以目录扫描真值重建当前版本各层占用、对象数与成本。"""
        version = self.ledger.active_spec_version()
        scans = self.engine.scan_all(self.manifest)
        for tier in self.manifest.tiers:
            scan = scans[tier.tier_id]
            self.ledger.conn.execute(
                "UPDATE tier_state SET used_bytes=?, object_count=?, cost=? "
                "WHERE spec_version=? AND tier_id=?",
                (scan.total_bytes, len(scan.files),
                 scan.total_bytes * tier.unit_cost, version, tier.tier_id))
