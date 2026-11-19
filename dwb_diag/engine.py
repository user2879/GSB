"""诊断编排引擎: 组装存储/恢复/视图/快照, 并生成确定性样本。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    DataLayout,
    DEFAULT_CAPACITY,
    DEFAULT_PAGE_SIZE,
    DEMO_SEED,
    DeterministicClock,
)
from .recovery import RecoveryEngine, RecoveryReport
from .snapshot import Snapshot, SnapshotStore
from .storage import CrashPlan, Storage
from .util import StructuredLogger
from .views import ViewCatalog, ViewDef, ViewManager
from .fault_injection import clone_layout, replay_compare


@dataclass
class InvalidationRecord:
    page_no: int
    sequence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"page_no": self.page_no, "sequence": list(self.sequence)}


class DiagnosticEngine:
    """对外门面: 一个目录对应一次可复现诊断会话。"""

    def __init__(
        self,
        root: Path | str,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        capacity: int = DEFAULT_CAPACITY,
        deterministic: bool = True,
    ) -> None:
        self.layout = DataLayout(Path(root)).ensure()
        self.page_size = page_size
        self.capacity = capacity
        self.clock = DeterministicClock() if deterministic else None
        from .config import Clock

        self.clock = self.clock or Clock()
        self.logger = StructuredLogger(self.layout.log_file, self.clock)
        self.storage = Storage(
            self.layout, page_size, capacity, self.clock, self.logger
        )
        self.catalog = ViewCatalog.load(self.layout.views_file)
        self.views = ViewManager(self.catalog, self.storage.keyspace)
        self.snapshots = SnapshotStore(self.layout.snapshot_db, self.logger)
        self._invalidation: list[InvalidationRecord] = []
        self._recovery_invalidation: list[InvalidationRecord] = []

    # ---------- 写入 ----------
    def submit(
        self, changes: dict[str, int | None], *, crash_plan: CrashPlan | None = None
    ):
        result = self.storage.submit(changes, crash_plan=crash_plan)
        # 正常提交后沿覆盖页推导视图失效
        changed = set(result.page_nos)
        try:
            sequence = self.views.propagate_invalidation(changed)
        except Exception:
            raise
        for page_no in result.page_nos:
            self._invalidation.append(InvalidationRecord(page_no, sequence))
        self.views.rebuild_graph()
        self.views.rebuild_all()
        return result

    # ---------- 恢复 ----------
    def backup_pre_recovery(self) -> Path:
        """恢复前备份崩溃态(主文件/缓冲/版本/标记), 供确定性重放。"""
        import shutil

        backup = self.layout.pre_recovery_dir
        backup.mkdir(parents=True, exist_ok=True)
        for src in [
            self.layout.main_file,
            self.layout.buffer_file,
            self.layout.version_map_file,
            self.layout.markers_file,
            self.layout.views_file,
        ]:
            if src.exists():
                shutil.copy2(src, backup / src.name)
        return backup

    def recover(self, *, raise_on_lost: bool = True) -> RecoveryReport:
        self.backup_pre_recovery()
        engine = RecoveryEngine(
            self.storage.main,
            self.storage.dwb,
            self.storage.versions,
            self.storage.keyspace,
            self.logger,
        )
        report = engine.run(raise_on_lost=raise_on_lost)
        # 修复页 -> 视图失效传播
        # 先按当前已恢复主文件重建依赖图, 再推导失效, 保证传播序列非空且确定
        self.views.rebuild_graph()
        repaired_pages = {e.page_no for e in report.repaired}
        if repaired_pages:
            sequence = self.views.propagate_invalidation(repaired_pages)
            records = [
                InvalidationRecord(page_no, sequence)
                for page_no in sorted(repaired_pages)
            ]
            self._invalidation.extend(records)
            self._recovery_invalidation.extend(records)
        self.views.rebuild_graph()
        self.views.rebuild_all()
        return report

    def verify_views(self) -> list[dict[str, Any]]:
        return self.views.verify()

    # ---------- 快照 ----------
    def save_snapshot(self, session_id: str, report: RecoveryReport) -> Snapshot:
        return self.snapshots.save(
            session_id=session_id,
            created_seq=report.max_seq,
            buffer_digest=self.storage.dwb.file_digest_hex(),
            main=self.storage.main,
            versions=self.storage.versions,
            view_manager=self.views,
            repair_trace=[e.to_dict() for e in report.repaired],
            invalidation=[r.to_dict() for r in self._recovery_invalidation],
            created_ts_ms=self.clock.now_ms(),
        )

    def open_snapshot(self, session_id: str):
        snap = self.snapshots.load(session_id)
        if snap is None:
            raise FileNotFoundError(f"snapshot {session_id} not found")
        return self.snapshots.reconcile(
            snap,
            self.storage.dwb,
            self.storage.main,
            self.storage.versions,
            self.views,
        )

    # ---------- 重放 ----------
    def replay(self, snapshot: Snapshot, scratch_root: Path) -> dict[str, Any]:
        # 从崩溃态备份克隆, 确保重放确实重新执行页修复而非读到已修复主文件
        if self.layout.pre_recovery_dir.exists():
            scratch_root.mkdir(parents=True, exist_ok=True)
            import shutil

            for src in self.layout.pre_recovery_dir.iterdir():
                shutil.copy2(src, scratch_root / src.name)
            scratch = DataLayout(scratch_root)
        else:
            scratch = clone_layout(self.layout, scratch_root)
        scratch_logger = StructuredLogger(scratch.log_file, self.clock)
        scratch_storage = Storage(
            scratch, self.page_size, self.capacity, self.clock, scratch_logger
        )
        scratch_catalog = ViewCatalog.load(scratch.views_file)
        scratch_views = ViewManager(scratch_catalog, scratch_storage.keyspace)
        report = RecoveryEngine(
            scratch_storage.main,
            scratch_storage.dwb,
            scratch_storage.versions,
            scratch_storage.keyspace,
            scratch_logger,
        ).run()
        repaired_pages = {e.page_no for e in report.repaired}
        invalidation: list[dict[str, Any]] = []
        if repaired_pages:
            scratch_views.rebuild_graph()
            sequence = scratch_views.propagate_invalidation(repaired_pages)
            for page_no in sorted(repaired_pages):
                invalidation.append(
                    InvalidationRecord(page_no, sequence).to_dict()
                )
        replay_compare(
            report.final_digests,
            self._snapshot_main_digests(snapshot),
            invalidation,
            snapshot.invalidation,
        )
        return {
            "status": "replay_ok",
            "digests": report.final_digests,
            "invalidation": invalidation,
        }

    @staticmethod
    def _snapshot_main_digests(snapshot: Snapshot) -> dict[str, str]:
        return {
            str(k): v["digest"] for k, v in snapshot.version_map.items()
        }

    def invalidation_trace(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._invalidation]


def build_demo_views(catalog: ViewCatalog) -> None:
    catalog.add(ViewDef("cnt_user", "prefix_count", {"prefix": "user:"}))
    catalog.add(ViewDef("cnt_order", "prefix_count", {"prefix": "order:"}))
    catalog.add(ViewDef("range_low", "value_range", {"lo": 0, "hi": 9}))
    catalog.add(ViewDef("range_high", "value_range", {"lo": 10, "hi": 1_000_000}))
    # 视图依赖视图的聚合链, 用于间接传播
    catalog.add(
        ViewDef("rollup_all", "rollup", {"sources": ["cnt_user", "cnt_order"]})
    )
    catalog.save()


def deterministic_changes(seed: int = DEMO_SEED) -> list[dict[str, int]]:
    """固定种子生成三批确定性键值变更。"""
    rng = random.Random(seed)
    batches: list[dict[str, int]] = []
    for batch in range(3):
        changes: dict[str, int] = {}
        for i in range(8):
            key = f"{'user' if i % 2 == 0 else 'order'}:{batch:02d}:{i:02d}"
            changes[key] = rng.randint(0, 999)
        batches.append(changes)
    return batches
