"""写路径存储引擎: 双写先后顺序、三阶段崩溃点标记与崩溃注入。

每次 submit 的固定顺序(要求 2):
  1. buffer_write    : 全部受影响页镜像追加进 DWB 并 fsync 落定
  2. main_overwrite  : 逐页覆盖主数据文件对应槽位并 fsync
  3. commit_mark     : 更新页版本映射并写入完成标记
三个阶段各自写一条崩溃点标记(写入序号、阶段名、单调时钟毫秒)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .buffer import DoubleWriteBuffer
from .config import PHASES
from .keyspace import Keyspace
from .paging import MainDataFile, Page
from .util import StructuredLogger
from .versions import CrashMarker, MarkerLog, VersionMap


class SimulatedCrash(RuntimeError):
    """按崩溃计划主动抛出的模拟崩溃(控制流, 非业务错误码)。"""

    def __init__(self, phase: str, commit_id: int, seq: int) -> None:
        super().__init__(f"simulated crash at phase={phase} commit={commit_id} seq={seq}")
        self.phase = phase
        self.commit_id = commit_id
        self.seq = seq


@dataclass
class CommitResult:
    commit_id: int
    seqs: list[int]
    page_nos: list[int]
    markers: list[dict] = field(default_factory=list)


@dataclass
class CrashPlan:
    """commit_id 命中时在指定阶段写入标记后、阶段动作完成点模拟崩溃。"""

    commit_id: int
    phase: str  # buffer_write / main_overwrite / commit_mark

    def matches(self, commit_id: int, phase: str) -> bool:
        return self.commit_id == commit_id and self.phase == phase


class Storage:
    def __init__(
        self,
        layout: Path | object,
        page_size: int,
        capacity: int,
        clock: object,
        logger: StructuredLogger,
    ) -> None:
        from .config import DataLayout

        self.layout = layout if isinstance(layout, DataLayout) else DataLayout(Path(layout))
        self.layout.ensure()
        self.page_size = page_size
        self.capacity = capacity
        self.clock = clock
        self.logger = logger
        self.page = Page(page_size)
        self.main = MainDataFile(self.layout.main_file, page_size, capacity)
        self.dwb = DoubleWriteBuffer(self.layout.buffer_file, capacity)
        self.versions = VersionMap.load(self.layout.version_map_file)
        self.markers = MarkerLog.load(self.layout.markers_file)
        self.keyspace = Keyspace(self.main)
        self._next_seq = self._bootstrap_next_seq()
        self._next_commit = self._bootstrap_next_commit()

    def _bootstrap_next_seq(self) -> int:
        if self.markers.markers:
            return max(m.seq for m in self.markers.markers) + 1
        return 1

    def _bootstrap_next_commit(self) -> int:
        if self.markers.markers:
            return max(m.commit_id for m in self.markers.markers) + 1
        return 1

    def _mark(self, seq: int, phase: str, commit_id: int) -> CrashMarker:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase}")
        marker = CrashMarker(
            seq=seq, phase=phase, ts_ms=self.clock.now_ms(), commit_id=commit_id
        )
        self.markers.append(marker)
        self.logger.log(
            "phase_marker",
            detail={"seq": seq, "phase": phase, "commit_id": commit_id, "ts_ms": marker.ts_ms},
        )
        return marker

    def _maybe_crash(self, plan: CrashPlan | None, commit_id: int, phase: str, seq: int) -> None:
        if plan is not None and plan.matches(commit_id, phase):
            self.logger.log(
                "simulated_crash",
                detail={"commit_id": commit_id, "phase": phase, "seq": seq},
            )
            raise SimulatedCrash(phase, commit_id, seq)

    def submit(
        self,
        changes: dict[str, int | None],
        *,
        crash_plan: CrashPlan | None = None,
    ) -> CommitResult:
        """把键值变更编译为页变更并按双写顺序提交。"""
        commit_id = self._next_commit
        # 1) 按键归属页聚合变更
        per_page: dict[int, dict[str, int | None]] = {}
        for key, value in changes.items():
            page_no = self.keyspace.page_for(key)
            per_page.setdefault(page_no, {})[key] = value
        page_nos = sorted(per_page)

        seqs: list[int] = []
        frames: dict[int, bytes] = {}
        markers: list[dict] = []

        # 阶段 1: 先写阶段标记, 再把全部镜像追加进 DWB 并落定
        first_seq = self._next_seq
        self._mark(first_seq, PHASES[0], commit_id)
        markers.append(self.markers.markers[-1].to_dict())
        self._maybe_crash(crash_plan, commit_id, PHASES[0], first_seq)
        for idx, page_no in enumerate(page_nos):
            seq = self._next_seq + idx
            frame, _ = self.keyspace.apply_change(page_no, per_page[page_no])
            self.dwb.append_mirror(page_no, self.page_size, seq, frame)
            seqs.append(seq)
            frames[page_no] = frame
        self._next_seq += len(page_nos)

        # 阶段 2: 主文件覆盖(标记 -> 逐页落盘), 可在该阶段中途崩溃形成半写
        last_seq = seqs[-1] if seqs else first_seq
        self._mark(last_seq, PHASES[1], commit_id)
        markers.append(self.markers.markers[-1].to_dict())
        for page_no in page_nos:
            self.main.write_slot(page_no, frames[page_no])
            if crash_plan is not None and crash_plan.matches(commit_id, PHASES[1]):
                # 覆盖部分页后崩溃: 已写页可能半写, 缓冲中仍有完整镜像
                self._maybe_crash(crash_plan, commit_id, PHASES[1], last_seq)
        self._maybe_crash(crash_plan, commit_id, PHASES[1], last_seq)

        # 阶段 3: 页版本映射落盘 + 完成标记
        self._mark(last_seq, PHASES[2], commit_id)
        markers.append(self.markers.markers[-1].to_dict())
        for page_no in page_nos:
            seq = seqs[page_nos.index(page_no)]
            self.versions.upsert(
                page_no, seq, Page.digest_hex(frames[page_no])
            )
        self.versions.save()
        self._maybe_crash(crash_plan, commit_id, PHASES[2], last_seq)

        self._next_commit += 1
        self.logger.log(
            "commit_complete",
            detail={"commit_id": commit_id, "pages": page_nos, "seqs": seqs},
        )
        return CommitResult(commit_id, seqs, page_nos, markers)
