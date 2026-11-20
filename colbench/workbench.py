"""工作台高层编排：构建段、计划、调度、账本、重算、版本替换、快照。"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from . import inject, snapshot
from .ledger import Ledger
from .planner import build_plan
from .replay import recompute, verify_ledger
from .scheduler import Scheduler
from .segment import Segment, build_segment, load_segment
from .schema import SegmentSpec, load_segment_spec
from .versioning import BudgetParams, ParameterVersion, ViewState, ViewStore, VersionRegistry
from .view import DashboardModel, build_block_rows, build_segment_quota


class Workbench:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace)
        self.data_dir = self.workspace / "segments"
        self.db_path = self.workspace / "ledger.sqlite3"
        self.snapshot_path = self.workspace / "session.snapshot.json"
        self.version_path = self.workspace / "versions.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry = VersionRegistry(self.version_path)
        self.segments: dict[str, Segment] = {}
        self.view_store = ViewStore()
        self._params: BudgetParams | None = None

    # ------------------------------------------------------------------
    def build_from_specs(self, spec_paths: list[str | Path]) -> dict[str, Segment]:
        specs = [load_segment_spec(path) for path in spec_paths]
        segments = {}
        for spec in specs:
            segments[spec.segment_id] = build_segment(spec, self.data_dir)
        self.segments = segments
        encoding_sig = tuple(
            (key, self.segments[sid].headers[key].encoding,
             self.segments[sid].headers[key].compressed)
            for sid in sorted(self.segments)
            for key in sorted(self.segments[sid].headers))
        self.registry.observe_encoding(encoding_sig)
        self.registry.observe_format(1)
        return segments

    def reopen_segments(self) -> dict[str, Segment]:
        """模拟重新打开：从段文件与块表恢复，不走内存缓存。"""
        recovered = {}
        for path in sorted(self.data_dir.glob("*.cseg")):
            segment_id = path.stem
            spec_path = self.workspace / "samples" / f"segment_{segment_id}.json"
            if not spec_path.exists():
                spec_path = self._find_spec(segment_id)
            recovered[segment_id] = load_segment(spec_path,
                                                 directory=self.data_dir)
        self.segments = recovered
        return recovered

    def _find_spec(self, segment_id: str) -> Path:
        matches = list(self.workspace.rglob(f"segment_{segment_id}.json"))
        if not matches:
            raise FileNotFoundError(f"找不到 {segment_id} 的段定义")
        return matches[0]

    # ------------------------------------------------------------------
    def plan(self, query: str) -> "ScanPlan":
        if len(self.segments) != 1:
            # 多段使用任意段的 schema（样本中两段结构一致），逐段生成候选
            spec = next(iter(self.segments.values())).spec
        else:
            spec = next(iter(self.segments.values())).spec
        plans = [build_plan(query, segment.spec) for segment in
                 self.segments.values()]
        merged = plans[0]
        for extra in plans[1:]:
            merged.candidate_blocks.extend(extra.candidate_blocks)
            merged.skipped_blocks.extend(extra.skipped_blocks)
        merged.candidate_blocks.sort(
            key=lambda item: (item["segment_id"], item["row_group"],
                              item["column"], item["block_index"]))
        return merged

    def run_scan(self, query: str, params: BudgetParams,
                 *, business_codes: list[str] | None = None) -> DashboardModel:
        version = self.registry.observe_budget(params)
        self._params = params
        plan = self.plan(query)
        scheduler = Scheduler(params, self.segments)
        schedule = scheduler.run(plan.candidate_blocks)

        with Ledger(self.db_path) as ledger:
            stale_count = ledger.mark_stale(version)
            ledger.append_batch(schedule, version)
            verify_report = verify_ledger(plan, self.segments, ledger,
                                          version)
            trajectories = [dict(row) for row in ledger.active_rows(version)]

        recomputed = recompute(plan, self.segments)
        result_digests = [
            {"segment_id": block.segment_id, "row_group": block.row_group,
             "column": block.column,
             "block_index": block.block_index,
             "result_digest": block.result_digest}
            for block in recomputed]
        new_state = ViewState(
            version=version, trajectories=trajectories,
            result_digests=result_digests,
            statistics=schedule.statistics())
        replacement = self.view_store.replace(new_state)

        snapshot.save_snapshot(
            self.snapshot_path, self.segments, version,
            [task.to_dict() for task in schedule.tasks], result_digests,
            schedule.statistics())

        model = DashboardModel(
            plan=plan, schedule=schedule, version=version,
            block_rows=build_block_rows(self.segments),
            segment_quota=build_segment_quota(
                schedule, params.per_segment_concurrency),
            business_codes=business_codes or ["OK"],
            replacement=replacement,
            snapshots=[self.snapshot_path.name],
            verify_report=verify_report)
        model.stale_count = stale_count
        return model

    # ------------------------------------------------------------------
    def restore(self, current_params: BudgetParams) -> dict:
        version = self.registry.observe_budget(current_params)
        return snapshot.restore_snapshot(self.snapshot_path, self.segments,
                                         version)

    def inject_fault(self, kind: str, *, segment_id: str,
                     row_group: int, column: str) -> dict:
        segment_path = self.data_dir / f"{segment_id}.cseg"
        if kind == "truncate":
            injection = inject.inject_payload_truncation(
                segment_path, row_group, column)
        elif kind == "encoding_mismatch":
            injection = inject.inject_encoding_mismatch(
                segment_path, row_group, column)
        else:
            raise ValueError(f"未知注入类型 {kind}")
        return injection.to_dict()
