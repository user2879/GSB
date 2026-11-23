"""预算约束调度、线程池并发、增量运行与全量重建的统一管线。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .attribution import GroupErrorReport, analyze_column
from .calibration import ColumnCalib, calibrate_column
from .config import CalibrationConfig, FeatureTable, GroupConfig
from .errors import E_CALIB_BUDGET_PENDING, Failure, FailureCollector
from .fingerprint import cache_key
from .kernel import ColumnRescale, rescale_column
from .store import CacheRecord, StageStore


@dataclass(frozen=True)
class Task:
    sort_key: Tuple[str, str]
    group_id: str
    column: str
    cost: int

    def dumps(self) -> Dict[str, Any]:
        return {
            "sort_key": list(self.sort_key),
            "group_id": self.group_id,
            "column": self.column,
            "cost": self.cost,
        }


@dataclass
class PipelineResult:
    calibration: Dict[Tuple[str, str], ColumnCalib]
    rescale: Dict[Tuple[str, str], ColumnRescale]
    errors: Dict[Tuple[str, str], GroupErrorReport]
    reused: List[str] = field(default_factory=list)
    recomputed: List[str] = field(default_factory=list)
    invalidated: List[str] = field(default_factory=list)
    pending: List[Dict[str, Any]] = field(default_factory=list)
    failures: FailureCollector = field(default_factory=lambda: FailureCollector(source="pipeline"))

    def group_calibration(self, group_id: str) -> List[ColumnCalib]:
        return [v for (g, _), v in sorted(self.calibration.items()) if g == group_id]


class Pipeline:
    """串行/并发结果一致的标定管线，支持精确增量复用。"""

    def __init__(
        self,
        table: FeatureTable,
        config: CalibrationConfig,
        store: StageStore,
        element_budget: Optional[int] = None,
        max_workers: int = 4,
        accumulator_bits: int = 32,
    ) -> None:
        self.table = table
        self.config = config
        self.store = store
        self.element_budget = element_budget
        self.max_workers = max_workers
        self.accumulator_bits = accumulator_bits
        self._row_to_groups = self._build_row_group_map()

    # ------------------------------------------------------------------
    def _build_row_group_map(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {rid: [] for rid in self.table.row_ids}
        for cfg in self.config.groups:
            for _ in cfg.columns:
                for rid in self.table.row_ids:
                    mapping[rid].append(cfg.group_id)
        # 去重并稳定排序
        return {rid: sorted(set(gids)) for rid, gids in mapping.items()}

    def _all_tasks(self) -> List[Task]:
        tasks: List[Task] = []
        for cfg in self.config.groups:  # 已按 partition 稳定排序
            for column in cfg.columns:
                vector = self.table.column_vector(column)
                cost = sum(1 for v in vector if v is not None)
                tasks.append(Task((cfg.group_id, column), cfg.group_id, column, cost))
        tasks.sort(key=lambda t: t.sort_key)
        return tasks

    # ------------------------------------------------------------------
    def invalidate(self, changed_groups: Sequence[str]) -> List[str]:
        """使指定列组的全部阶段失效，返回稳定有序的失效键清单。"""
        removed: List[str] = []
        for group_id in sorted(changed_groups):
            removed.extend(self.store.invalidate_group(group_id))
        return sorted(removed)

    def run(self, force_full: bool = False) -> PipelineResult:
        result = PipelineResult({}, {}, {})
        if force_full:
            result.invalidated = sorted(self.store.index.keys())
            self.store.index.clear()
            if self.store.blobs_dir.exists():
                for blob in self.store.blobs_dir.glob("*"):
                    blob.unlink()

        tasks = self._all_tasks()
        # 预算消耗（按稳定顺序）
        budget = self.element_budget
        runnable: List[Task] = []
        pending: List[Task] = []
        for task in tasks:
            if budget is None:
                runnable.append(task)
                continue
            if budget >= task.cost:
                budget -= task.cost
                runnable.append(task)
            else:
                pending.append(task)
        for task in pending:
            result.failures.add(
                E_CALIB_BUDGET_PENDING,
                f"预算不足，任务挂起: {task.group_id}/{task.column} 需 {task.cost}",
                location=f"{task.group_id}:{task.column}",
                remaining_budget=budget,
                required=task.cost,
            )
        result.pending = [t.dumps() for t in pending]

        # 先识别可复用键（在计算前快照，保证复用清单稳定）
        planned_keys: Set[str] = set()
        for task in runnable:
            cfg = self.config.group_by_id[task.group_id]
            vector = self.table.column_vector(task.column)
            used_rows = [rid for rid, v in zip(self.table.row_ids, vector) if v is not None]
            calib = self._preview_calib(cfg, task.column, vector, used_rows)
            planned_keys.add(calib.cache_key)

        for key in sorted(planned_keys):
            if self.store.has(key):
                result.reused.append(key)

        # 并发执行（每任务携带稳定排序键）
        worker = self._make_worker(result)
        if self.max_workers and self.max_workers > 1 and runnable:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                indexed = list(pool.map(worker, runnable))
        else:
            indexed = [worker(task) for task in runnable]

        # 按排序键回填，保证与串行一致
        indexed.sort(key=lambda item: item[0])
        for _, payload in indexed:
            if payload is None:
                continue
            calib, rescale, report, failures, reused_here = payload
            key = (calib.group_id, calib.column)
            result.calibration[key] = calib
            result.rescale[key] = rescale
            result.errors[key] = report
            result.failures.extend(failures)
            if not reused_here and calib.cache_key not in result.recomputed:
                result.recomputed.append(calib.cache_key)

        result.reused = sorted(set(result.reused))
        result.recomputed = sorted(set(result.recomputed))
        result.invalidated = sorted(result.invalidated)
        self.store.save_index()
        return result

    # ------------------------------------------------------------------
    def _preview_calib(
        self,
        cfg: GroupConfig,
        column: str,
        vector: List[Optional[float]],
        used_rows: List[str],
    ) -> ColumnCalib:
        return calibrate_column(cfg, column, vector, self.table.row_ids, used_rows)

    def _make_worker(self, result: PipelineResult) -> Callable[[Task], Any]:
        def worker(task: Task) -> Tuple[Tuple[str, str], Any]:
            cfg = self.config.group_by_id[task.group_id]
            vector = self.table.column_vector(task.column)
            used_rows = [rid for rid, v in zip(self.table.row_ids, vector) if v is not None]

            # 1) 标定（带缓存）
            calib = calibrate_column(cfg, task.column, vector, self.table.row_ids, used_rows)
            cached_calib = self.store.get(calib.cache_key)
            reused_here = False
            if cached_calib is not None:
                reused_here = True
                calib_payload = cached_calib
            else:
                calib_payload = calib.to_dict()
                self.store.put(
                    CacheRecord(
                        cache_key=calib.cache_key,
                        stage="calibration",
                        group_id=cfg.group_id,
                        column=task.column,
                        row_refs=used_rows,
                        param_refs={"params_fingerprint": cfg.params_fingerprint},
                        blob_path=f"blobs/{calib.cache_key}.json",
                    ),
                    calib_payload,
                )

            # 2) 重缩放 + 饱和截断
            rescale, kernel_failures = rescale_column(
                cfg, calib, vector, self.table.row_ids, self.accumulator_bits
            )
            cached_rescale = self.store.get(rescale.cache_key)
            if cached_rescale is None:
                self.store.put(
                    CacheRecord(
                        cache_key=rescale.cache_key,
                        stage="rescale",
                        group_id=cfg.group_id,
                        column=task.column,
                        row_refs=used_rows,
                        param_refs={
                            "params_fingerprint": cfg.params_fingerprint,
                            "fbits": cfg.fbits,
                            "qbits": cfg.qbits,
                        },
                        blob_path=f"blobs/{rescale.cache_key}.json",
                    ),
                    rescale.to_dict(),
                )

            # 3) 误差归因
            report = analyze_column(cfg, calib, rescale)
            if self.store.get(report.cache_key) is None:
                self.store.put(
                    CacheRecord(
                        cache_key=report.cache_key,
                        stage="error",
                        group_id=cfg.group_id,
                        column=task.column,
                        row_refs=used_rows,
                        param_refs={"params_fingerprint": cfg.params_fingerprint},
                        blob_path=f"blobs/{report.cache_key}.json",
                    ),
                    report.to_dict(),
                )
            return task.sort_key, (calib, rescale, report, kernel_failures, reused_here)

        return worker
