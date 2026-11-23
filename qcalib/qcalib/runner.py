"""端到端编排：状态指纹 -> 精确失效 -> 管线运行 -> 快照整体替换 -> 报告。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CalibrationConfig, FeatureTable
from .fingerprint import cache_key, content_fingerprint
from .pipeline import Pipeline, PipelineResult
from .snapshot import Snapshot, version_changed
from .store import StageStore

STATE_NAME = "input_state.json"


class CalibrationRunner:
    def __init__(
        self,
        table: FeatureTable,
        config: CalibrationConfig,
        work_dir: str | Path,
        element_budget: Optional[int] = None,
        max_workers: int = 4,
        accumulator_bits: int = 32,
    ) -> None:
        self.table = table
        self.config = config
        self.work_dir = Path(work_dir)
        self.cache_dir = self.work_dir / "cache"
        self.snapshot_dir = self.work_dir / "snapshot"
        self.state_path = self.work_dir / STATE_NAME
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.store = StageStore(self.cache_dir)
        self.element_budget = element_budget
        self.max_workers = max_workers
        self.accumulator_bits = accumulator_bits

    # ------------------------------------------------------------------
    def _column_state(self) -> Dict[str, Dict[str, Any]]:
        state: Dict[str, Dict[str, Any]] = {}
        for cfg in self.config.groups:
            for column in cfg.columns:
                vector = self.table.column_vector(column)
                state[f"{cfg.group_id}|{column}"] = {
                    "group_id": cfg.group_id,
                    "column": column,
                    # 仅内容指纹：单值变化只改变所属列，不牵连其他列
                    "values_fp": content_fingerprint(
                        [repr(v) for v in vector if v is not None]
                    ),
                    "params_fp": cfg.params_fingerprint,
                    "qbits": cfg.qbits,
                    "fbits": cfg.fbits,
                }
        return state

    def _diff_groups(self, current: Dict[str, Dict[str, Any]]) -> List[str]:
        if not self.state_path.exists():
            return []  # 空白起步，无需失效
        previous = json.loads(self.state_path.read_text(encoding="utf-8"))
        changed: set[str] = set()
        keys = set(previous) | set(current)
        for key in keys:
            if previous.get(key) != current.get(key):
                changed.add(key.split("|", 1)[0])
        return sorted(changed)

    # ------------------------------------------------------------------
    def run(self, force_full: bool = False) -> Dict[str, Any]:
        current_state = self._column_state()
        invalidated: List[str] = []
        if force_full:
            invalidated = sorted(self.store.index.keys())
            self.store.index.clear()
            if self.store.blobs_dir.exists():
                for blob in self.store.blobs_dir.glob("*"):
                    blob.unlink()
        else:
            changed_groups = self._diff_groups(current_state)
            removed: List[str] = []
            for group_id in changed_groups:
                removed.extend(self.store.invalidate_group(group_id))
            invalidated = sorted(removed)

        pipeline = Pipeline(
            self.table,
            self.config,
            self.store,
            element_budget=self.element_budget,
            max_workers=self.max_workers,
            accumulator_bits=self.accumulator_bits,
        )
        result = pipeline.run(force_full=False)
        if force_full:
            result.invalidated = invalidated

        # 持久化输入状态
        self.state_path.write_text(
            json.dumps(current_state, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        report = self._build_report(result, invalidated, force_full)
        fingerprint = self._commit_snapshot(result, report)
        report["snapshot_fingerprint"] = fingerprint
        report["snapshot_version_changed"] = version_changed(self.snapshot_dir, fingerprint)
        return report

    # ------------------------------------------------------------------
    def _build_report(
        self,
        result: PipelineResult,
        invalidated: List[str],
        force_full: bool,
    ) -> Dict[str, Any]:
        groups_out: Dict[str, Any] = {}
        for (group_id, column), calib in sorted(result.calibration.items()):
            rescale = result.rescale[(group_id, column)]
            report = result.errors[(group_id, column)]
            groups_out.setdefault(group_id, []).append(
                {
                    "column": column,
                    "calibration": calib.to_dict(),
                    "rescale": {
                        k: v
                        for k, v in rescale.to_dict().items()
                        if k != "elements"
                    },
                    "error": {
                        "max_rel_error": report.max_rel_error,
                        "saturation_ratio": report.saturation_ratio,
                        "outlier_ratio": report.outlier_ratio,
                        "dominant_shares": report.dominant_shares,
                        "layer_conclusion": report.layer_conclusion,
                        "cache_key": report.cache_key,
                    },
                }
            )
        return {
            "mode": "full" if force_full else "incremental",
            "groups": {gid: groups_out[gid] for gid in sorted(groups_out)},
            "reused": result.reused,
            "recomputed": result.recomputed,
            "invalidated": sorted(invalidated),
            "pending": result.pending,
            "failures": result.failures.to_list(),
        }

    def _commit_snapshot(self, result: PipelineResult, report: Dict[str, Any]) -> str:
        with Snapshot(self.snapshot_dir) as snap:
            snap.write_json("report.json", report)
            for (group_id, column), calib in result.calibration.items():
                base = f"groups/{group_id}/{column}"
                snap.write_json(f"{base}/calibration.json", calib.to_dict())
                snap.write_json(
                    f"{base}/rescale.json",
                    result.rescale[(group_id, column)].to_dict(),
                )
                snap.write_json(
                    f"{base}/error.json",
                    result.errors[(group_id, column)].to_dict(),
                )
            content_only = {
                "groups": report["groups"],
                "stages": ["calibration", "rescale", "error"],
            }
            manifest = {
                "stages": ["calibration", "rescale", "error"],
                "group_count": len({g for g, _ in result.calibration}),
                "column_count": len(result.calibration),
                "content_fingerprint": content_fingerprint(content_only),
            }
            return snap.commit(manifest)
