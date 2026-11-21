"""顶层运行器：版本管理、快照恢复、双精度调度与诊断报告。"""
from __future__ import annotations

import signal
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .artifact_store import ArtifactStore
from .config import RebuildConfig
from .engine import WorkbenchEngine
from .errors import FailureRecord, WorkbenchError
from .loader import LoadResult, input_fingerprint
from .precision_analysis import analyze
from .utils import atomic_write_json, content_hash


@dataclass
class RunReport:
    report: Dict[str, Any]
    exit_code: int = 0
    recovery_point: Optional[str] = None


class GracefulShutdown:
    """可处理退出信号：完成当前视图阶段原子替换后退出码 0。"""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.previous: Dict[int, Any] = {}

    def install(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self.previous[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):
                pass

    def _handler(self, signum: int, frame: Any) -> None:
        self.event.set()

    @property
    def requested(self) -> bool:
        return self.event.is_set()


def _map_injection_hits(failures: List[FailureRecord],
                        injection_hits: List[Any]) -> List[Dict[str, Any]]:
    """把注入项编号映射到实际命中的业务码，证明注入生效。"""
    hit_codes: Dict[int, List[str]] = {}
    all_codes: List[str] = []
    for failure in failures:
        all_codes.append(failure.code)
        if failure.view is not None:
            hit_codes.setdefault(failure.view, []).append(failure.code)
    mapped: List[Dict[str, Any]] = []
    for hit in injection_hits:
        codes = hit_codes.get(hit.view, [])
        if codes and hit.expected_code in codes:
            matched = True
        elif hit.expected_code in all_codes:
            # 阶段级故障（如零体素）不绑定单视图，按全局业务码命中
            matched = True
            codes = [c for c in all_codes if c == hit.expected_code]
        else:
            matched = False
        mapped.append(
            {
                **hit.to_dict(),
                "observed_codes": codes,
                "hit": matched,
            }
        )
    return mapped


class ReconstructionRunner:
    def __init__(self, load_result: LoadResult, config: RebuildConfig,
                 work_dir: str, injection_hits: Optional[List[Any]] = None,
                 concurrency: Optional[int] = None) -> None:
        self.load_result = load_result
        self.config = config
        self.store = ArtifactStore(work_dir)
        self.injection_hits = injection_hits or []
        self.concurrency = concurrency
        self.config_hash = config.fingerprint()
        self.input_hash = input_fingerprint(load_result.views)

    def _resolve_version(self) -> tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        """启动续跑判定，不一致触发整体替换并返回恢复点描述。"""
        latest = self.store.latest_version()
        if latest is None:
            return 1, None, None
        try:
            manifests = self.store.verify_resume(
                latest, self.config_hash, self.input_hash
            )
        except WorkbenchError as exc:
            # E_SNAPSHOT_MISMATCH：废弃旧版本整体替换，而非逐文件覆盖
            self.store.discard_version(latest)
            recovery = f"discarded mismatched snapshot v{latest:04d}: {exc.reason}"
            fallback = self.store.latest_version()
            if fallback is not None:
                try:
                    manifests = self.store.verify_resume(
                        fallback, self.config_hash, self.input_hash
                    )
                    return fallback + 1, manifests, recovery
                except WorkbenchError:
                    self.store.discard_version(fallback)
            return 1, None, recovery
        return latest + 1, manifests, None

    def execute(self, shutdown: Optional[GracefulShutdown] = None,
                report_path: Optional[str] = None) -> RunReport:
        version, manifests, recovery_note = self._resolve_version()
        recovery_point = recovery_note
        if manifests:
            done = sorted(manifests)
            recovery_point = (recovery_point or "") + (
                f"|resume from v{version - 1:04d} stages={done}"
            )

        engine = WorkbenchEngine(
            self.load_result, self.config, self.store,
            injection_hits=self.injection_hits,
            concurrency=self.concurrency,
        )
        shutdown_event = shutdown.event if shutdown else None

        results: Dict[str, Dict[str, Any]] = {}
        self.last_results = results
        shutdown_seen = False
        for precision in ("float32", "float64"):
            result = engine.run(
                precision, version, shutdown_event=shutdown_event
            )
            results[precision] = {
                "features_flat": result.features_flat,
                "matches_flat": result.matches_flat,
                "tripoints_flat": result.tripoints_flat,
                "depth_maps": result.depth_maps,
                "cloud": result.cloud,
                "raw": result,
            }
            if shutdown is not None and shutdown.requested:
                shutdown_seen = True
                break

        all_failures = list(self.load_result.failures)
        for precision in results:
            raw = results[precision]["raw"]
            for failure in raw.failures:
                if failure not in all_failures:
                    all_failures.append(failure)

        precision_report = (
            analyze(results, self.config)
            if "float64" in results and "float32" in results
            else {"note": "中断发生，双精度对比未完成"}
        )

        report: Dict[str, Any] = {
            "run_id": content_hash(
                {"config": self.config_hash, "input": self.input_hash,
                 "version": version}
            ),
            "snapshot_version": version,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "precision_tracks": list(results.keys()),
            "views_loaded": [v.view for v in self.load_result.views],
            "failures": [f.to_dict() for f in all_failures],
            "cache_decisions": {"reused": [], "recomputed": []},
            "precision_analysis": precision_report,
            "injection_hits": _map_injection_hits(
                all_failures, self.injection_hits
            ),
            "recovery_point": recovery_point,
            "shutdown_completed": shutdown_seen,
            "counts": self._counts(results),
        }
        # 缓存复用/重算清单（取自 float32 主路径引擎决策）
        if "float32" in results:
            raw = results["float32"]["raw"]
            if engine.last_context is not None:
                report["cache_decisions"] = engine.last_context.decision.to_dict()
            report["budget"] = {
                "deductions": raw.budget_deductions,
                "pending": raw.pending,
            }
        if report_path:
            atomic_write_json(report_path, report)
        exit_code = 0
        return RunReport(
            report=report, exit_code=exit_code,
            recovery_point=recovery_point,
        )

    @staticmethod
    def _counts(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, Any] = {}
        for precision, data in results.items():
            raw = data["raw"]
            counts[precision] = {
                "features": sum(len(v) for v in raw.features.values()),
                "matches": len(raw.matches_flat),
                "tri_points": len(raw.tripoints_flat),
                "depth_maps": len(raw.depth_maps),
                "cloud_points": len(raw.cloud),
            }
        return counts
