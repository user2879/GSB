"""按时间轴归并测量结果并生成确定性报告与内容哈希。

所有 JSON 均以 sort_keys、固定缩进、ensure_ascii=False 序列化，且不含时间戳
或环境相关字段，保证同一输入两次运行逐字节一致。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .errors import MergeOrderError
from .meter import gated_lufs, percentile_lra, block_lufs

LOUDNESS_TOLERANCE_LU = 0.1
TP_TOLERANCE_DB = 0.1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2)


def merge_results(results: List, reference: Optional[Dict] = None) -> Dict:
    """按任务键时间序归并：积分块均值拼接后做整段两级门控。"""
    ordered = sorted(results, key=lambda r: r.index)
    for i in range(1, len(ordered)):
        if ordered[i].index != ordered[i - 1].index + 1:
            raise MergeOrderError(
                "归并序列出现索引缺口或乱序",
                location={"at_index": i},
                evidence={"prev": ordered[i - 1].task_key, "next": ordered[i].task_key},
            )

    all_ms: List[float] = []
    block_owner: List[str] = []
    windows: List[Dict[str, Any]] = []
    for r in ordered:
        all_ms.extend(r.block_ms)
        block_owner.extend([r.task_key] * len(r.block_ms))
        windows.append(
            {
                "task_key": r.task_key,
                "index": r.index,
                "start_sample": r.start,
                "end_sample": r.end,
                "short_lufs": _round(r.short_lufs),
                "gated_abs_lufs": _round(r.gated_abs_lufs),
                "gated_rel_lufs": _round(r.gated_rel_lufs),
                "true_peak_db": _round(r.true_peak_db),
                "true_peak_channel": r.true_peak_channel,
                "lra_lu": _round(r.lra),
                "block_count": r.block_count,
                "cache_hits": r.cache_hits,
                "cache_misses": r.cache_misses,
                "recomputes": r.recomputes,
                "uncached": r.uncached,
            }
        )

    abs_lufs, rel_lufs, abs_count, rel_count = gated_lufs(all_ms)
    lra = percentile_lra([block_lufs(m) for m in all_ms])
    tp_result = max(ordered, key=lambda r: r.true_peak_db)

    report: Dict[str, Any] = {
        "schema": "loudbench.report/v1",
        "window_count": len(ordered),
        "whole_program": {
            "abs_gated_lufs": _round(abs_lufs),
            "rel_gated_lufs": _round(rel_lufs),
            "abs_block_count": abs_count,
            "rel_block_count": rel_count,
            "total_block_count": len(all_ms),
            "lra_lu": _round(lra),
            "true_peak_db": _round(tp_result.true_peak_db),
            "true_peak_task": tp_result.task_key,
            "true_peak_channel": tp_result.true_peak_channel,
        },
        "windows": windows,
        "block_owner_order": block_owner,
    }

    if reference is not None:
        report["reference_comparison"] = _compare(report["whole_program"], reference)
    return report


def _round(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return round(float(value), ndigits)


def _compare(whole: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    checks = []

    def add(name: str, got: Optional[float], ref: Optional[float], tol: float) -> None:
        if got is None or ref is None:
            ok = got == ref
            delta = None
        else:
            delta = abs(got - ref)
            ok = delta <= tol
        checks.append({"metric": name, "got": got, "reference": ref, "abs_delta": delta, "tolerance": tol, "pass": ok})

    add("rel_gated_lufs", whole["rel_gated_lufs"], reference.get("rel_gated_lufs"), LOUDNESS_TOLERANCE_LU)
    add("abs_gated_lufs", whole["abs_gated_lufs"], reference.get("abs_gated_lufs"), LOUDNESS_TOLERANCE_LU)
    add("true_peak_db", whole["true_peak_db"], reference.get("true_peak_db"), TP_TOLERANCE_DB)
    add("block_count", whole["total_block_count"], reference.get("block_count"), 0)
    return {"all_pass": all(c["pass"] for c in checks), "checks": checks}


def build_run_report(
    run_id: str,
    config: Dict[str, Any],
    task_manifest: Dict[str, Any],
    run,
    reference: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = merge_results(run.results, reference)
    cp = run.cache_plan
    report = {
        "schema": "loudbench.run/v1",
        "run_id": run_id,
        "config": config,
        "effective_workers": run.effective_workers,
        "requested_workers": run.requested_workers,
        "degradation": run.degradation,
        "errors": [e.to_dict() for e in run.errors],
        "cache_stats": {
            "stores": cp.total_stores,
            "hits": cp.total_hits,
            "recomputes": cp.total_recomputes,
            "uncached": cp.total_uncached,
            "evictions": cp.total_evictions,
            "denied_buffers": len(cp.denied_records),
            "final_resident_count": len(cp.final_resident),
        },
        "ledger": cp.ledger,
        "merged": merged,
    }
    return report


def write_report(run_dir: str, name: str, obj: Any) -> Dict[str, str]:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    text = canonical_json(obj)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "sha256": sha256_bytes(text.encode("utf-8"))}
