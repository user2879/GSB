from __future__ import annotations

import copy
import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vibration_pipeline.data import load_yaml
from vibration_pipeline.engine import run_pipeline
from vibration_pipeline.util import atomic_write_json, canonical_json


def _business_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("waveform_fingerprint", "segments", "boundary_segments", "failures", "injection_hits")
    return all(left[key] == right[key] for key in keys) and left["result_fingerprint"] == right["result_fingerprint"]


def _action_counts(report: dict[str, Any], key: str) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{item['layer']}:{item['algorithm'] or '-'}" for item in report[key]
            ).items()
        )
    )


def _provenance_sorted(report: dict[str, Any]) -> bool:
    for key in ("reuse", "recompute"):
        expected = sorted(
            report[key],
            key=lambda item: (item["sort_key"], item["layer"], item["algorithm"] or "", item["action"]),
        )
        if expected != report[key]:
            return False
    return True


def _step_record(name: str, incremental: dict[str, Any], full: dict[str, Any], stale_expected: bool) -> dict[str, Any]:
    equivalent = _business_equal(incremental, full)
    stale_ok = (incremental["snapshot"]["stale_code"] == "E_SNAPSHOT_STALE") if stale_expected else True
    record = {
        "step": name,
        "business_fields_equal": equivalent,
        "result_fingerprint": incremental["result_fingerprint"],
        "full_result_fingerprint": full["result_fingerprint"],
        "segments": incremental["segment_count"],
        "failures": incremental["failure_count"],
        "boundaries": len(incremental["boundary_segments"]),
        "reuse_count": incremental["reuse_count"],
        "recompute_count": incremental["recompute_count"],
        "reused": _action_counts(incremental, "reuse"),
        "recomputed": _action_counts(incremental, "recompute"),
        "snapshot_stale_code": incremental["snapshot"]["stale_code"],
        "snapshot_stale_expected": stale_expected,
        "snapshot_stale_ok": stale_ok,
        "provenance_order_stable": _provenance_sorted(incremental),
        "incremental_stage_timings_seconds": incremental["stage_timings_seconds"],
        "full_stage_timings_seconds": full["stage_timings_seconds"],
    }
    if not equivalent or not stale_ok or not record["provenance_order_stable"]:
        raise AssertionError(f"verification step failed: {name}")
    return record


def main() -> int:
    config = load_yaml(ROOT / "config" / "pipeline.yaml")
    data = load_yaml(ROOT / "data" / "devices.yaml")
    injections = load_yaml(ROOT / "config" / "injections.yaml")
    artifacts = ROOT / "artifacts"
    run_root = artifacts / "verification"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    append_edits = {
        "segment_edits": {"pump-a": {"append": [{"count": 1, "seed_delta": 991, "source": "verify-append"}]}}
    }
    replace_edits = copy.deepcopy(append_edits)
    replace_edits["segment_edits"]["motor-b"] = {
        "replace": [{"segment_index": 3, "seed_delta": 443, "source": "verify-replace"}]
    }
    cumulative_edits = copy.deepcopy(replace_edits)
    cumulative_edits["segment_edits"]["fan-c"] = {"delete": [1]}

    parameter_config = copy.deepcopy(config)
    parameter_config["algorithms"]["robust_mahalanobis"]["candidate_index"] = 1
    band_config = copy.deepcopy(parameter_config)
    band_config["features"]["band_edges_hz"] = [0.0, 75.0, 150.0, 300.0, 500.0]
    quant_config = copy.deepcopy(band_config)
    quant_config["quantization"]["decimal_places"] = 2
    quant_config["quantization"]["quantum"] = 2
    threshold_config = copy.deepcopy(quant_config)
    threshold_config["threshold"] = 0.625

    steps = [
        ("append", config, append_edits, False),
        ("replace", config, replace_edits, False),
        ("delete", config, cumulative_edits, False),
        ("algorithm_parameter", parameter_config, cumulative_edits, True),
        ("band_configuration", band_config, cumulative_edits, True),
        ("quantization", quant_config, cumulative_edits, True),
        ("threshold", threshold_config, cumulative_edits, True),
    ]

    incremental_snapshot = run_root / "incremental-snapshot"
    initial = run_pipeline(config, data, None, snapshot_dir=incremental_snapshot, mode="incremental")
    baseline_full = run_pipeline(config, data, None, snapshot_dir=run_root / "full-initial", mode="full")
    initial_record = _step_record("initial", initial, baseline_full, False)
    step_records: list[dict[str, Any]] = []
    incremental = initial
    for name, step_config, edits, stale_expected in steps:
        incremental = run_pipeline(
            step_config,
            data,
            None,
            snapshot_dir=incremental_snapshot,
            mode="incremental",
            edits_spec=edits,
        )
        full = run_pipeline(
            step_config,
            data,
            None,
            snapshot_dir=run_root / f"full-{name}",
            mode="full",
            edits_spec=edits,
        )
        step_records.append(_step_record(name, incremental, full, stale_expected))

    parameter_record = next(item for item in step_records if item["step"] == "algorithm_parameter")
    band_record = next(item for item in step_records if item["step"] == "band_configuration")
    quant_record = next(item for item in step_records if item["step"] == "quantization")
    threshold_record = next(item for item in step_records if item["step"] == "threshold")
    cross_algorithm_checks = {
        "parameter_change_only_robust_mahalanobis_raw": all(
            key.startswith("algorithm.raw:") and key.endswith("robust_mahalanobis")
            for key in parameter_record["recomputed"]
            if key.startswith("algorithm.raw:")
        ),
        "band_change_reuses_crest_raw": "algorithm.raw:crest_factor" in band_record["reused"],
        "band_change_recomputes_robust_and_spectral_raw": all(
            key in band_record["recomputed"]
            for key in ("algorithm.raw:robust_mahalanobis", "algorithm.raw:spectral_residual")
        ),
        "quantization_change_reuses_all_raw": not any(key.startswith("algorithm.raw:") for key in quant_record["recomputed"]),
        "threshold_change_reuses_raw_and_quantized": not any(
            key.startswith("algorithm.raw:") or key.startswith("algorithm.quantized:")
            for key in threshold_record["recomputed"]
        ),
    }
    if not all(cross_algorithm_checks.values()):
        raise AssertionError(f"cross-algorithm invalidation checks failed: {cross_algorithm_checks}")

    injection_report = run_pipeline(
        config,
        data,
        injections,
        snapshot_dir=run_root / "injection-snapshot",
        mode="full",
        report_path=artifacts / "report.json",
    )
    injection_hits = {item["id"]: item for item in injection_report["injection_hits"]}
    expected_injection_ids = {item["id"] for item in injections["injections"]}
    injection_checks = {
        "all_configured_injections_hit": set(injection_hits) == expected_injection_ids
        and all(item["hit"] for item in injection_hits.values()),
        "nonfinite_failure_recorded": any(
            item["code"] == "E_NONFINITE_SAMPLE" and item["injected_id"] == "inj-01-nonfinite"
            for item in injection_report["failures"]
        ),
        "band_failure_recorded": any(
            item["code"] == "E_BAND_INVALID" and item["injected_id"] == "inj-02-band-oob"
            for item in injection_report["failures"]
        ),
        "zero_window_failure_recorded": any(
            item["code"] == "E_ALGO_PARAM_INVALID" and item["injected_id"] == "inj-03-crest-window-zero"
            for item in injection_report["failures"]
        ),
        "threshold_boundary_recorded": any(
            item["device_id"] == "motor-b"
            and item["segment_index"] == 3
            and item["score_ticks"] == item["threshold_ticks"]
            and "inj-04-threshold-boundary" in item["injection_ids"]
            for item in injection_report["boundary_segments"]
        ),
        "three_algorithms_on_healthy_segments": any(
            {score["algorithm"] for score in segment["algorithm_scores"]}
            == {"crest_factor", "robust_mahalanobis", "spectral_residual"}
            for segment in injection_report["segments"]
        ),
    }
    if not all(injection_checks.values()):
        raise AssertionError(f"injection checks failed: {injection_checks}")

    quantized = copy.deepcopy(config)
    quantized["quantization"]["quantum"] = 2
    quant_one = run_pipeline(config, data, None, snapshot_dir=run_root / "quantum-1", mode="full")
    quant_two = run_pipeline(quantized, data, None, snapshot_dir=run_root / "quantum-2", mode="full")
    marks_one = [(item["device_id"], item["segment_index"]) for item in quant_one["segments"] if item["segment_anomaly"]]
    marks_two = [(item["device_id"], item["segment_index"]) for item in quant_two["segments"] if item["segment_anomaly"]]
    ticks_one = [[score["normalized_ticks"] for score in item["algorithm_scores"]] for item in quant_one["segments"]]
    ticks_two = [[score["normalized_ticks"] for score in item["algorithm_scores"]] for item in quant_two["segments"]]
    quantization_checks = {
        "fixed_point_integers_change": ticks_one != ticks_two,
        "anomaly_marks_change": marks_one != marks_two,
        "marks_quantum_1": [f"{device}/{index}" for device, index in marks_one],
        "marks_quantum_2": [f"{device}/{index}" for device, index in marks_two],
    }
    if not quantization_checks["fixed_point_integers_change"] or not quantization_checks["anomaly_marks_change"]:
        raise AssertionError(f"quantization checks failed: {quantization_checks}")

    serial_config = copy.deepcopy(config)
    serial_config["concurrency"] = 1
    parallel_config = copy.deepcopy(config)
    parallel_config["concurrency"] = 4
    serial = run_pipeline(serial_config, data, injections, snapshot_dir=run_root / "serial", mode="full")
    parallel = run_pipeline(parallel_config, data, injections, snapshot_dir=run_root / "parallel", mode="full")
    concurrency_checks = {
        "serial_parallel_result_fields_equal": _business_equal(serial, parallel),
        "serial_max_observed": serial["run"]["max_observed_concurrency"],
        "parallel_max_observed": parallel["run"]["max_observed_concurrency"],
        "parallel_within_limit": parallel["run"]["max_observed_concurrency"] <= 4,
        "failure_order_equal": serial["failures"] == parallel["failures"],
        "boundary_order_equal": serial["boundary_segments"] == parallel["boundary_segments"],
    }
    if not all(
        value
        for key, value in concurrency_checks.items()
        if key not in ("serial_max_observed", "parallel_max_observed")
    ):
        raise AssertionError(f"concurrency checks failed: {concurrency_checks}")

    atomic_write_json(artifacts / "incremental_report.json", incremental)
    verification = {
        "status": "pass",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "fixtures": {
            "config": "config/pipeline.yaml",
            "data": "data/devices.yaml",
            "injections": "config/injections.yaml",
            "config_hash": initial["config_hash"],
            "final_incremental_config_hash": incremental["config_hash"],
            "final_incremental_injection_hash": incremental["injection_hash"],
            "final_waveform_fingerprint": incremental["waveform_fingerprint"],
        },
        "incremental_vs_full": {
            "initial": initial_record,
            "steps": step_records,
            "business_fields_checked": [
                "segments (including features, raw/normalized algorithm scores and decisions)",
                "boundary_segments",
                "failures",
                "injection_hits",
                "waveform_fingerprint",
                "result_fingerprint",
            ],
            "provenance_note": (
                "reuse/recompute lists describe the execution path and intentionally differ between incremental "
                "and blank full rebuild; their canonical ordering is checked independently"
            ),
        },
        "cross_algorithm_invalidation": cross_algorithm_checks,
        "injection": {
            "checks": injection_checks,
            "hits": injection_report["injection_hits"],
            "failure_count": injection_report["failure_count"],
            "boundary_segments": injection_report["boundary_segments"],
            "report": "artifacts/report.json",
        },
        "quantization": quantization_checks,
        "concurrency": concurrency_checks,
        "final_incremental_report": "artifacts/incremental_report.json",
    }
    verification["summary"] = {
        "all_checks_pass": True,
        "incremental_steps": len(step_records),
        "injection_hits": sum(1 for item in injection_hits.values() if item["hit"]),
        "segments_in_injection_run": injection_report["segment_count"],
        "injection_failures_continued": injection_report["failure_count"],
        "final_incremental_reuse_count": incremental["reuse_count"],
        "final_incremental_recompute_count": incremental["recompute_count"],
    }
    # Keep this report free of any circular self-reference while preserving a stable hash.
    verification["verification_hash"] = __import__("hashlib").sha256(canonical_json(verification).encode("utf-8")).hexdigest()
    atomic_write_json(artifacts / "verification_report.json", verification)
    _write_results_markdown(verification, ROOT / "docs" / "RESULTS.md")
    print(json.dumps(verification["summary"], ensure_ascii=False, sort_keys=True))
    return 0


def _write_results_markdown(verification: dict[str, Any], path: Path) -> None:
    lines = [
        "# 实际验证结果",
        "",
        "本文件由 `python tools/verify.py` 在本地确定性样本上自动生成。",
        "",
        "## 环境",
        "",
        f"- Python `{verification['environment']['python']}`",
        f"- NumPy `{verification['environment']['numpy']}`",
        f"- Platform `{verification['environment']['platform']}`",
        "",
        "## 增量编辑序列",
        "",
        "编辑序列依次累积执行：追加分段、替换单段、删除分段、调整马氏距离候选参数、切换频带、切换定点量化档位、修改阈值。",
        "",
        "| 步骤 | 增量=全量 | 段数 | 复用 | 重算 | 快照码 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    records = [verification["incremental_vs_full"]["initial"]] + verification["incremental_vs_full"]["steps"]
    for item in records:
        lines.append(
            f"| {item['step']} | {str(item['business_fields_equal']).lower()} | {item['segments']} | "
            f"{item['reuse_count']} | {item['recompute_count']} | `{item['snapshot_stale_code'] or '-'}` |"
        )
    lines.extend(
        [
            "",
            "逐段特征摘要、三算法原始分数、定点归一化整数、融合总分、异常标记、边界清单、失败清单、注入命中与波形指纹均逐字段相等；增量与空白全量的复用/重算清单描述执行路径，按定义允许不同，但均验证了稳定排序。",
            "",
            "## 跨算法失效粒度",
            "",
        ]
    )
    for key, value in verification["cross_algorithm_invalidation"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            "",
            "## 注入与边界",
            "",
        ]
    )
    for key, value in verification["injection"]["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            f"- 注入运行段数：`{verification['summary']['segments_in_injection_run']}`",
            f"- 注入后失败记录数：`{verification['summary']['injection_failures_continued']}`（其余分段继续执行）",
            f"- 命中注入数：`{verification['summary']['injection_hits']}`",
            "",
            "## 量化与并发",
            "",
            f"- 定点整数随量化档位变化：`{str(verification['quantization']['fixed_point_integers_change']).lower()}`",
            f"- 异常标记随量化档位变化：`{str(verification['quantization']['anomaly_marks_change']).lower()}`；"
            f"档位 1 标记 `{verification['quantization']['marks_quantum_1']}`，档位 2 标记 `{verification['quantization']['marks_quantum_2']}`",
            f"- 串行与 4 线程业务结果一致：`{str(verification['concurrency']['serial_parallel_result_fields_equal']).lower()}`",
            f"- 最大实际并发：串行 `{verification['concurrency']['serial_max_observed']}`，并发版 `{verification['concurrency']['parallel_max_observed']}`",
            "",
            "## 报告文件",
            "",
            "- `artifacts/report.json`：带四类注入的完整运行报告",
            "- `artifacts/incremental_report.json`：最终编辑状态下的增量运行报告",
            "- `artifacts/verification_report.json`：机器可读验收汇总",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
