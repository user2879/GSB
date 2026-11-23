"""Repeatable end-to-end demonstration and diagnostic-report builder."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .config import FingerprintConfig, config_from_overrides, load_config, validate_config_payload
from .engine import NearDuplicateEngine, SimulatedCrash
from .errors import (
    E_CLUSTER_ID_CONFLICT,
    E_CONFIG_INVALID,
    E_PLAN_INVALID,
    E_SIGNATURE_BUDGET_PENDING,
    E_SNAPSHOT_VERSION_MISMATCH,
    E_TEXT_INVALID,
    BusinessError,
)
from .plan import apply_edit_batch, load_edit_plan, load_injections
from .sample_data import BASE_CONFIG, generate_samples
from .text_store import load_text_jsonl, records_by_id


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _count_codes(failures: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(item["code"] for item in failures).items()))


def _cluster_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "cluster_id": cluster["cluster_id"],
            "members": cluster["members"],
            "singleton": cluster["singleton"],
            "bucket": cluster["bucket"],
        }
        for cluster in result["clusters"]
    ]


def _safe_injection_payload(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    value = result.get("value")
    if isinstance(value, float) and value != value:
        result["value"] = "NaN"
    return result


def run_demo(
    *,
    sample_dir: str = "samples",
    run_dir: str = "runs/demo",
    report_dir: str = "reports",
) -> dict[str, Any]:
    samples = Path(sample_dir)
    runs = Path(run_dir)
    reports = Path(report_dir)
    reports.mkdir(parents=True, exist_ok=True)
    if runs.exists():
        shutil.rmtree(runs)
    runs.mkdir(parents=True, exist_ok=True)

    generation = generate_samples(str(samples))
    config, config_failures = load_config(str(samples / "config.yaml"))
    if config is None:
        raise RuntimeError([failure.to_dict() for failure in config_failures])
    records, text_failures = load_text_jsonl(str(samples / "texts.jsonl"), config)
    if text_failures:
        raise RuntimeError([failure.to_dict() for failure in text_failures])
    initial_records = records_by_id(records)
    initial_config = config

    plans, plan_load_failures = load_edit_plan(str(samples / "edit_plan.json"), initial_config)
    if plan_load_failures:
        raise RuntimeError([failure.to_dict() for failure in plan_load_failures])
    injections, injection_load_failures = load_injections(str(samples / "injections.json"))
    if injection_load_failures:
        raise RuntimeError([failure.to_dict() for failure in injection_load_failures])

    invalid_records, invalid_text_failures = load_text_jsonl(
        str(samples / "invalid_texts.jsonl"), initial_config
    )
    _, invalid_plan_failures = load_edit_plan(str(samples / "invalid_plan.json"), initial_config)
    validation_report = {
        "invalid_texts": {
            "accepted_records": [record.text_id for record in invalid_records],
            "failure_count": len(invalid_text_failures),
            "codes": _count_codes([failure.to_dict() for failure in invalid_text_failures]),
            "samples": [failure.to_dict() for failure in invalid_text_failures[:8]],
            "expected_code_present": any(
                failure.code == E_TEXT_INVALID for failure in invalid_text_failures
            ),
        },
        "invalid_plan": {
            "failure_count": len(invalid_plan_failures),
            "codes": _count_codes([failure.to_dict() for failure in invalid_plan_failures]),
            "samples": [failure.to_dict() for failure in invalid_plan_failures[:8]],
            "expected_codes_present": {
                E_PLAN_INVALID: any(failure.code == E_PLAN_INVALID for failure in invalid_plan_failures),
                E_TEXT_INVALID: any(failure.code == E_TEXT_INVALID for failure in invalid_plan_failures),
            },
        },
    }

    incremental_dir = runs / "incremental"
    incremental_engine = NearDuplicateEngine(str(incremental_dir))
    initial_result = incremental_engine.sync(records, initial_config)
    sequence: list[dict[str, Any]] = []
    current_records = dict(initial_records)
    current_config = initial_config
    accumulated_failures = [failure.to_dict() for failure in plan_load_failures]
    for batch_number, operations in sorted(plans.items()):
        current_records, current_config, batch_failures = apply_edit_batch(
            current_records, current_config, operations
        )
        accumulated_failures.extend(failure.to_dict() for failure in batch_failures)
        result = incremental_engine.sync(
            current_records.values(),
            current_config,
            external_failures=accumulated_failures,
        )
        sequence.append(
            {
                "batch": batch_number,
                "config_hash": current_config.config_hash,
                "signature_invalidated": result["audit"]["signature_invalidated"],
                "signature_reused_count": len(result["audit"]["signature_reused"]),
                "signature_computed": result["audit"]["signature_computed"],
                "bucket_invalidated_count": len(result["audit"]["bucket_invalidated"]),
                "cluster_invalidated_buckets": result["audit"]["cluster_invalidated_buckets"],
                "cluster_reused_buckets": result["audit"]["cluster_reused_buckets"],
                "pending_signatures": result["pending_signatures"],
                "failure_codes": _count_codes(result["failures"]),
                "manifest_version": result["manifest"]["global_version"],
            }
        )

    final_result = incremental_engine.sync(
        current_records.values(),
        current_config,
        external_failures=accumulated_failures,
    )
    incremental_projection = incremental_engine.state_projection(final_result)

    full_dir = runs / "full-rebuild"
    full_engine = NearDuplicateEngine(str(full_dir))
    full_result = full_engine.sync(
        current_records.values(),
        current_config,
        external_failures=accumulated_failures,
    )
    full_projection = full_engine.state_projection(full_result)
    equality = {
        "equal": incremental_projection == full_projection,
        "incremental_projection_sha256": _digest(incremental_projection),
        "full_projection_sha256": _digest(full_projection),
        "fields": sorted(incremental_projection),
    }
    if not equality["equal"]:
        differing = [
            field
            for field in incremental_projection
            if incremental_projection[field] != full_projection[field]
        ]
        equality["differing_fields"] = differing

    threshold_batch = next(
        (item for item in sequence if item["batch"] == 4),
        None,
    )
    reuse_evidence = {
        "batch": 4,
        "description": "similarity_threshold-only edit",
        "signature_invalidated": threshold_batch["signature_invalidated"] if threshold_batch else None,
        "signature_reused_count": threshold_batch["signature_reused_count"] if threshold_batch else None,
        "all_signatures_reused": bool(
            threshold_batch
            and not threshold_batch["signature_invalidated"]
            and threshold_batch["signature_reused_count"] == len(current_records)
        ),
    }

    budget_config, budget_failures = config_from_overrides(
        initial_config,
        [("signature_batch_budget", initial_config.signature_dimension)],
        source="budget_demo",
    )
    if budget_config is None:
        raise RuntimeError([failure.to_dict() for failure in budget_failures])
    budget_engine = NearDuplicateEngine(str(runs / "budget"))
    budget_first = budget_engine.sync(initial_records.values(), budget_config)
    budget_second = budget_engine.sync(initial_records.values(), budget_config)
    budget_report = {
        "limit": budget_config.signature_batch_budget,
        "per_text_cost": budget_config.signature_dimension,
        "first_run_computed": budget_first["audit"]["signature_computed"],
        "first_run_pending": budget_first["pending_signatures"],
        "first_run_consumed": budget_first["manifest"]["signature_budget"]["batch_consumed"],
        "first_run_pending_failure_count": sum(
            1 for failure in budget_first["failures"] if failure["code"] == E_SIGNATURE_BUDGET_PENDING
        ),
        "second_run_computed": budget_second["audit"]["signature_computed"],
        "second_run_pending": budget_second["pending_signatures"],
        "pending_priority_preserved": (
            budget_first["pending_signatures"][0] == budget_second["audit"]["signature_computed"][0]
            and budget_second["pending_signatures"] == budget_first["pending_signatures"][1:]
        ),
        "consumed_total": budget_second["manifest"]["signature_budget"]["consumed_total"],
    }

    recovery_dir = runs / "recovery"
    seed_engine = NearDuplicateEngine(str(recovery_dir))
    seed_engine.sync(initial_records.values(), initial_config)
    crash_engine = NearDuplicateEngine(str(recovery_dir), crash_after_stage="signatures")
    crash_error: SimulatedCrash | None = None
    try:
        crash_engine.sync(current_records.values(), current_config)
    except SimulatedCrash as exc:
        crash_error = exc
    partial = NearDuplicateEngine(str(recovery_dir)).latest_snapshot()
    if partial is None:
        raise AssertionError("crash demonstration did not leave a resumable snapshot")
    partial_manifest = partial["manifest"]
    resumed_engine = NearDuplicateEngine(str(recovery_dir))
    resumed_result = resumed_engine.sync(current_records.values(), current_config)
    resumed_projection = resumed_engine.state_projection(resumed_result)
    recovery_report = {
        "simulated_crash": str(crash_error) if crash_error else None,
        "partial_generation": partial_manifest.get("generation"),
        "partial_version": partial_manifest.get("global_version"),
        "partial_completed_count": len(partial_manifest.get("completed_texts", [])),
        "partial_pending_count": len(partial_manifest.get("pending_signatures", [])),
        "resumed_projection_sha256": _digest(resumed_projection),
        "full_projection_sha256": equality["full_projection_sha256"],
        "resumed_equals_full": resumed_projection == full_projection,
    }

    version_mismatch = None
    hash_mismatch = None
    checker = NearDuplicateEngine(str(full_dir))
    try:
        checker.recover_manifest(expected_version=999999)
    except BusinessError as exc:
        version_mismatch = {"code": exc.code, "detail": exc.detail}
    try:
        checker.recover_manifest(expected_config_hash="not-the-config-hash")
    except BusinessError as exc:
        hash_mismatch = {"code": exc.code, "detail": exc.detail}
    recovery_report["version_mismatch"] = version_mismatch
    recovery_report["config_hash_mismatch"] = hash_mismatch
    recovery_report["mismatch_code_expected"] = E_SNAPSHOT_VERSION_MISMATCH

    serial_config, serial_failures = config_from_overrides(
        current_config, [("max_workers", 1)], source="serial_demo"
    )
    concurrent_config, concurrent_failures = config_from_overrides(
        current_config, [("max_workers", max(4, current_config.max_workers))], source="concurrent_demo"
    )
    if serial_config is None or concurrent_config is None:
        raise RuntimeError(
            [failure.to_dict() for failure in serial_failures + concurrent_failures]
        )
    serial_engine = NearDuplicateEngine(str(runs / "serial"))
    concurrent_engine = NearDuplicateEngine(str(runs / "concurrent"))
    serial_result = serial_engine.sync(current_records.values(), serial_config)
    concurrent_result = concurrent_engine.sync(current_records.values(), concurrent_config)
    serial_projection = serial_engine.state_projection(serial_result)
    concurrent_projection = concurrent_engine.state_projection(concurrent_result)
    concurrency_report = {
        "serial_workers": serial_config.max_workers,
        "concurrent_workers": concurrent_config.max_workers,
        "equal": serial_projection == concurrent_projection,
        "serial_projection_sha256": _digest(serial_projection),
        "concurrent_projection_sha256": _digest(concurrent_projection),
    }

    injection_results: list[dict[str, Any]] = []
    for injection in injections:
        if injection.kind == "cluster_id_conflict":
            injection_engine = NearDuplicateEngine(str(runs / "injection-cluster"))
            observed_code = None
            observed_message = None
            try:
                injection_engine.sync(
                    initial_records.values(),
                    initial_config,
                    runtime_injection=injection.payload,
                )
            except BusinessError as exc:
                observed_code = exc.code
                observed_message = exc.message
            injection_results.append(
                {
                    "injection_id": injection.injection_id,
                    "kind": injection.kind,
                    "payload": _safe_injection_payload(injection.payload),
                    "expected_code": E_CLUSTER_ID_CONFLICT,
                    "observed_code": observed_code,
                    "message": observed_message,
                    "hit": observed_code == E_CLUSTER_ID_CONFLICT,
                }
            )
            continue
        raw = dict(BASE_CONFIG)
        field = str(injection.payload.get("field"))
        if injection.kind == "non_finite_similarity":
            raw[field] = float("nan")
        elif injection.kind == "zero_signature_dimension":
            raw[field] = 0
        elif injection.kind == "bucket_bits_out_of_range":
            raw[field] = int(initial_config.signature_dimension) + 1
        _, failures = validate_config_payload(raw, source=injection.injection_id)
        observed_code = failures[0].code if failures else None
        injection_results.append(
            {
                "injection_id": injection.injection_id,
                "kind": injection.kind,
                "payload": _safe_injection_payload(injection.payload),
                "expected_code": E_CONFIG_INVALID,
                "observed_code": observed_code,
                "message": failures[0].message if failures else None,
                "hit": observed_code == E_CONFIG_INVALID,
            }
        )

    initial_cluster_summary = _cluster_summary(initial_result)
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _numpy_version(),
        },
        "samples": generation,
        "initial_state": {
            "clusters": initial_cluster_summary,
            "cluster_count": len(initial_cluster_summary),
            "singleton_cluster_ids": [
                cluster["cluster_id"] for cluster in initial_cluster_summary if cluster["singleton"]
            ],
            "boundary_texts": initial_result["boundary_texts"],
            "boundary_count": len(initial_result["boundary_texts"]),
            "merge_event_count": len(initial_result["merge_log"]),
        },
        "validation": validation_report,
        "incremental_sequence": sequence,
        "cache_reuse": reuse_evidence,
        "incremental_vs_full_rebuild": equality,
        "final_state": {
            "config_hash": current_config.config_hash,
            "text_count": len(current_records),
            "cluster_count": len(final_result["clusters"]),
            "boundary_count": len(final_result["boundary_texts"]),
            "pending_signatures": final_result["pending_signatures"],
            "failure_count": len(final_result["failures"]),
            "clusters": _cluster_summary(final_result),
        },
        "budget_scheduler": budget_report,
        "concurrency": concurrency_report,
        "snapshot_recovery": recovery_report,
        "injections": injection_results,
        "snapshot_locations": {
            "incremental": str(incremental_dir),
            "full_rebuild": str(full_dir),
            "budget": str(runs / "budget"),
            "recovery": str(recovery_dir),
            "serial": str(runs / "serial"),
            "concurrent": str(runs / "concurrent"),
        },
    }
    (reports / "diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (reports / "verification.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _numpy_version() -> str | None:
    try:
        import numpy

        return numpy.__version__
    except Exception:
        return None


def _render_markdown(report: dict[str, Any]) -> str:
    initial = report["initial_state"]
    budget = report["budget_scheduler"]
    recovery = report["snapshot_recovery"]
    concurrency = report["concurrency"]
    equality = report["incremental_vs_full_rebuild"]
    injections = report["injections"]
    lines = [
        "# 近重复探测组件验证报告",
        "",
        "本报告由 `python -m near_dup.cli demo` 自动生成，所有数据来自本地固定种子样本。",
        "",
        "## 环境与样本",
        "",
        f"- Python: `{report['environment']['python']}`",
        f"- NumPy: `{report['environment']['numpy']}`",
        f"- 初始文本数: `{report['samples']['text_count']}`",
        f"- 初始簇数: `{initial['cluster_count']}`",
        f"- 单元素簇: `{', '.join(initial['singleton_cluster_ids'])}`",
        f"- 边界文本对数: `{initial['boundary_count']}`",
        "",
        "## 校验与故障注入",
        "",
        f"- 非法文本库命中码: `{report['validation']['invalid_texts']['codes']}`",
        f"- 非法编辑计划命中码: `{report['validation']['invalid_plan']['codes']}`",
    ]
    for item in injections:
        lines.append(
            f"- `{item['injection_id']}` / `{item['kind']}`: 期望 `{item['expected_code']}`，实际 `{item['observed_code']}`，命中={item['hit']}"
        )
    lines.extend(
        [
            "",
            "## 增量、预算与并发",
            "",
            f"- 阈值单改时签名复用: `{report['cache_reuse']['all_signatures_reused']}`（复用 {report['cache_reuse']['signature_reused_count']} 条）",
            f"- 首轮预算计算: `{budget['first_run_computed']}`，剩余待办: `{len(budget['first_run_pending'])}`",
            f"- 待办优先: `{budget['pending_priority_preserved']}`；累计预算: `{budget['consumed_total']}`",
            f"- 并发/串行逐字段一致: `{concurrency['equal']}`",
            f"- 增量与全量重建逐字段一致: `{equality['equal']}`",
            f"- 投影哈希: `{equality['incremental_projection_sha256']}`",
            "",
            "## 快照恢复",
            "",
            f"- 模拟阶段中断: `{recovery['simulated_crash']}`",
            f"- 中断快照: 版本 `{recovery['partial_version']}`，已完成 `{recovery['partial_completed_count']}`，待办 `{recovery['partial_pending_count']}`",
            f"- 恢复结果等于全量重建: `{recovery['resumed_equals_full']}`",
            f"- 版本不符码: `{recovery['version_mismatch']['code'] if recovery['version_mismatch'] else None}`",
            f"- 配置哈希不符码: `{recovery['config_hash_mismatch']['code'] if recovery['config_hash_mismatch'] else None}`",
            "",
            "## 复现命令",
            "",
            "```bash",
            ".venv/bin/python scripts/generate_samples.py --out samples",
            ".venv/bin/python -m near_dup.cli demo --samples samples --run-dir runs/demo --report-dir reports",
            ".venv/bin/python -m unittest discover -s tests -v",
            "```",
            "",
            "机器可读明细见 `reports/diagnostic_report.json`。",
            "",
        ]
    )
    return "\n".join(lines)

