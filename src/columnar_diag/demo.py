from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .errors import AppError
from .models import BudgetConfig, TaskResult
from .planner import build_plans
from .sample import (
    build_sample_spec,
    sample_budget,
    sample_changes,
    write_sample_files,
)
from .snapshot import load_snapshot, restore_snapshot, save_snapshot
from .spec import parse_spec
from .util import atomic_write_json, canonical_json
from .verifier import authoritative_replay
from .workspace import Workspace


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _budget_from_sample() -> BudgetConfig:
    return BudgetConfig(**sample_budget())


def _constraint_evidence(metrics: dict[str, Any], budget: BudgetConfig) -> dict[str, Any]:
    column_memory = metrics.get("peak_column_memory_bytes", {})
    segment_tasks = metrics.get("peak_segment_tasks", {})
    return {
        "global_parallelism_budget": budget.global_parallelism,
        "peak_global_tasks": metrics.get("peak_global_tasks"),
        "global_within_budget": metrics.get("peak_global_tasks", 0)
        <= budget.global_parallelism,
        "column_memory_budget_bytes": budget.column_memory_budget_bytes,
        "peak_column_memory_bytes": column_memory,
        "column_memory_within_budget": all(
            value <= budget.column_memory_budget_bytes
            for value in column_memory.values()
        ),
        "segment_concurrency_quota": budget.segment_concurrency,
        "peak_segment_tasks": segment_tasks,
        "segment_concurrency_within_quota": all(
            value <= budget.segment_concurrency for value in segment_tasks.values()
        ),
        "peak_column_tasks_observed": metrics.get("peak_column_tasks", {}),
        "max_queue_wait_ms": round(float(metrics.get("max_queue_wait_ms", 0.0)), 4),
        "worker_threads": metrics.get("worker_threads"),
    }


def run_demo(output_dir: str | Path = "artifacts") -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_dir = output / "samples"
    _clean_dir(sample_dir)
    samples = write_sample_files(sample_dir)
    report: dict[str, Any] = {
        "application": "Columnar Segment Diagnostic",
        "python_environment": {
            "platform": "Windows-local target; verification executed headlessly where Tk runtime is unavailable",
            "sqlite": "stdlib",
            "numpy": "1.26.4",
        },
        "sample_files": samples,
    }

    raw = build_sample_spec().to_dict()
    normalized_raw = json.loads(canonical_json(raw))
    normalized_raw["queries"][0]["where"] = "((region IN (1,3))) AND (((amount>=10)))"
    spec_a = parse_spec(raw)
    spec_b = parse_spec(normalized_raw)
    plan_a = build_plans(spec_a)
    plan_b = build_plans(spec_b)
    report["plan_normalization"] = {
        "original_where": spec_a.queries[0].where,
        "normalized_where": spec_b.queries[0].where,
        "original_plan_digest": plan_a.plan_map["q1"].digest,
        "whitespace_parenthesis_plan_digest": plan_b.plan_map["q1"].digest,
        "same_digest": plan_a.plan_map["q1"].digest == plan_b.plan_map["q1"].digest,
        "q1_pruned_columns": list(plan_a.plan_map["q1"].pruned_columns),
        "q1_candidate_blocks": list(plan_a.plan_map["q1"].candidate_blocks),
    }

    incremental_dir = output / "incremental_workspace"
    _clean_dir(incremental_dir)
    initial_budget = _budget_from_sample()
    ws = Workspace.create(
        incremental_dir,
        build_sample_spec(),
        budget=initial_budget,
        overwrite=True,
    )
    try:
        initial_scan = ws.run_full_scan(batch_size=5, delay_ms=2.0)
        replay_initial = authoritative_replay(ws)
        before_block_digests = {
            block_id: encoded.block_digest
            for block_id, encoded in ws.image.blocks.items()
        }
        report["initial_scan"] = {
            "task_count": len(ws.bundle.tasks),
            "plan_digest": ws.bundle.digest,
            "committed_batches": len(initial_scan["batches"]),
            "cursor": initial_scan["cursor_after"],
            "constraints": _constraint_evidence(initial_scan["metrics"], initial_budget),
            "authoritative_replay": replay_initial,
            "statistics": ws.ledger.statistics(),
            "plans": [plan.to_dict() for plan in ws.bundle.plans],
        }

        change_payload = sample_changes()
        incremental = ws.apply_changeset(
            change_payload["changes"], delay_ms=1.5, reason="acceptance-incremental"
        )
        replay_incremental = authoritative_replay(ws)
        after_block_digests = {
            block_id: encoded.block_digest
            for block_id, encoded in ws.image.blocks.items()
        }
        changed_encoded_blocks = sorted(
            block_id
            for block_id in before_block_digests
            if before_block_digests[block_id] != after_block_digests[block_id]
        )
        report["incremental_update"] = {
            "changeset": change_payload,
            "affected_queries": incremental["affected_queries"],
            "direct_invalid_blocks": incremental["invalid_blocks"],
            "indirectly_affected_blocks": incremental[
                "indirectly_affected_blocks"
            ],
            "affected_block_set": incremental["affected_block_set"],
            "recomputed_encoding_blocks": incremental["recomputed_encoding_blocks"],
            "recomputed_scan_tasks": incremental["recomputed_scan_tasks"],
            "affected_task_count": len(incremental["affected_tasks"]),
            "propagation_steps": incremental["propagation_steps"],
            "candidate_transitions": incremental["candidate_transitions"],
            "preserved_block_count": len(incremental["preserved_blocks"]),
            "all_preserved_blocks_byte_identical": incremental[
                "all_preserved_blocks_byte_identical"
            ],
            "preserved_task_count": len(incremental["preserved_task_results"]),
            "all_preserved_tasks_identical": incremental[
                "all_preserved_tasks_identical"
            ],
            "preserved_query_result_count": len(
                incremental["preserved_query_results"]
            ),
            "all_preserved_query_results_identical": incremental[
                "all_preserved_query_results_identical"
            ],
            "only_expected_block_encoding_changed": changed_encoded_blocks
            == incremental["invalid_blocks"],
            "changed_encoded_blocks": changed_encoded_blocks,
            "versions_after_update": incremental["versions"],
            "encoding_version_bumped": incremental["encoding_version_bumped"],
            "conclusion_diff": incremental["conclusion_diff"],
            "authoritative_replay": replay_incremental,
            "stale_generation_count": len(
                [row for row in ws.ledger.generation_rows() if not int(row["is_current"])]
            ),
        }

        new_budget = BudgetConfig(
            global_parallelism=3,
            column_memory_budget_bytes=100,
            segment_concurrency=2,
        )
        budget_change = ws.change_budget(new_budget, batch_size=6, delay_ms=1.0)
        replay_budget = authoritative_replay(ws)
        report["budget_version_evolution"] = {
            "version_changed": budget_change["version_changed"],
            "budget_params_version": budget_change["budget_params_version"],
            "budget": budget_change["budget"],
            "constraints": _constraint_evidence(
                budget_change["scan"]["metrics"], new_budget
            ),
            "conclusion_diff": budget_change["conclusion_diff"],
            "stale_generations": [
                {
                    "generation_id": row["generation_id"],
                    "budget_version": row["budget_version"],
                    "status": row["status"],
                    "is_current": row["is_current"],
                }
                for row in budget_change["stale_generations"]
            ],
            "authoritative_replay": replay_budget,
        }

        encoding_change = ws.change_encoding_policy(
            column="region",
            encoding="raw",
            compressed=False,
            delay_ms=1.0,
        )
        replay_encoding = authoritative_replay(ws)
        report["encoding_policy_evolution"] = {
            "version_changed": encoding_change["version_changed"],
            "encoding_selection_version": encoding_change[
                "encoding_selection_version"
            ],
            "recomputed_encoding_blocks": encoding_change[
                "recomputed_encoding_blocks"
            ],
            "recomputed_scan_tasks": encoding_change["recomputed_scan_tasks"],
            "preserved_blocks_identical": encoding_change[
                "all_preserved_blocks_byte_identical"
            ],
            "preserved_tasks_identical": encoding_change[
                "all_preserved_tasks_identical"
            ],
            "conclusion_diff": encoding_change["conclusion_diff"],
            "authoritative_replay": replay_encoding,
            "current_generation": ws.ledger.current_generation(),
            "generations": ws.ledger.generation_rows(),
        }
    finally:
        ws.close()

    snapshot_dir = output / "snapshot_workspace"
    _clean_dir(snapshot_dir)
    ws_snapshot = Workspace.create(
        snapshot_dir,
        build_sample_spec(),
        budget=_budget_from_sample(),
        overwrite=True,
    )
    try:
        partial = ws_snapshot.scan_pending(batch_size=5, max_tasks=10, delay_ms=2.0)
        snapshot_path = snapshot_dir / "session_snapshot.json"
        snapshot_save = save_snapshot(ws_snapshot, snapshot_path)
        continued_before_restore = ws_snapshot.scan_pending(batch_size=5)
        restored = restore_snapshot(ws_snapshot, snapshot_path)
        cursor_after_restore = int(ws_snapshot.ledger.current_generation()["cursor"])
        resumed = ws_snapshot.scan_pending(batch_size=5, delay_ms=1.0)
        replay_after_resume = authoritative_replay(ws_snapshot)

        snapshot_for_tamper = load_snapshot(snapshot_path)
        tampered_snapshot_path = snapshot_dir / "snapshot_checksum_tampered.json"
        tampered_snapshot = json.loads(json.dumps(snapshot_for_tamper))
        tampered_snapshot["statistics"]["completed_rows"] += 1
        atomic_write_json(tampered_snapshot_path, tampered_snapshot)
        checksum_error: dict[str, Any]
        try:
            load_snapshot(tampered_snapshot_path)
            checksum_error = {"raised": False}
        except AppError as error:
            checksum_error = error.to_dict()

        row = ws_snapshot.ledger.current_task_rows()[2]
        with ws_snapshot.ledger.conn:
            ws_snapshot.ledger.conn.execute(
                "UPDATE scan_ledger SET result_digest=? WHERE generation_id=? AND task_id=? AND is_current=1",
                ("0" * 64, int(ws_snapshot.ledger.current_generation()["generation_id"]), row["task_id"]),
            )
        diverge_error: dict[str, Any]
        try:
            authoritative_replay(ws_snapshot)
            diverge_error = {"raised": False}
        except AppError as error:
            diverge_error = error.to_dict()
        replay_error: dict[str, Any]
        try:
            authoritative_replay(ws_snapshot, error_code="E_REPLAY")
            replay_error = {"raised": False}
        except AppError as error:
            replay_error = error.to_dict()
        restore_snapshot(ws_snapshot, snapshot_path)
        ws_snapshot.scan_pending(batch_size=5)
        replay_after_repair = authoritative_replay(
            ws_snapshot, error_code="E_REPLAY"
        )

        batch_count_before = len(ws_snapshot.ledger.current_batches())
        first_task = ws_snapshot.bundle.tasks[0]
        dummy_result = TaskResult(
            task=first_task,
            completed_rows=0,
            decode_memory_bytes=0,
            queue_started_ms=0.0,
            wait_ended_ms=0.0,
            run_started_ms=0.0,
            run_ended_ms=0.0,
            duration_ms=0.0,
            block_digest="fault-test",
            result_digest="fault-test",
            result_json={"fault": "injected"},
            stats={},
        )
        atomic_error: dict[str, Any]
        try:
            ws_snapshot.ledger.commit_batch(
                generation_id=int(ws_snapshot.ledger.current_generation()["generation_id"]),
                reason="forced-termination-simulation",
                task_results=[dummy_result],
                query_results={},
                plans={},
                cursor_before=10,
                cursor_after=11,
                tasks_total=len(ws_snapshot.bundle.tasks),
                budget_version=ws_snapshot.versions["budget_params_version"],
                encoding_version=ws_snapshot.versions["encoding_selection_version"],
                segment_digest=ws_snapshot.image.file_digest,
                plan_digest=ws_snapshot.bundle.digest,
                fail_after_rows=1,
            )
            atomic_error = {"raised": False}
        except AppError as error:
            atomic_error = error.to_dict()
        batch_count_after = len(ws_snapshot.ledger.current_batches())
        replay_after_fault = authoritative_replay(
            ws_snapshot, error_code="E_REPLAY"
        )

        report["snapshot_recovery"] = {
            "snapshot": snapshot_save,
            "partial_scan": {
                "cursor_before": partial["cursor_before"],
                "cursor_after": partial["cursor_after"],
                "batches": partial["batches"],
            },
            "continued_before_restore_cursor": continued_before_restore[
                "cursor_after"
            ],
            "restore": restored,
            "cursor_after_restore": cursor_after_restore,
            "resume": {
                "cursor_before": resumed["cursor_before"],
                "cursor_after": resumed["cursor_after"],
            },
            "replay_after_resume": replay_after_resume,
            "snapshot_checksum_tamper": checksum_error,
            "ledger_digest_tamper_diverge": diverge_error,
            "ledger_digest_tamper_replay": replay_error,
            "replay_after_repair": replay_after_repair,
            "forced_termination": {
                "injected_error": atomic_error,
                "committed_batches_before": batch_count_before,
                "committed_batches_after": batch_count_after,
                "rollback_complete": batch_count_before == batch_count_after,
                "replay_after_fault": replay_after_fault,
            },
        }
    finally:
        ws_snapshot.close()

    checks = {
        "plan_normalization_stable": report["plan_normalization"]["same_digest"],
        "initial_replay_ok": report["initial_scan"]["authoritative_replay"]["ok"],
        "global_budget_respected": report["initial_scan"]["constraints"][
            "global_within_budget"
        ],
        "column_memory_budget_respected": report["initial_scan"]["constraints"][
            "column_memory_within_budget"
        ],
        "segment_quota_respected": report["initial_scan"]["constraints"][
            "segment_concurrency_within_quota"
        ],
        "incremental_only_target_encoding_changed": report["incremental_update"][
            "only_expected_block_encoding_changed"
        ],
        "incremental_preserved_blocks_byte_identical": report["incremental_update"][
            "all_preserved_blocks_byte_identical"
        ],
        "incremental_preserved_tasks_identical": report["incremental_update"][
            "all_preserved_tasks_identical"
        ],
        "incremental_preserved_query_results_identical": report[
            "incremental_update"
        ]["all_preserved_query_results_identical"],
        "incremental_authoritative_replay_ok": report["incremental_update"][
            "authoritative_replay"
        ]["ok"],
        "budget_version_monotonic": report["budget_version_evolution"][
            "budget_params_version"
        ]
        > 1,
        "old_generations_stale": all(
            row["is_current"] == 0
            for row in report["budget_version_evolution"]["stale_generations"]
        ),
        "encoding_version_monotonic": report["encoding_policy_evolution"][
            "encoding_selection_version"
        ]
        > 1,
        "snapshot_cursor_restored": report["snapshot_recovery"][
            "cursor_after_restore"
        ]
        == 10,
        "snapshot_replay_ok": report["snapshot_recovery"]["replay_after_resume"][
            "ok"
        ],
        "snapshot_checksum_rejected": report["snapshot_recovery"][
            "snapshot_checksum_tamper"
        ].get("code")
        == "E_SNAPSHOT",
        "ledger_tamper_reported_replay": report["snapshot_recovery"][
            "ledger_digest_tamper_replay"
        ].get("code")
        == "E_REPLAY",
        "ledger_tamper_reported_diverge": report["snapshot_recovery"][
            "ledger_digest_tamper_diverge"
        ].get("code")
        == "E_DIVERGE",
        "forced_termination_rolled_back": report["snapshot_recovery"][
            "forced_termination"
        ]["rollback_complete"],
    }
    report["checks"] = checks
    report["all_checks_passed"] = all(checks.values())
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    atomic_write_json(output / "acceptance_report.json", report)
    (output / "ACCEPTANCE.md").write_text(
        _render_acceptance_markdown(report), encoding="utf-8"
    )
    return report


def _render_acceptance_markdown(report: dict[str, Any]) -> str:
    initial = report["initial_scan"]
    incremental = report["incremental_update"]
    budget = report["budget_version_evolution"]
    encoding = report["encoding_policy_evolution"]
    snapshot = report["snapshot_recovery"]
    lines = [
        "# 本地验收结果",
        "",
        f"- 总检查: **{'PASS' if report['all_checks_passed'] else 'FAIL'}**",
        f"- 用时: {report['elapsed_seconds']} 秒",
        "",
        "## 计划与初次扫描",
        f"- 块任务数: {initial['task_count']}",
        f"- 计划摘要: `{initial['plan_digest']}`",
        f"- 空白/括号归一化摘要一致: {report['plan_normalization']['same_digest']}",
        f"- 全局峰值/预算: {initial['constraints']['peak_global_tasks']} / {initial['constraints']['global_parallelism_budget']}",
        f"- 段并发峰值/配额: {initial['constraints']['peak_segment_tasks']} / {initial['constraints']['segment_concurrency_quota']}",
        f"- 每列解码内存峰值: `{initial['constraints']['peak_column_memory_bytes']}`，预算 {initial['constraints']['column_memory_budget_bytes']} 字节",
        f"- 排队最大等待: {initial['constraints']['max_queue_wait_ms']} ms",
        "",
        "## 增量失效与仅失效子集重算",
        f"- 直接失效块: `{incremental['direct_invalid_blocks']}`",
        f"- 间接扫描依赖块: `{incremental['indirectly_affected_blocks']}`",
        f"- 重编码块: `{incremental['recomputed_encoding_blocks']}`",
        f"- 受影响查询: `{incremental['affected_queries']}`",
        f"- 重算扫描任务数: {len(incremental['recomputed_scan_tasks'])}",
        f"- 未失效块保持字节一致: {incremental['all_preserved_blocks_byte_identical']}（{incremental['preserved_block_count']} 块）",
        f"- 未失效任务结果保持逐项一致: {incremental['all_preserved_tasks_identical']}（{incremental['preserved_task_count']} 项）",
        f"- 未失效查询结论保持逐项一致: {incremental['all_preserved_query_results_identical']}（{incremental['preserved_query_result_count']} 项）",
        f"- 权威重放: {incremental['authoritative_replay']['ok']}",
        "",
        "## 版本演进",
        f"- 预算版本: {budget['budget_params_version']}，旧生成全部陈旧: {report['checks']['old_generations_stale']}",
        f"- 预算版本结论差异: `{budget['conclusion_diff']['counts']}`",
        f"- 编码选择版本: {encoding['encoding_selection_version']}",
        f"- 编码版本结论差异: `{encoding['conclusion_diff']['counts']}`",
        f"- 编码版本权威重放: {encoding['authoritative_replay']['ok']}",
        "",
        "## 快照恢复与强制终止",
        f"- 快照游标: {snapshot['snapshot']['cursor']}",
        f"- 恢复后游标: {snapshot['cursor_after_restore']}",
        f"- 恢复前已推进到: {snapshot['continued_before_restore_cursor']}",
        f"- 继续扫描后游标: {snapshot['resume']['cursor_after']}",
        f"- 恢复后逐块重放: {snapshot['replay_after_resume']['ok']}",
        f"- 快照内容摘要篡改拒绝: `{snapshot['snapshot_checksum_tamper'].get('code')}`",
        f"- 账本与权威重算发散: `{snapshot['ledger_digest_tamper_diverge'].get('code')}` block_ordinal={snapshot['ledger_digest_tamper_diverge'].get('block_ordinal')} field={snapshot['ledger_digest_tamper_diverge'].get('field')}",
        f"- 账本摘要篡改定位: `{snapshot['ledger_digest_tamper_replay'].get('code')}` block_ordinal={snapshot['ledger_digest_tamper_replay'].get('block_ordinal')} field={snapshot['ledger_digest_tamper_replay'].get('field')}",
        f"- 强制终止批次回滚: {snapshot['forced_termination']['rollback_complete']}",
        "",
        "机器可读明细见 `acceptance_report.json`。",
    ]
    return "\n".join(lines) + "\n"
