from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

from .config import MODE_FIXED, ORDER_B_ABS_DESC, SpMMConfig
from .audit import replay_audit
from .csr import generate_csr
from .engine import (
    IncrementalEngine,
    RunReport,
    compare_incremental_to_full,
    full_rebuild,
)
from .fixture import DemoFixture, make_demo_fixture
from .injections import run_injection_suite
from .kernels import compare_float_and_fixed
from .kernels import dense_fingerprint
from .snapshot import SnapshotStore, bootstrap_engine


def _distribution_report(fixture: DemoFixture) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind, spec in fixture.distributions.items():
        matrix = generate_csr(30, 18, spec, seed=fixture.seed + 202)
        counts = np.diff(matrix.row_offsets)
        result[kind] = {
            "params": {
                "min_nnz": spec.min_nnz,
                "max_nnz": spec.max_nnz,
                "power": spec.power,
                "zero_probability": spec.zero_probability,
                "extreme_fraction": spec.extreme_fraction,
            },
            "nnz": matrix.nnz,
            "empty_rows": int(np.sum(counts == 0)),
            "max_row_nnz": int(counts.max(initial=0)),
            "valid": not matrix.validate(),
            "fingerprint": matrix.fingerprint(),
        }
    return result


def _report_entry(
    step: str,
    engine: IncrementalEngine,
    report: RunReport,
    edit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "step": step,
        "edit": edit or {},
        "report": report.to_dict(),
        "hot_table": engine.hot.to_dict(),
        "pending_after": sorted(engine.pending),
    }


def _run_and_drain(
    engine: IncrementalEngine,
    step: str,
    *,
    budget: int | None = None,
    edit: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], RunReport]:
    first = engine.run(budget=budget)
    entries = [_report_entry(step, engine, first, edit)]
    while engine.pending:
        followup = engine.run()
        entries.append(_report_entry(f"{step}:drain", engine, followup))
        if followup.run_id > first.run_id + 100:
            raise RuntimeError("demo drain did not converge")
    return entries, first


def _exercise_script(fixture: DemoFixture, config: SpMMConfig) -> tuple[IncrementalEngine, list[RunReport]]:
    engine = IncrementalEngine(fixture.csr, fixture.b_matrix, fixture.c_matrix, config)
    reports: list[RunReport] = []
    reports.append(engine.run())
    if engine.pending:
        reports.extend(engine.run_until_complete())
    engine.replace_row(4, [0, 4, 11, 17], [1.1, -0.4, 0.7, -1.2])
    reports.append(engine.run())
    reports.append(engine.run(budget=0))
    if engine.pending:
        reports.extend(engine.run_until_complete())
    engine.set_alpha_beta(config.alpha * 1.1 + 0.01, config.beta + 0.05)
    reports.append(engine.run(budget=max(0, engine._block_cost(0))))
    if engine.pending:
        reports.extend(engine.run_until_complete())
    return engine, reports


def _concurrency_report(fixture: DemoFixture) -> dict[str, Any]:
    single, single_reports = _exercise_script(fixture, fixture.single_thread_config)
    concurrent, concurrent_reports = _exercise_script(fixture, fixture.config)
    mismatches: list[int] = []
    for index, (left, right) in enumerate(zip(single_reports, concurrent_reports)):
        left_value = left.to_dict()
        right_value = right.to_dict()
        left_value.pop("max_workers_used")
        right_value.pop("max_workers_used")
        if left_value != right_value:
            mismatches.append(index)
    return {
        "single_workers": fixture.single_thread_config.max_workers,
        "concurrent_workers": fixture.config.max_workers,
        "report_count": len(single_reports),
        "report_mismatch_indices": mismatches,
        "numeric_exact": bool(np.array_equal(single.materialize_c(), concurrent.materialize_c())),
        "hot_table_exact": single.hot.to_dict() == concurrent.hot.to_dict(),
        "budget_ledger_exact": single.budget_ledger == concurrent.budget_ledger,
        "output_fingerprint_exact": single.output_fingerprint() == concurrent.output_fingerprint(),
    }


def _copy_store(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def run_demo(
    *,
    seed: int = 20261123,
    output_dir: str | Path = "artifacts",
    snapshot_dir: str | Path = "artifacts/snapshots",
) -> dict[str, Any]:
    fixture = make_demo_fixture(seed)
    output_path = Path(output_dir)
    snapshot_path = Path(snapshot_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    store = SnapshotStore(snapshot_path)
    engine = IncrementalEngine(fixture.csr, fixture.b_matrix, fixture.c_matrix, fixture.config)
    timeline: list[dict[str, Any]] = []

    initial_reports, initial = _run_and_drain(engine, "initial_build")
    timeline.extend(initial_reports)
    store.save(engine)

    target_row = 4
    target_block = target_row // engine.config.row_block_size
    dependency_before = engine.dependencies[target_block]
    engine.replace_row(target_row, [0, 3, 9, 17], [1.2, -0.6, 0.9, -1.4])
    replace_reports, replace_first = _run_and_drain(
        engine,
        "replace_row",
        budget=engine.csr.nnz + 1,
        edit={"operation": "replace_row", "row": target_row, "block_id": target_block},
    )
    timeline.extend(replace_reports)
    dependency_after = engine.dependencies[target_block]
    narrow_scope = {
        "target_block": target_block,
        "recomputed": list(replace_first.recomputed),
        "reused": list(replace_first.reused),
        "a_rows_before": list(dependency_before.a_rows),
        "a_rows_after": list(dependency_after.a_rows),
        "b_rows_before": list(dependency_before.b_rows),
        "b_rows_after": list(dependency_after.b_rows),
        "b_row_set_changed": dependency_before.b_rows != dependency_after.b_rows,
        "all_other_blocks_reused": set(replace_first.reused) == set(engine.current_block_ids) - {target_block},
    }

    engine.replace_row(target_row, [2, 8, 15], [-1.1, 0.8, 0.6])
    zero_budget_reports, zero_budget = _run_and_drain(engine, "replace_row_zero_budget", budget=0)
    timeline.extend(zero_budget_reports)
    pending_probe = {
        "pending": list(zero_budget.pending),
        "codes": [item.code for item in zero_budget.failures],
        "deductions": zero_budget.budget_deductions,
    }
    store.save(engine)

    engine.delete_row(2)
    delete_reports, delete_first = _run_and_drain(engine, "delete_row", edit={"row": 2})
    timeline.extend(delete_reports)

    engine.append_row(
        fixture.append_row_columns,
        fixture.append_row_values,
        c_init_row=fixture.append_c_row,
    )
    append_reports, append_first = _run_and_drain(
        engine,
        "append_row",
        edit={
            "columns": list(fixture.append_row_columns),
            "values": list(fixture.append_row_values),
        },
    )
    timeline.extend(append_reports)

    next_block_candidate = next(
        candidate for candidate in fixture.block_size_candidates
        if candidate[0] != engine.config.row_block_size
    )
    engine.set_block_size(next_block_candidate[0], next_block_candidate[1])
    block_reports, block_first = _run_and_drain(
        engine,
        "change_block_size",
        edit={"row_block_size": next_block_candidate[0], "nnz_chunk_size": next_block_candidate[1]},
    )
    timeline.extend(block_reports)

    engine.set_alpha_beta(engine.config.alpha * 1.2 + 0.03, engine.config.beta + 0.08)
    alpha_reports, alpha_first = _run_and_drain(
        engine,
        "change_alpha_beta",
        edit={"alpha": engine.config.alpha, "beta": engine.config.beta},
    )
    timeline.extend(alpha_reports)

    engine.set_precision(MODE_FIXED, 6, 12)
    precision_reports, precision_first = _run_and_drain(
        engine,
        "change_precision",
        edit={"mode": "fixed", "fractional_bits": 6, "accumulator_bits": 12},
    )
    timeline.extend(precision_reports)

    engine.set_order(ORDER_B_ABS_DESC, 3)
    order_reports, order_first = _run_and_drain(
        engine,
        "change_order",
        edit={"order": ORDER_B_ABS_DESC, "nnz_chunk_size": 3},
    )
    timeline.extend(order_reports)

    all_costs = [engine._block_cost(block_id) for block_id in engine.current_block_ids]
    prefix_budget = all_costs[0] + all_costs[1] if len(all_costs) > 1 else all_costs[0]
    engine.set_precision(MODE_FIXED, 5, 10)
    prefix_reports, prefix_first = _run_and_drain(
        engine,
        "prefix_budget_schedule",
        budget=prefix_budget,
        edit={"budget": prefix_budget, "accumulator_bits": 10},
    )
    timeline.extend(prefix_reports)

    store.save(engine)
    # Keep one more same-generation snapshot so stale-input recovery can prove
    # partial block reuse from the nearest valid version.
    store.save(engine)
    final_reference = full_rebuild(engine.csr, engine.b_matrix, engine.c_matrix, engine.config)
    consistency = compare_incremental_to_full(engine, final_reference)
    final_c = engine.materialize_c()
    np.save(output_path / "final_c.npy", final_c)

    fp32_reference = full_rebuild(
        engine.csr,
        engine.b_matrix,
        engine.c_matrix,
        engine.config.with_changes(
            precision_mode="fp32", fractional_bits=8, accumulator_bits=24
        ),
    )
    fixed_reference = full_rebuild(
        engine.csr,
        engine.b_matrix,
        engine.c_matrix,
        engine.config.with_changes(
            precision_mode=MODE_FIXED, fractional_bits=8, accumulator_bits=32
        ),
    )
    dual_path = compare_float_and_fixed(
        list(fp32_reference.blocks.values()), list(fixed_reference.blocks.values())
    )
    overflow_probe = {
        "precision": {
            "mode": MODE_FIXED,
            "fractional_bits": 5,
            "accumulator_bits": 10,
        },
        "overflow_records": fixed_reference.overflow_records,
        "overflow_block_count": len(
            [block for block in fixed_reference.blocks.values() if block.first_overflow is not None]
        ),
    }

    current_loaded, current_info = bootstrap_engine(
        store, engine.csr, engine.b_matrix, engine.c_matrix, engine.config
    )
    current_exact = bool(np.array_equal(current_loaded.materialize_c(), final_c))
    changed_csr = engine.csr.replace_row(
        min(3, engine.csr.rows - 1),
        [0, 2, 5, 12],
        [0.9, -1.3, 0.6, 1.1],
    )
    stale_engine, stale_info = bootstrap_engine(
        store, changed_csr, engine.b_matrix, engine.c_matrix, engine.config
    )
    if stale_engine.pending:
        stale_engine.run_until_complete()
    stale_reference = full_rebuild(
        changed_csr, engine.b_matrix, engine.c_matrix, engine.config
    )
    stale_consistency = compare_incremental_to_full(stale_engine, stale_reference)

    config_changed = engine.config.with_changes(max_workers=engine.config.max_workers + 1)
    config_stale_engine, config_stale_info = bootstrap_engine(
        store, engine.csr, engine.b_matrix, engine.c_matrix, config_changed
    )
    if config_stale_engine.pending:
        config_stale_engine.run_until_complete()
    config_stale_reference = full_rebuild(
        engine.csr, engine.b_matrix, engine.c_matrix, config_changed
    )
    config_stale_consistency = compare_incremental_to_full(
        config_stale_engine, config_stale_reference
    )

    with tempfile.TemporaryDirectory(prefix="spmm-snapshot-corrupt-") as temp_dir:
        corrupt_root = Path(temp_dir) / "snapshots"
        _copy_store(store.root, corrupt_root)
        corrupt_store = SnapshotStore(corrupt_root)
        state_path = corrupt_store.current / "state.json"
        corrupt_state = json.loads(state_path.read_text(encoding="utf-8"))
        corrupt_state["edit_version"] = -1
        state_path.write_text(json.dumps(corrupt_state), encoding="utf-8")
        recovered, corrupt_info = bootstrap_engine(
            corrupt_store, engine.csr, engine.b_matrix, engine.c_matrix, engine.config
        )
        corrupt_recovered = bool(
            corrupt_info["source"] == "previous_version"
            and corrupt_info["current_error"]["code"] == "E_SNAPSHOT_CORRUPT"
        )
        if recovered.pending:
            recovered.run_until_complete()
        corrupt_consistency = compare_incremental_to_full(recovered, final_reference)

    concurrency = _concurrency_report(fixture)
    injections = run_injection_suite(
        engine.csr,
        engine.b_matrix,
        engine.c_matrix,
        engine.config,
        list(fixture.injections),
    )

    candidate_report = {
        "row_block_and_nnz_chunk_candidates": [list(item) for item in fixture.block_size_candidates],
        "precision_candidates": list(fixture.precision_candidates),
        "chosen_config": fixture.config.to_dict(),
        "single_thread_config": fixture.single_thread_config.to_dict(),
        "seed": seed,
    }
    hot_diagnostics = {
        "capacity": engine.config.hot_capacity,
        "entries": engine.hot.to_dict()["entries"],
        "row_replace_scope": narrow_scope,
        "delete_recomputed": list(delete_first.recomputed),
        "append_recomputed": list(append_first.recomputed),
        "block_size_change_recomputed": list(block_first.recomputed),
        "alpha_change_recomputed": list(alpha_first.recomputed),
        "precision_change_recomputed": list(precision_first.recomputed),
        "order_change_recomputed": list(order_first.recomputed),
        "prefix_budget_recomputed": list(prefix_first.recomputed),
    }
    audit_replay = replay_audit(engine.audit_manifest(), engine.config.hot_capacity)
    audit_replay["hot_table_exact"] = audit_replay["hot_table"] == engine.hot.to_dict()
    budget_diagnostics = {
        "zero_budget_probe": pending_probe,
        "prefix_budget_probe": {
            "budget": prefix_budget,
            "targets": list(prefix_first.targets),
            "deductions": prefix_first.budget_deductions,
            "recomputed": list(prefix_first.recomputed),
            "pending": list(prefix_first.pending),
        },
        "cumulative_deduction_count": len(engine.budget_ledger),
        "final_pending": sorted(engine.pending),
    }
    initial_counts = np.diff(fixture.csr.row_offsets)
    assertions = {
        "csr_all_distributions_valid": all(
            item["valid"] for item in _distribution_report(fixture).values()
        ),
        "csr_columns_strictly_sorted": all(
            np.all(np.diff(fixture.csr.row_columns(row)) > 0)
            for row in range(fixture.csr.rows)
            if fixture.csr.row_nnz(row) > 1
        ),
        "row_replace_scope_exact": bool(narrow_scope["all_other_blocks_reused"]),
        "zero_budget_emits_pending": bool(zero_budget.pending)
        and any(item.code == "E_RECOMPUTE_PENDING" for item in zero_budget.failures),
        "prefix_budget_is_stable_prefix": [event["block_id"] for event in prefix_first.budget_deductions]
        == list(range(len(prefix_first.budget_deductions))),
        "history_audit_hot_replay_exact": bool(audit_replay["hot_table_exact"]),
        "budget_sequence_arithmetic_valid": bool(
            audit_replay["budget_sequence_valid"]
            and audit_replay["budget_arithmetic_valid"]
            and audit_replay["budget_block_order_stable"]
        ),
        "incremental_full_exact": bool(consistency["all_exact"]),
        "concurrent_single_exact": bool(
            concurrency["numeric_exact"]
            and concurrency["hot_table_exact"]
            and concurrency["budget_ledger_exact"]
            and not concurrency["report_mismatch_indices"]
        ),
        "dual_path_comparable": bool(dual_path["within_tolerance"]),
        "snapshot_current_valid": bool(current_info["source"] == "current" and current_exact),
        "snapshot_stale_detected": bool(
            stale_info["current_error"]["code"] == "E_SNAPSHOT_STALE"
            and stale_info["source"] == "previous_version"
        ),
        "snapshot_config_stale_detected": bool(
            config_stale_info["current_error"]["code"] == "E_SNAPSHOT_STALE"
        ),
        "snapshot_stale_rebuild_exact": bool(stale_consistency["all_exact"]),
        "snapshot_config_rebuild_exact": bool(config_stale_consistency["all_exact"]),
        "snapshot_corruption_recovered": bool(
            corrupt_recovered and corrupt_consistency["all_exact"]
        ),
        "injections_all_detected": bool(injections["all_injections_detected"]),
        "injections_isolated": all(injections["isolation"].values()),
    }
    report = {
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "temp_dir": tempfile.gettempdir(),
            "temp_writable": os.access(tempfile.gettempdir(), os.W_OK),
        },
        "candidate_generation": candidate_report,
        "distributions": _distribution_report(fixture),
        "fixture": {
            "shape": [fixture.csr.rows, fixture.csr.cols, fixture.b_matrix.shape[1]],
            "nnz": fixture.csr.nnz,
            "empty_rows": int(np.sum(initial_counts == 0)),
            "min_row_nnz": int(initial_counts.min(initial=0)),
            "max_row_nnz": int(initial_counts.max(initial=0)),
            "csr_fingerprint": fixture.csr.fingerprint(),
            "b_fingerprint": dense_fingerprint(engine.b_matrix),
            "final_c_fingerprint": engine.output_fingerprint(),
            "final_c_shape": list(final_c.shape),
        },
        "timeline": timeline,
        "hot_block_diagnostics": hot_diagnostics,
        "budget_diagnostics": budget_diagnostics,
        "history_audit_replay": audit_replay,
        "incremental_full_consistency": consistency,
        "dual_path_comparison": dual_path,
        "overflow_probe": overflow_probe,
        "concurrency": concurrency,
        "snapshots": {
            "root": str(snapshot_path),
            "current": {
                "source": current_info["source"],
                "version": current_info["snapshot_version"],
                "completed_blocks": current_info["completed_blocks"],
                "exact": current_exact,
            },
            "stale_input": {
                "source": stale_info["source"],
                "current_error": stale_info["current_error"],
                "completed_blocks": stale_info["completed_blocks"],
                "rejected_blocks": stale_info["rejected_blocks"],
                "numeric_exact_after_rebuild": bool(stale_consistency["all_exact"]),
            },
            "stale_config": {
                "source": config_stale_info["source"],
                "current_error": config_stale_info["current_error"],
                "numeric_exact_after_rebuild": bool(config_stale_consistency["all_exact"]),
            },
            "corruption": {
                "source": corrupt_info["source"],
                "current_error": corrupt_info["current_error"],
                "recovered": corrupt_recovered,
                "numeric_exact_after_recovery": bool(corrupt_consistency["all_exact"]),
            },
        },
        "fault_injections": injections,
        "assertions": assertions,
        "all_acceptance_checks_passed": all(assertions.values()),
    }
    (output_path / "demo_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    (output_path / "diagnostic_report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    fixture = report["fixture"]
    consistency = report["incremental_full_consistency"]
    concurrency = report["concurrency"]
    stale = report["snapshots"]["stale_input"]
    lines = [
        "# SpMM 增量诊断运行报告",
        "",
        "## 环境与样本",
        "",
        f"- Python {report['environment']['python']} / NumPy {report['environment']['numpy']}",
        f"- 形状 M×K×N: {fixture['shape'][0]}×{fixture['shape'][1]}×{fixture['shape'][2]}",
        f"- NNZ: {fixture['nnz']}，空行: {fixture['empty_rows']}，单行最大 NNZ: {fixture['max_row_nnz']}",
        f"- 最终输出指纹: `{fixture['final_c_fingerprint']}`",
        "",
        "## 验收结论",
        "",
        f"- 增量 vs 全量逐元素一致: **{consistency['numeric_exact']}**",
        f"- 行结束指纹/溢出/块指纹/依赖图全一致: **{consistency['all_exact']}**",
        f"- 并发 vs 单线程报告一致: **{not concurrency['report_mismatch_indices']}**",
        f"- 双路径可比较: **{report['dual_path_comparison']['within_tolerance']}**，最大绝对误差 {report['dual_path_comparison']['max_abs_error']:.6g}",
        f"- 行替换仅重算目标块: **{report['hot_block_diagnostics']['row_replace_scope']['all_other_blocks_reused']}**",
        f"- 零预算产生 E_RECOMPUTE_PENDING: **{any(code == 'E_RECOMPUTE_PENDING' for code in report['budget_diagnostics']['zero_budget_probe']['codes'])}**",
        f"- 输入变化快照判定 E_SNAPSHOT_STALE 并回退: **{stale['current_error']['code'] == 'E_SNAPSHOT_STALE'}**",
        f"- 快照回退后重建一致: **{stale['numeric_exact_after_rebuild']}**",
        f"- 四类故障注入全部命中: **{report['fault_injections']['all_injections_detected']}**",
        "",
        "## 热块与预算",
        "",
        f"- 热块容量: {report['hot_block_diagnostics']['capacity']}",
        f"- 行替换失效块: {report['hot_block_diagnostics']['row_replace_scope']['target_block']}",
        f"- 同批复用块: {report['hot_block_diagnostics']['row_replace_scope']['reused']}",
        f"- 零预算待办: {report['budget_diagnostics']['zero_budget_probe']['pending']}",
        f"- 前缀预算扣减: {[item['block_id'] for item in report['budget_diagnostics']['prefix_budget_probe']['deductions']]}",
        "",
        "## 故障注入",
        "",
    ]
    for item in report["fault_injections"]["results"]:
        lines.append(
            f"- `{item['injection_id']}` ({item['kind']}): {', '.join(item['failure_codes'])}，路径 `{item['trace']['path']}`"
        )
    lines.extend(
        [
            "",
            "## 一致性口径",
            "",
            "C 元素、行结束指纹、溢出记录、块产出指纹与依赖图均与最终状态一次性全量重建逐位一致。"
            "热块失效次数与预算扣减序列依赖编辑/调度历史；历史无关的一次性最终状态无法反推出这些计数，"
            "因此用同一编辑/预算脚本的单线程独立重放做逐项一致比对。",
            "",
            f"总体验收：**{report['all_acceptance_checks_passed']}**",
            "",
        ]
    )
    return "\n".join(lines)
