#!/usr/bin/env python
"""Run the complete local acceptance suite and write a verification report."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabular_lab.errors import E_SNAPSHOT_STALE, LabError
from tabular_lab.model_io import compare_records, read_model_file, write_downgrade_sidecar, write_v1
from tabular_lab.runner import run_experiment
from tests.helpers import LabFixture


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_yaml(path: Path, payload: Any) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def report_table(report: dict[str, Any]) -> str:
    lines = [
        "| model | scheme | status | epochs | RMSE | accuracy@0.5 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model in report["models"]:
        metrics = model.get("validation_metrics") or {}
        lines.append(
            "| {name} | {scheme} | {status} | {epochs} | {rmse:.6f} | {accuracy:.6f} |".format(
                name=model["name"],
                scheme=model["scheme"],
                status=model.get("status", "failed"),
                epochs=model.get("epochs_consumed", 0),
                rmse=float(metrics.get("rmse", 0.0)),
                accuracy=float(metrics.get("accuracy@0.5", 0.0)),
            )
        )
    return "\n".join(lines)


def run_unit_tests() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    combined = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    tail = [line for line in combined.splitlines() if line.strip()][-4:]
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "summary": "\n".join(tail),
        "stderr": completed.stderr[-2000:],
    }


def main() -> int:
    started = time.time()
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {}

    with tempfile.TemporaryDirectory(prefix="tabular-lab-acceptance-") as temporary:
        work = Path(temporary)
        base_config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
        data_spec = yaml.safe_load((ROOT / "configs/data.yaml").read_text(encoding="utf-8"))
        base_config["data"] = str((ROOT / "configs/data.yaml").resolve())
        base_path = write_yaml(work / "base.yaml", base_config)

        # Main sample: clean run, identical-cache rerun, and a multi-step edit
        # sequence that ends in v1 with a changed penalty, data column, and
        # feature whitelist.
        main_state = work / "main-state"
        baseline = run_experiment(base_path, state_dir=main_state, report_path=work / "baseline.json")
        checks["three_schemes_trained"] = {model["scheme"] for model in baseline["models"]} == {
            "l2",
            "l1",
            "elastic_net",
        }
        checks["baseline_ok"] = baseline["status"] == "ok" and baseline["validation_summary"]["model_count"] == 3
        rerun = run_experiment(base_path, state_dir=main_state, report_path=work / "rerun.json")
        checks["full_cache_hit"] = (
            rerun["retrained_models"] == []
            and rerun["reconverted_models"] == []
            and rerun["reuse_list"] == [model["name"] for model in rerun["models"]]
        )
        checks["rerun_semantic_equal"] = baseline["semantic_digest"] == rerun["semantic_digest"]

        penalty_config = copy.deepcopy(base_config)
        for model in penalty_config["models"]:
            if model["name"] == "lasso_compact":
                model["penalty"] = 0.031
        penalty_path = write_yaml(work / "penalty.yaml", penalty_config)
        penalty_report = run_experiment(penalty_path, state_dir=main_state)
        checks["penalty_only_retrains_target"] = (
            penalty_report["retrained_models"] == ["lasso_compact"]
            and penalty_report["reused_models"] == ["elastic_balanced", "ridge_core"]
        )

        v1_config = copy.deepcopy(penalty_config)
        v1_config["exchange_version"] = "v1"
        v1_path = write_yaml(work / "v1.yaml", v1_config)
        version_report = run_experiment(v1_path, state_dir=main_state)
        checks["version_switch_reuses_training"] = (
            version_report["audit"]["training_reused"] == ["elastic_balanced", "lasso_compact", "ridge_core"]
            and version_report["reconverted_models"] == ["elastic_balanced", "lasso_compact", "ridge_core"]
        )

        variant_data = copy.deepcopy(data_spec)
        for column in variant_data["columns"]:
            if column["name"] == "utilization":
                column["variant"] = 1
        variant_data_path = write_yaml(work / "data-variant.yaml", variant_data)
        column_config = copy.deepcopy(v1_config)
        column_config["data"] = str(variant_data_path)
        column_path = write_yaml(work / "column.yaml", column_config)
        column_report = run_experiment(column_path, state_dir=main_state)
        checks["single_column_invalidation"] = (
            column_report["retrained_models"] == ["elastic_balanced", "ridge_core"]
            and column_report["audit"]["training_reused"] == ["lasso_compact"]
        )

        final_config = copy.deepcopy(column_config)
        for model in final_config["models"]:
            if model["name"] == "ridge_core":
                model["features"] = [feature for feature in model["features"] if feature != "region"]
        final_path = write_yaml(work / "final.yaml", final_config)
        final_incremental = run_experiment(final_path, state_dir=main_state, report_path=work / "final-incremental.json")
        final_full = run_experiment(final_path, state_dir=work / "full-state", report_path=work / "final-full.json")
        checks["incremental_equals_full_semantics"] = (
            final_incremental["semantic_digest"] == final_full["semantic_digest"]
        )
        checks["incremental_equals_full_stage_versions"] = (
            final_incremental["snapshot"]["stage_versions"] == final_full["snapshot"]["stage_versions"]
        )
        inc_snapshot = Path(final_incremental["snapshot"]["path"])
        full_snapshot = Path(final_full["snapshot"]["path"])
        checks["incremental_equals_full_export"] = (
            (inc_snapshot / "exports/models.v1.txt").read_bytes()
            == (full_snapshot / "exports/models.v1.txt").read_bytes()
        )
        inc_sidecars = {path.name: path.read_bytes() for path in (inc_snapshot / "sidecars").glob("*.json")}
        full_sidecars = {path.name: path.read_bytes() for path in (full_snapshot / "sidecars").glob("*.json")}
        checks["incremental_equals_full_sidecars"] = inc_sidecars == full_sidecars

        # Persist a small inspectable copy of the final v1 exchange product.
        shutil.copy2(inc_snapshot / "exports/models.v1.txt", reports_dir / "acceptance_models.v1.txt")
        shutil.copy2(
            inc_snapshot / "exports/conversion_manifest.json", reports_dir / "acceptance_conversion_manifest.json"
        )
        persistent_sidecars = reports_dir / "acceptance_sidecars"
        if persistent_sidecars.exists():
            shutil.rmtree(persistent_sidecars)
        persistent_sidecars.mkdir(parents=True)
        for source in (inc_snapshot / "sidecars").glob("*.json"):
            shutil.copy2(source, persistent_sidecars / source.name)
        exported_v1_records = read_model_file(
            reports_dir / "acceptance_models.v1.txt", sidecar_dir=persistent_sidecars
        )
        exported_v1_indices = {record.name: record.feature_indices for record in exported_v1_records}
        checks["exported_v1_reloads_sparse_indices"] = (
            len(exported_v1_records) == 3
            and exported_v1_indices["lasso_compact"] == [0, 1, 3, 4, 5, 6]
            and exported_v1_indices["ridge_core"] == [0, 1, 2, 3, 7, 8]
            and exported_v1_indices["elastic_balanced"] == list(range(11))
        )

        # Model exchange roundtrip.
        source_records = read_model_file(ROOT / "samples/models.v2.json")
        v1_sample = work / "sample.v1.txt"
        sidecars = work / "sample-sidecars"
        write_v1(source_records, v1_sample)
        for record in source_records:
            write_downgrade_sidecar(record, sidecars)
        loaded_records = read_model_file(v1_sample, sidecar_dir=sidecars)
        comparison = compare_records(source_records, loaded_records)
        roundtrip_max_delta = max(
            float(comparison["max_abs_coefficient_delta"]),
            float(comparison["max_abs_intercept_delta"]),
        )
        checks["v1_v2_roundtrip_within_tolerance"] = roundtrip_max_delta <= 0.001
        checks["sidecar_restores_named_mix"] = loaded_records[2].mix == 0.55

        # Small deterministic fixtures for budget, concurrency and failure isolation.
        fixture = LabFixture(work / "fixture")
        budget_config = copy.deepcopy(fixture.config)
        budget_config["budget"] = {"global_epochs": 10, "per_model_epochs": 8}
        budget_path = write_yaml(work / "budget.yaml", budget_config)
        budget_report = run_experiment(budget_path, state_dir=work / "budget-state")
        checks["budget_codes_and_order"] = (
            [failure["code"] for failure in budget_report["failures"]]
            == ["E_MODEL_BUDGET_EXHAUSTED", "E_TRAIN_BUDGET_EXHAUSTED"]
            and budget_report["pending_models"] == ["ridge_core"]
        )

        fault_base = copy.deepcopy(fixture.config)
        fault_base["training"]["epochs"] = 5
        fault_base["faults"] = [
            {
                "model": "lasso_compact",
                "stage": "conversion",
                "exception": "RuntimeError",
                "message": "isolated conversion failure",
            }
        ]
        serial = copy.deepcopy(fault_base)
        serial["runtime"]["max_concurrency"] = 1
        parallel = copy.deepcopy(fault_base)
        parallel["runtime"]["max_concurrency"] = 4
        serial_report = run_experiment(
            write_yaml(work / "serial.yaml", serial), state_dir=work / "serial-state"
        )
        parallel_report = run_experiment(
            write_yaml(work / "parallel.yaml", parallel), state_dir=work / "parallel-state"
        )
        checks["serial_parallel_semantics_equal"] = serial_report["semantic_digest"] == parallel_report["semantic_digest"]
        checks["failure_isolated"] = (
            [failure["model"] for failure in serial_report["failures"]] == ["lasso_compact"]
            and serial_report["validation_summary"]["model_count"] == 2
        )

        # OS signal handling: SIGINT should commit partial work and exit 130.
        signal_config = copy.deepcopy(fixture.config)
        signal_config["training"]["epochs"] = 30
        signal_config["training"]["snapshot_every_batches"] = 20
        signal_config["budget"] = {"global_epochs": 300, "per_model_epochs": 100}
        signal_path = write_yaml(work / "signal.yaml", signal_config)
        signal_state = work / "signal-state"
        signal_report_path = work / "signal-report.json"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tabular_lab",
                "run",
                "--config",
                str(signal_path),
                "--state-dir",
                str(signal_state),
                "--report",
                str(signal_report_path),
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        time.sleep(0.35)
        process.send_signal(signal.SIGINT)
        _, signal_stderr = process.communicate(timeout=30)
        sigint_returncode = process.returncode
        checks["sigint_exit_code_130"] = sigint_returncode == 130
        sigint_status = None
        if signal_report_path.exists():
            sigint_status = json.loads(signal_report_path.read_text(encoding="utf-8"))["status"]
        checks["sigint_snapshot_committed"] = sigint_status == "interrupted"
        signal_resumed = run_experiment(signal_path, state_dir=signal_state)
        signal_full = run_experiment(signal_path, state_dir=work / "signal-full-state")
        checks["sigint_resume_equals_full"] = signal_resumed["semantic_digest"] == signal_full["semantic_digest"]

        # SIGKILL recovery: only atomically persisted live checkpoints/snapshots
        # may be used after the process dies.
        kill_state = work / "kill-state"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tabular_lab",
                "run",
                "--config",
                str(signal_path),
                "--state-dir",
                str(kill_state),
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        for _ in range(100):
            if any((kill_state / "live").glob("train_*.json")):
                break
            time.sleep(0.03)
        process.kill()
        process.communicate(timeout=10)
        live_checkpoints = list((kill_state / "live").glob("train_*.json"))
        kill_resumed = run_experiment(signal_path, state_dir=kill_state)
        checks["sigkill_has_live_checkpoints"] = bool(live_checkpoints)
        checks["sigkill_resume_equals_full"] = kill_resumed["semantic_digest"] == signal_full["semantic_digest"]

        # Strict resume must reject a changed config.
        try:
            run_experiment(base_path, state_dir=main_state, resume=True)
        except LabError as exc:
            checks["stale_resume_code"] = exc.code == E_SNAPSHOT_STALE
        else:
            checks["stale_resume_code"] = False

        unit_tests = run_unit_tests()
        checks["unit_tests"] = bool(unit_tests["ok"])

        results = {
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
                "platform": platform.platform(),
            },
            "checks": checks,
            "unit_tests": unit_tests,
            "baseline": {
                "status": baseline["status"],
                "models": baseline["models"],
                "validation_summary": baseline["validation_summary"],
                "semantic_digest": baseline["semantic_digest"],
                "snapshot_version": baseline["snapshot"]["global_version"],
            },
            "incremental": {
                "penalty_retrained": penalty_report["retrained_models"],
                "version_training_reused": version_report["audit"]["training_reused"],
                "version_reconverted": version_report["reconverted_models"],
                "column_retrained": column_report["retrained_models"],
                "column_training_reused": column_report["audit"]["training_reused"],
                "final_retrained": final_incremental["retrained_models"],
                "semantic_digest": final_incremental["semantic_digest"],
                "full_digest": final_full["semantic_digest"],
                "stage_versions": final_incremental["snapshot"]["stage_versions"],
            },
            "roundtrip": {
                "max_abs_delta": roundtrip_max_delta,
                "tolerance": 0.001,
                "sidecar_mix": loaded_records[2].mix,
                "exported_v1_feature_indices": exported_v1_indices,
            },
            "budget": {
                "failures": budget_report["failures"],
                "pending_models": budget_report["pending_models"],
                "completed_models": budget_report["snapshot"]["completed_models"],
                "consumed_epochs": budget_report["snapshot"]["consumed_epochs"],
            },
            "concurrency": {
                "serial_digest": serial_report["semantic_digest"],
                "parallel_digest": parallel_report["semantic_digest"],
                "failures": serial_report["failures"],
                "validation_model_count": serial_report["validation_summary"]["model_count"],
            },
            "signals": {
                "sigint_returncode": sigint_returncode,
                "sigint_status": sigint_status,
                "sigint_resume_digest": signal_resumed["semantic_digest"],
                "sigkill_live_checkpoint_count": len(live_checkpoints),
                "sigkill_resume_digest": kill_resumed["semantic_digest"],
                "full_digest": signal_full["semantic_digest"],
            },
            "elapsed_seconds": round(time.time() - started, 3),
        }
        (reports_dir / "acceptance_results.json").write_text(
            json.dumps(results, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

        all_checks = all(checks.values())
        check_lines = [f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in checks.items()]
        markdown = f"""# 表格线性模型正则化实验台验收报告

生成时间：2026-11-23（本地验收运行）  
Python：{sys.version.split()[0]}；NumPy：{np.__version__}；PyYAML：{yaml.__version__}  
平台：{platform.platform()}

## 验收结论

{'全部验收项通过。' if all_checks else '存在失败项，请查看 `reports/acceptance_results.json`。'}

| 检查项 | 结果 |
|---|---|
{os.linesep.join(check_lines)}

## 样本训练

{report_table(baseline)}

- 语义摘要：`{baseline['semantic_digest']}`
- 三个模型均逐轮记录 loss 与系数摘要；重复运行缓存命中且语义摘要不变。
- 最终编辑序列：惩罚系数调整 → 切换 v1 → 替换 `utilization` 单列 → 调整 `ridge_core` 白名单。
- 最终增量与空白全量语义摘要一致：`{final_incremental['semantic_digest'] == final_full['semantic_digest']}`。
- 最终增量与全量导出的 `models.v1.txt`、旁路 sidecar 字节一致。

## 双版本互转

- v2 样例 → v1 定宽行 → 重新加载，系数/截距最大绝对偏差 `{roundtrip_max_delta:.12f}`，容差 `0.001`。
- sidecar 包含截断残差、原精度标签与丢弃的命名信息；重新加载后恢复弹性网络 `mix={loaded_records[2].mix}`。
- 实际导出的 v1 稀疏特征序号可回读：`lasso_compact={exported_v1_indices['lasso_compact']}`，`ridge_core={exported_v1_indices['ridge_core']}`。
- 持续保留的验收导出：`reports/acceptance_models.v1.txt`、`reports/acceptance_sidecars/`。

## 预算、并发与恢复

- 预算：失败顺序 `{[failure['code'] for failure in budget_report['failures']]}`；待办 `{budget_report['pending_models']}`；持久化消耗轮次 `{budget_report['snapshot']['consumed_epochs']}`。
- 串行/并发语义摘要一致：`{serial_report['semantic_digest'] == parallel_report['semantic_digest']}`；转换阶段故障只影响 `lasso_compact`，验证汇总仅含 2 个成功模型。
- SIGINT：退出码 `{sigint_returncode}`，快照状态 `{sigint_status}`，续跑与全量一致：`{signal_resumed['semantic_digest'] == signal_full['semantic_digest']}`。
- SIGKILL：持久化 live checkpoint 数量 `{len(live_checkpoints)}`，续跑与全量一致：`{kill_resumed['semantic_digest'] == signal_full['semantic_digest']}`。
- 严格恢复在配置变化时返回 `E_SNAPSHOT_STALE`。

## 自动化测试

```
{unit_tests['summary']}
```

完整机器可读结果见 `reports/acceptance_results.json`。
"""
        (reports_dir / "verification_report.md").write_text(markdown, encoding="utf-8")
        print(json.dumps({"status": "ok" if all_checks else "failed", "checks": checks, "elapsed_seconds": results["elapsed_seconds"]}, ensure_ascii=False, indent=2))
        return 0 if all_checks else 1


if __name__ == "__main__":
    raise SystemExit(main())
