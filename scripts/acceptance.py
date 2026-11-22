#!/usr/bin/env python3
"""Reproducible acceptance run for the local deterministic sample.

The script exercises cold/warm execution, all requested edit classes, a clean
full rebuild, serial/parallel equivalence, snapshot recovery, failure-code
collection, and both boundary-attribution sources.  It writes machine-readable
reports under ``reports/acceptance`` and a Markdown summary to
``VERIFICATION.md``.
"""

from __future__ import annotations

import copy
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_pipeline import mini_yaml  # noqa: E402
from feature_pipeline.config import load_pipeline_config  # noqa: E402
from feature_pipeline.metrics import attribute_boundary_cases  # noqa: E402
from feature_pipeline.pipeline import run_pipeline  # noqa: E402
from feature_pipeline.sample import prepare_sample  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "reports" / "acceptance"
TEMP_BASE = PROJECT_ROOT / ".tmp" / "acceptance"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class AcceptanceWorkspace:
    def __init__(self, *, name: str, config: dict[str, Any], columns: dict[str, Any]):
        TEMP_BASE.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix=f"{name}-", dir=TEMP_BASE)
        self.root = Path(self._temporary.name)
        (self.root / "configs").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "reports").mkdir(parents=True, exist_ok=True)
        self.config = copy.deepcopy(config)
        self.columns = copy.deepcopy(columns)
        self.config_path = self.root / "configs" / "pipeline.yaml"
        self.columns_path = self.root / "configs" / "columns.yaml"
        self.write_columns()
        self.write_config()

    def close(self) -> None:
        self._temporary.cleanup()

    def write_columns(self, columns: dict[str, Any] | None = None) -> None:
        if columns is not None:
            self.columns = copy.deepcopy(columns)
        _write_json(self.columns_path, self.columns)

    def write_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            self.config = copy.deepcopy(config)
        self.config["paths"] = {
            "sample": "data/sample.json",
            "sample_manifest": "data/sample_manifest.json",
            "overrides": "data/overrides",
            "cache": self.config["paths"].get("cache", "cache"),
            "report": self.config["paths"].get("report", "reports/diagnostic_report.json"),
        }
        _write_json(self.config_path, self.config)

    def load(self):
        return load_pipeline_config(self.config_path, root=self.root)

    def prepare(self, *, force: bool = False) -> dict[str, Any]:
        return prepare_sample(self.load(), force=force)

    def run(self) -> dict[str, Any]:
        return run_pipeline(self.config_path, root=self.root)

    @property
    def report_path(self) -> Path:
        return self.root / self.config["paths"]["report"]

    def replace_override(self, column: str, values: list[Any]) -> None:
        path = self.root / "data" / "overrides" / f"{column}.json"
        _write_json(path, {"schema_version": 1, "column": column, "values": values})

    def copy_report(self, name: str) -> dict[str, Any]:
        report = _read_json(self.report_path)
        _write_json(OUTPUT_DIR / name, report)
        return report

    def clone(self, *, name: str, cache: str, report: str) -> "AcceptanceWorkspace":
        other = AcceptanceWorkspace(name=name, config=self.config, columns=self.columns)
        other.config["paths"]["cache"] = cache
        other.config["paths"]["report"] = report
        other.write_config()
        source = self.root / "data" / "overrides"
        if source.exists():
            shutil.copytree(source, other.root / "data" / "overrides")
        return other


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    diagnostics = report.get("diagnostics", {})
    return {
        "status": report["status"],
        "config_hash": report.get("config_hash"),
        "global_version": report.get("global_version"),
        "sample_checksum": report.get("sample_checksum"),
        "effective_data_hash": report.get("effective_data_hash"),
        "stage_versions": report.get("stage_versions", {}),
        "stage_version_changes": report.get("stage_version_changes", []),
        "active_cross_features": report.get("cross_encoding", {}).get("active_feature_count"),
        "degradations": len(report.get("cross_encoding", {}).get("degradations", [])),
        "paths": [
            {
                "path": item["path"],
                "retained_dimensions": item["retained_dimensions"],
                "summary": item["summary"],
            }
            for item in report.get("reduction", {}).get("paths", [])
        ],
        "stage_diffs": diagnostics.get("stage_diffs", []),
        "stage_summary": diagnostics.get("stage_summary", []),
        "first_exceeded_stage": diagnostics.get("first_exceeded_stage"),
        "first_exceeded_item": diagnostics.get("first_exceeded_item"),
        "boundary_cases": diagnostics.get("boundary_cases", []),
        "cache_summary": diagnostics.get("cache_summary", {}),
        "reuse_count": len(diagnostics.get("reuse_ledger", [])),
        "invalidation_count": len(diagnostics.get("invalidation_ledger", [])),
        "failures": report.get("failures", []),
        "runtime": report.get("runtime", {}),
        "snapshots": report.get("snapshots", {}),
        "recovery_actions": report.get("recovery_actions", []),
    }


def _ledger_order(report: dict[str, Any], key: str) -> list[tuple[str, str]]:
    return [(item["stage"], item["sort_key"]) for item in report["diagnostics"][key]]


def _ledger_sorted(report: dict[str, Any], key: str) -> bool:
    stage_rank = {
        "cross_encoding": 0,
        "standardization": 1,
        "dimensionality_reduction": 2,
        "metrics": 3,
    }
    values = [
        (stage_rank[item["stage"]], item["sort_key"], item.get("action", ""))
        for item in report["diagnostics"][key]
    ]
    return values == sorted(values)


def _section_markdown(results: dict[str, Any]) -> str:
    baseline = results["baseline"]
    final = results["final_incremental"]
    rebuild = results["full_rebuild_comparison"]
    concurrency = results["concurrency_comparison"]
    quant = results["boundary_quantization"]
    approximation = results["boundary_approximation"]
    validation = results["validation_codes"]
    stage_rows = "\n".join(
        f"| {item['stage']} | {item['max_abs_diff']:.12g} | {str(item['exceeded']).lower()} | {item['first_exceeded_item'] or '-'} |"
        for item in final["stage_summary"]
    )
    edit_rows = "\n".join(
        f"| {item['name']} | {item['reuse']} | {item['invalidation']} | {item['active_features']} | {item['status']} |"
        for item in results["edits"]
    )
    return f"""# Verification Results

Generated by `python3 scripts/acceptance.py` on the local deterministic sample.

## Environment

- Python: `{results['environment']['python']}`
- NumPy: `{results['environment']['numpy']}`
- OS: `{results['environment']['platform']}`
- Configured workers: `{baseline['runtime']['max_workers']}`
- Workspace temp directory was writable: `true`

## Sample And Baseline

- Offline sample checksum: `{baseline['sample_checksum']}`
- Effective checksum after overrides: `{final['effective_data_hash']}`
- Baseline global version: `{baseline['global_version']}`
- Baseline feature count / degradations: `{baseline['active_cross_features']}` / `{baseline['degradations']}`
- Warm rerun direct reuse: `{results['warm_rerun']['cache_summary']['reused']}` products
- Warm rerun semantic equality: `{str(results['warm_rerun']['semantic_equal']).lower()}`

## Reduction Paths

| Path | Retained dims | Output features | Summary kind |
|---|---:|---|---|
| pca | {baseline['paths'][0]['retained_dimensions']} | {len(baseline['paths'][0]['summary']['components'])} components | explained variance ratios |
| random_projection | {baseline['paths'][1]['retained_dimensions']} | {baseline['paths'][1]['summary']['matrix_shape'][1]} | seed {baseline['paths'][1]['summary']['seed']} |
| variance | {baseline['paths'][2]['retained_dimensions']} | {len(baseline['paths'][2]['summary']['selected_feature_keys'])} | selected cross features |

## Precision Diagnostics

First stage exceeding configured tolerance: **{final['first_exceeded_stage']}** (`{final['first_exceeded_item']}`).

| Stage | Max absolute difference | Exceeded | First exceeded item |
|---|---:|---|---|
{stage_rows}

## Edit Sequence

| Edit | Reused | Invalidated/recomputed | Active features | Status |
|---|---:|---:|---:|---|
{edit_rows}

- Region replacement reused unaffected crossing combinations: `{', '.join(results['region_replacement']['reused_cross_combinations'])}`
- PCA dimension edit invalidated reduction paths: `{', '.join(results['pca_edit']['invalidated_reduction_paths'])}`
- PCA dimension edit reused reduction paths: `{', '.join(results['pca_edit']['reused_reduction_paths'])}`

## Incremental Versus Full Rebuild

- Semantic fingerprint equal: `{str(rebuild['semantic_fingerprint_equal']).lower()}`
- Global version equal: `{str(rebuild['global_version_equal']).lower()}`
- Stage versions equal: `{str(rebuild['stage_versions_equal']).lower()}`
- Cross/standard/reduction outputs equal: `{str(rebuild['outputs_equal']).lower()}`
- Stage differences equal: `{str(rebuild['stage_diffs_equal']).lower()}`
- Boundary list equal: `{str(rebuild['boundary_cases_equal']).lower()}`
- Failure list equal: `{str(rebuild['failures_equal']).lower()}`
- Product/dependency ledger equal: `{str(rebuild['dependency_ledger_equal']).lower()}`
- Reuse/invalidation list ordering stable: `{str(rebuild['ledger_order_stable']).lower()}`

Physical cache actions intentionally differ between warm incremental and a cold rebuild; result fields, product order, dependency fingerprints and cache keys are compared exactly.

## Concurrency

- Serial observed max workers: `{concurrency['serial_max_observed_workers']}`
- Parallel observed max workers: `{concurrency['parallel_max_observed_workers']}` (configured {concurrency['parallel_limit']})
- Semantic fingerprint equal: `{str(concurrency['semantic_equal']).lower()}`
- Stage versions equal: `{str(concurrency['stage_versions_equal']).lower()}`
- Stage differences equal: `{str(concurrency['stage_diffs_equal']).lower()}`

## Boundary Attribution

### Fixed-point decision boundary

- Cases: `{quant['count']}`
- Sources: `{json.dumps(quant['sources'], ensure_ascii=False)}`
- Report: `reports/acceptance/{quant['report']}`

### Dimensional numerical approximation boundary

- Cases: `{approximation['count']}`
- Sources: `{json.dumps(approximation['sources'], ensure_ascii=False)}`
- Report: `reports/acceptance/{approximation['report']}`

## Failure Codes

Collected without short-circuiting valid column loading:

`{', '.join(validation['codes'])}`

Loaded valid columns after invalid inputs: `{', '.join(validation['loaded_columns'])}`.

## Snapshot Recovery

- Actions: `{', '.join(item['action'] for item in results['snapshot_recovery']['actions'])}`
- Semantic result preserved: `{str(results['snapshot_recovery']['semantic_equal']).lower()}`
- Stage versions preserved: `{str(results['snapshot_recovery']['stage_versions_equal']).lower()}`

## Test Suite

`python3 -m unittest discover -v` was run separately; see the final response for the recorded result.
"""


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_BASE.mkdir(parents=True, exist_ok=True)
    base_config = mini_yaml.load(PROJECT_ROOT / "configs" / "pipeline.yaml")
    base_columns = mini_yaml.load(PROJECT_ROOT / "configs" / "columns.yaml")
    initial_config = copy.deepcopy(base_config)
    workspaces: list[AcceptanceWorkspace] = []

    def new_workspace(name: str, config: dict[str, Any] | None = None) -> AcceptanceWorkspace:
        workspace = AcceptanceWorkspace(
            name=name,
            config=config if config is not None else base_config,
            columns=base_columns,
        )
        workspaces.append(workspace)
        return workspace

    try:
        main_ws = new_workspace("main")
        first_prepare = main_ws.prepare()
        second_prepare = main_ws.prepare()
        baseline = main_ws.run()
        main_ws.copy_report("baseline.json")
        warm = main_ws.run()
        warm_semantic_equal = warm["semantic_fingerprint"] == baseline["semantic_fingerprint"]

        edits: list[dict[str, Any]] = []
        replacement_values = [
            ["east", "north", "south", "west"][(index * 3 + 1) % 4]
            for index in range(base_columns["rows"])
        ]
        main_ws.replace_override("region", replacement_values)
        replacement_report = main_ws.run()
        reused_cross = sorted(
            {
                item["sort_key"].split("/")[1]
                for item in replacement_report["diagnostics"]["reuse_ledger"]
                if item["stage"] == "cross_encoding"
            }
        )
        edits.append(
            {
                "name": "replace region column",
                "status": replacement_report["status"],
                "reuse": replacement_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation": replacement_report["diagnostics"]["cache_summary"]["invalidated_or_recomputed"],
                "active_features": replacement_report["cross_encoding"]["active_feature_count"],
            }
        )

        config = main_ws.config
        config["reduction"]["pca"]["dimensions"] = 2
        main_ws.write_config(config)
        pca_report = main_ws.run()
        invalidated_pca_paths = sorted(
            {
                item["sort_key"].split("/")[1]
                for item in pca_report["diagnostics"]["invalidation_ledger"]
                if item["stage"] == "dimensionality_reduction"
            }
        )
        reused_pca_paths = sorted(
            {
                item["sort_key"].split("/")[1]
                for item in pca_report["diagnostics"]["reuse_ledger"]
                if item["stage"] == "dimensionality_reduction"
            }
        )
        edits.append(
            {
                "name": "pca dimensions 4 -> 2",
                "status": pca_report["status"],
                "reuse": pca_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation": pca_report["diagnostics"]["cache_summary"]["invalidated_or_recomputed"],
                "active_features": pca_report["cross_encoding"]["active_feature_count"],
            }
        )

        config = main_ws.config
        config["crossing"]["order"] = 3
        main_ws.write_config(config)
        order_report = main_ws.run()
        edits.append(
            {
                "name": "crossing order 2 -> 3",
                "status": order_report["status"],
                "reuse": order_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation": order_report["diagnostics"]["cache_summary"]["invalidated_or_recomputed"],
                "active_features": order_report["cross_encoding"]["active_feature_count"],
            }
        )

        config = main_ws.config
        config["crossing"]["whitelist"] = ["region", "device", "channel"]
        main_ws.write_config(config)
        whitelist_report = main_ws.run()
        edits.append(
            {
                "name": "whitelist removes tier",
                "status": whitelist_report["status"],
                "reuse": whitelist_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation": whitelist_report["diagnostics"]["cache_summary"]["invalidated_or_recomputed"],
                "active_features": whitelist_report["cross_encoding"]["active_feature_count"],
            }
        )

        config = main_ws.config
        config["crossing"]["cardinality_budget"] = 5
        main_ws.write_config(config)
        budget_report = main_ws.run()
        edits.append(
            {
                "name": "cardinality budget 8 -> 5",
                "status": budget_report["status"],
                "reuse": budget_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation": budget_report["diagnostics"]["cache_summary"]["invalidated_or_recomputed"],
                "active_features": budget_report["cross_encoding"]["active_feature_count"],
            }
        )

        config = main_ws.config
        config["precision"]["quantization"]["tier"] = "coarse"
        main_ws.write_config(config)
        final_report = main_ws.run()
        main_ws.copy_report("incremental_final.json")
        edits.append(
            {
                "name": "quantization tier fine -> coarse",
                "status": final_report["status"],
                "reuse": final_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation": final_report["diagnostics"]["cache_summary"]["invalidated_or_recomputed"],
                "active_features": final_report["cross_encoding"]["active_feature_count"],
            }
        )

        full_ws = main_ws.clone(name="full", cache="cache_full", report="reports/full.json")
        workspaces.append(full_ws)
        full_ws.prepare()
        full_report = full_ws.run()
        full_ws.copy_report("full_rebuild.json")
        rebuild_comparison = {
            "semantic_fingerprint_equal": final_report["semantic_fingerprint"]
            == full_report["semantic_fingerprint"],
            "global_version_equal": final_report["global_version"] == full_report["global_version"],
            "stage_versions_equal": final_report["stage_versions"] == full_report["stage_versions"],
            "outputs_equal": (
                final_report["cross_encoding"] == full_report["cross_encoding"]
                and final_report["standardization"] == full_report["standardization"]
                and final_report["reduction"] == full_report["reduction"]
            ),
            "stage_diffs_equal": final_report["diagnostics"]["stage_diffs"]
            == full_report["diagnostics"]["stage_diffs"],
            "boundary_cases_equal": final_report["diagnostics"]["boundary_cases"]
            == full_report["diagnostics"]["boundary_cases"],
            "failures_equal": final_report["failures"] == full_report["failures"],
            "dependency_ledger_equal": final_report["diagnostics"]["dependency_ledger"]
            == full_report["diagnostics"]["dependency_ledger"],
            "ledger_order_stable": (
                _ledger_sorted(final_report, "reuse_ledger")
                and _ledger_sorted(final_report, "invalidation_ledger")
                and _ledger_sorted(full_report, "reuse_ledger")
                and _ledger_sorted(full_report, "invalidation_ledger")
                and _ledger_order(final_report, "invalidation_ledger")
                == _ledger_order(full_report, "invalidation_ledger")
            ),
        }

        serial_ws = new_workspace("serial")
        serial_config = serial_ws.config
        serial_config["runtime"]["max_workers"] = 1
        serial_ws.write_config(serial_config)
        serial_ws.prepare()
        serial_report = serial_ws.run()

        parallel_ws = new_workspace("parallel")
        parallel_config = parallel_ws.config
        parallel_config["runtime"]["max_workers"] = 4
        parallel_ws.write_config(parallel_config)
        parallel_ws.prepare()
        parallel_report = parallel_ws.run()
        concurrency_comparison = {
            "serial_max_observed_workers": serial_report["runtime"]["max_observed_workers"],
            "parallel_max_observed_workers": parallel_report["runtime"]["max_observed_workers"],
            "parallel_limit": parallel_report["runtime"]["max_workers"],
            "semantic_equal": serial_report["semantic_fingerprint"]
            == parallel_report["semantic_fingerprint"],
            "stage_versions_equal": serial_report["stage_versions"] == parallel_report["stage_versions"],
            "stage_diffs_equal": serial_report["diagnostics"]["stage_diffs"]
            == parallel_report["diagnostics"]["stage_diffs"],
        }

        quant_ws = new_workspace("boundary-q")
        quant_ws.prepare()
        quant_baseline = quant_ws.run()
        pca = next(item for item in quant_baseline["reduction"]["paths"] if item["path"] == "pca")
        single_scores = pca["tiers"]["single"]["scores"]
        double_scores = pca["tiers"]["double"]["scores"]
        row = min(range(len(double_scores)), key=lambda index: abs(double_scores[index]))
        quant_config = quant_ws.config
        quant_config["metrics"]["decision_threshold"] = (single_scores[row] + double_scores[row]) / 2.0
        quant_ws.write_config(quant_config)
        quant_report = quant_ws.run()
        quant_ws.copy_report("boundary_quantization.json")
        quant_sources = {
            source: sum(case["source"] == source for case in quant_report["diagnostics"]["boundary_cases"])
            for source in sorted({case["source"] for case in quant_report["diagnostics"]["boundary_cases"]})
        }

        dim_ws = new_workspace("boundary-dim")
        dim_config = dim_ws.config
        dim_config["precision"]["quantization"]["tier"] = "lossless"
        dim_ws.write_config(dim_config)
        dim_ws.prepare()
        dim_baseline = dim_ws.run()
        pca = next(item for item in dim_baseline["reduction"]["paths"] if item["path"] == "pca")
        single_scores = pca["tiers"]["single"]["scores"]
        double_scores = pca["tiers"]["double"]["scores"]
        row = min(range(len(double_scores)), key=lambda index: abs(double_scores[index]))
        dim_config = dim_ws.config
        dim_config["metrics"]["decision_threshold"] = (single_scores[row] + double_scores[row]) / 2.0
        dim_ws.write_config(dim_config)
        dim_report = dim_ws.run()
        dim_ws.copy_report("boundary_approximation.json")
        dim_sources = {
            source: sum(case["source"] == source for case in dim_report["diagnostics"]["boundary_cases"])
            for source in sorted({case["source"] for case in dim_report["diagnostics"]["boundary_cases"]})
        }

        recovery_ws = new_workspace("recovery")
        recovery_ws.prepare()
        recovery_baseline = recovery_ws.run()
        stages_root = recovery_ws.root / "cache" / "stages"
        metrics_current = stages_root / "metrics" / "current"
        metrics_current.unlink()
        abandoned = recovery_ws.root / ".tmp" / ".stage-build-metrics-abandoned"
        abandoned.mkdir(parents=True)
        (abandoned / "partial").write_text("partial", encoding="utf-8")
        recovered = recovery_ws.run()

        validation_ws = new_workspace("validation")
        bad_columns = validation_ws.columns
        bad_columns["columns"][-1]["type"] = "text"
        validation_ws.write_columns(bad_columns)
        bad_config = validation_ws.config
        bad_config["crossing"]["whitelist"] = ["region", "ghost", "tenure"]
        bad_config["crossing"]["order"] = 0
        bad_config["crossing"]["cardinality_budget"] = 0
        bad_config["precision"]["quantization"]["tier"] = "ultra"
        validation_ws.write_config(bad_config)
        invalid_config = validation_ws.load()
        validation_codes = sorted({item["code"] for item in invalid_config.failures})

        synthetic = attribute_boundary_cases(
            path="synthetic",
            single={"decisions": [0], "raw_decisions": [0], "scores": [0.1]},
            double={"decisions": [1], "raw_decisions": [1], "scores": [0.2]},
        )

        results: dict[str, Any] = {
            "status": "passed",
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "sample": {
                "checksum": first_prepare["checksum"],
                "second_prepare_reused": second_prepare["reused"],
                "checksum_stable": first_prepare["checksum"] == second_prepare["checksum"],
            },
            "baseline": _compact(baseline),
            "warm_rerun": {
                **_compact(warm),
                "semantic_equal": warm_semantic_equal,
            },
            "edits": edits,
            "region_replacement": {
                "reused_cross_combinations": reused_cross,
                "reuse_count": replacement_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation_count": replacement_report["diagnostics"]["cache_summary"][
                    "invalidated_or_recomputed"
                ],
            },
            "pca_edit": {
                "invalidated_reduction_paths": invalidated_pca_paths,
                "reused_reduction_paths": reused_pca_paths,
                "reuse_count": pca_report["diagnostics"]["cache_summary"]["reused"],
                "invalidation_count": pca_report["diagnostics"]["cache_summary"]["invalidated_or_recomputed"],
            },
            "final_incremental": _compact(final_report),
            "full_rebuild": _compact(full_report),
            "full_rebuild_comparison": rebuild_comparison,
            "concurrency_comparison": concurrency_comparison,
            "boundary_quantization": {
                "count": len(quant_report["diagnostics"]["boundary_cases"]),
                "sources": quant_sources,
                "cases": quant_report["diagnostics"]["boundary_cases"],
                "report": "boundary_quantization.json",
            },
            "boundary_approximation": {
                "count": len(dim_report["diagnostics"]["boundary_cases"]),
                "sources": dim_sources,
                "cases": dim_report["diagnostics"]["boundary_cases"],
                "report": "boundary_approximation.json",
            },
            "synthetic_approximation_attribution": synthetic,
            "validation_codes": {
                "codes": validation_codes,
                "loaded_columns": [descriptor["name"] for descriptor in invalid_config.descriptors],
                "failures": invalid_config.failures,
            },
            "snapshot_recovery": {
                "actions": recovered["recovery_actions"],
                "semantic_equal": recovered["semantic_fingerprint"]
                == recovery_baseline["semantic_fingerprint"],
                "stage_versions_equal": recovered["stage_versions"] == recovery_baseline["stage_versions"],
            },
        }
        checks = {
            "sample_checksum_stable": results["sample"]["checksum_stable"],
            "warm_semantic_equal": warm_semantic_equal,
            "replacement_reused_unaffected_crossings": bool(reused_cross),
            "pca_edit_only_invalidated_pca": invalidated_pca_paths == ["pca"],
            "incremental_full_semantic_equal": rebuild_comparison["semantic_fingerprint_equal"],
            "incremental_full_all_outputs_equal": all(
                value
                for key, value in rebuild_comparison.items()
                if key not in {"ledger_order_stable"}
            ),
            "serial_parallel_equal": concurrency_comparison["semantic_equal"]
            and concurrency_comparison["stage_versions_equal"]
            and concurrency_comparison["stage_diffs_equal"],
            "parallel_limit_respected": concurrency_comparison["parallel_max_observed_workers"]
            <= concurrency_comparison["parallel_limit"],
            "quantization_boundary_attributed": "quantization" in quant_sources,
            "approximation_boundary_attributed": "dimensional_approximation" in dim_sources,
            "all_business_codes_present": {
                "E_UNKNOWN_COLUMN",
                "E_TYPE_MISMATCH",
                "E_CROSS_ORDER",
                "E_CARDINALITY_BUDGET",
                "E_QUANTIZATION_PROFILE",
            }
            <= set(validation_codes),
            "snapshot_recovered": results["snapshot_recovery"]["semantic_equal"]
            and results["snapshot_recovery"]["stage_versions_equal"],
            "four_precision_stages_present": [item["stage"] for item in final_report["diagnostics"]["stage_summary"]]
            == ["cross_encoding", "standardization", "projection", "final_metrics"],
        }
        results["checks"] = checks
        results["status"] = "passed" if all(checks.values()) else "failed"
        _write_json(OUTPUT_DIR / "acceptance.json", results)
        (PROJECT_ROOT / "VERIFICATION.md").write_text(_section_markdown(results), encoding="utf-8")
        print(json.dumps({"status": results["status"], "checks": checks}, ensure_ascii=False, indent=2))
        return 0 if results["status"] == "passed" else 1
    finally:
        for workspace in workspaces:
            workspace.close()


if __name__ == "__main__":
    raise SystemExit(main())
