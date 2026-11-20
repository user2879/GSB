from __future__ import annotations

import platform
from typing import Any, Callable

import numpy as np

from .analysis import analyse_combination
from .comparison import build_sample_comparison
from .errors import (
    BUSINESS_CODE_CATALOG,
    E_CONCURRENCY_MISMATCH,
    E_INTERNAL,
    BusinessError,
)
from .injection import run_injections
from .kernel import KernelResult, bmm, gemm
from .model import BlockShape, ExperimentConfig, LoopOrder, SampleSpec
from .reference import ReferenceResult, reference_bmm, reference_gemm
from .report import write_report
from .samples import SampleData, generate_samples
from .util import payload_sha256


def _kernel_callable(data: SampleData) -> Callable[..., KernelResult]:
    return bmm if data.spec.is_bmm else gemm


def _reference_callable(data: SampleData) -> Callable[..., ReferenceResult]:
    return reference_bmm if data.spec.is_bmm else reference_gemm


def _combo_id(sample: SampleSpec, block: BlockShape, order: LoopOrder) -> str:
    return f"{sample.name}|BM{block.bm}-BN{block.bn}-BK{block.bk}|{order}"


def _combo_failure(
    sample: SampleSpec,
    block: BlockShape | None,
    order: LoopOrder | None,
    combo_id: str,
    error: BusinessError,
    phase: str,
) -> dict[str, Any]:
    return {
        "case_id": combo_id,
        "sample": sample.name,
        "combo_id": combo_id,
        "block_shape": block.as_dict() if block is not None else None,
        "loop_order": order,
        "phase": phase,
        **error.to_record(),
    }


def _evaluate_one(
    data: SampleData,
    reference: ReferenceResult,
    block: BlockShape,
    order: LoopOrder,
    workers: int,
    config: ExperimentConfig,
) -> tuple[KernelResult, dict[str, Any]]:
    kernel = _kernel_callable(data)
    main = kernel(
        data.A,
        data.B,
        data.C,
        alpha=data.spec.alpha,
        beta=data.spec.beta,
        block_shape=block,
        loop_order=order,
        max_workers=workers,
    )
    combination = analyse_combination(
        data.A,
        data.B,
        data.C,
        main,
        reference,
        config.tolerance,
        config,
        data.spec,
    )
    return main, combination


def _compare_parallel_serial(
    parallel_main: KernelResult,
    parallel_combo: dict[str, Any],
    serial_main: KernelResult,
    serial_combo: dict[str, Any],
    config: ExperimentConfig,
) -> dict[str, Any]:
    checks = {
        "c_bitwise_equal": parallel_main.c.tobytes(order="C") == serial_main.c.tobytes(order="C"),
        "output_sha256_equal": parallel_main.output_sha256 == serial_main.output_sha256,
        "element_metadata_equal": parallel_combo["element_trace_identifiers"]
        == serial_combo["element_trace_identifiers"],
        "task_order_equal": parallel_combo["tasks"] == serial_combo["tasks"],
        "error_attribution_equal": parallel_combo["exceeded_elements"]
        == serial_combo["exceeded_elements"],
        "layer_tables_equal": parallel_combo["layer_tables"] == serial_combo["layer_tables"],
        "layer_exceedances_equal": parallel_combo["layer_exceedances"]
        == serial_combo["layer_exceedances"],
        "audit_digest_equal": parallel_combo["audit_digest"] == serial_combo["audit_digest"],
    }
    equal = all(checks.values())
    result = {
        "sample": parallel_combo["sample"],
        "combo_id": (
            f"{parallel_combo['sample']}|BM{parallel_main.block_shape.bm}-"
            f"BN{parallel_main.block_shape.bn}-BK{parallel_main.block_shape.bk}|"
            f"{parallel_main.loop_order}"
        ),
        "block_shape": parallel_main.block_shape.as_dict(),
        "loop_order": parallel_main.loop_order,
        "parallel_workers": int(config.max_workers),
        "serial_workers": int(config.serial_workers),
        "parallel_output_sha256": parallel_main.output_sha256,
        "serial_output_sha256": serial_main.output_sha256,
        "parallel_audit_digest": parallel_combo["audit_digest"],
        "serial_audit_digest": serial_combo["audit_digest"],
        "checks": checks,
        "equal": equal,
    }
    if not equal:
        raise BusinessError(
            code=E_CONCURRENCY_MISMATCH,
            stage="serial_parallel_consistency",
            message="Thread-pool execution disagrees with serial execution.",
            shape={"C": list(parallel_main.c.shape)},
            details={"checks": checks, "combo": result["combo_id"]},
        )
    return result


def run_experiment(config: ExperimentConfig, *, write: bool = True) -> dict[str, Any]:
    config.validate()
    samples = generate_samples(config.samples)
    sample_reports = [samples[spec.name].metadata() for spec in config.samples]
    failures: list[dict[str, Any]] = []
    combinations: list[dict[str, Any]] = []
    concurrency_checks: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for sample in config.samples:
        data = samples[sample.name]
        sample_combinations: list[dict[str, Any]] = []
        reference_call = _reference_callable(data)
        try:
            reference = reference_call(
                data.A,
                data.B,
                data.C,
                alpha=sample.alpha,
                beta=sample.beta,
            )
        except BusinessError as exc:
            failure_id = f"{sample.name}|reference"
            failures.append(
                _combo_failure(sample, None, None, failure_id, exc, "reference")
            )
            comparisons.append(build_sample_comparison(sample, [], config))
            continue
        except Exception as exc:
            wrapped = BusinessError(
                code=E_INTERNAL,
                stage="reference_execution",
                message="Unexpected exception in float64 reference execution.",
                details={"exception_type": type(exc).__name__, "error": str(exc)},
            )
            failure_id = f"{sample.name}|reference"
            failures.append(
                _combo_failure(sample, None, None, failure_id, wrapped, "reference")
            )
            comparisons.append(build_sample_comparison(sample, [], config))
            continue

        for block in config.block_shapes:
            for order in config.loop_orders:
                combo_id = _combo_id(sample, block, order)
                try:
                    parallel_main, parallel_combo = _evaluate_one(
                        data,
                        reference,
                        block,
                        order,
                        config.max_workers,
                        config,
                    )
                    serial_main, serial_combo = _evaluate_one(
                        data,
                        reference,
                        block,
                        order,
                        config.serial_workers,
                        config,
                    )
                    check = _compare_parallel_serial(
                        parallel_main,
                        parallel_combo,
                        serial_main,
                        serial_combo,
                        config,
                    )
                    concurrency_checks.append(check)
                    sample_combinations.append(parallel_combo)
                    combinations.append(parallel_combo)
                except BusinessError as exc:
                    failures.append(
                        _combo_failure(sample, block, order, combo_id, exc, "combination")
                    )
                except Exception as exc:
                    wrapped = BusinessError(
                        code=E_INTERNAL,
                        stage="combination_execution",
                        message="Unexpected exception; remaining combinations continue.",
                        details={"exception_type": type(exc).__name__, "error": str(exc)},
                    )
                    failures.append(
                        _combo_failure(sample, block, order, combo_id, wrapped, "combination")
                    )

        comparisons.append(build_sample_comparison(sample, sample_combinations, config))

    injection_results, injection_failures = run_injections(config, samples)
    failures.extend(injection_failures)
    failures.sort(key=lambda item: (str(item.get("case_id")), str(item.get("business_code"))))

    layer_exceedance_summary: list[dict[str, Any]] = []
    for combo in combinations:
        for item in combo["layer_exceedances"]:
            layer_exceedance_summary.append(
                {
                    "sample": combo["sample"],
                    "sample_category": combo["sample_category"],
                    "axis": item["axis"],
                    "layer_key": item["layer_key"],
                    "exceed_ratio": item["exceed_ratio"],
                    "exceed_count": item["exceed_count"],
                    "element_count": item["element_count"],
                    "configured_limit": item["configured_limit"],
                    "block_shape": item["block_shape"],
                    "loop_order": item["loop_order"],
                }
            )
    layer_exceedance_summary.sort(
        key=lambda item: (
            item["sample"],
            item["axis"],
            -float(item["exceed_ratio"]),
            item["layer_key"],
            item["block_shape"]["BM"],
            item["block_shape"]["BN"],
            item["block_shape"]["BK"],
            item["loop_order"],
        )
    )
    concurrency_checks.sort(key=lambda item: item["combo_id"])
    combinations.sort(
        key=lambda item: (
            item["sample"],
            item["block_shape"]["BM"],
            item["block_shape"]["BN"],
            item["block_shape"]["BK"],
            item["loop_order"],
        )
    )
    comparisons.sort(key=lambda item: item["sample"])

    actual_codes = {
        str(item["business_code"])
        for item in failures
        if item.get("business_code") is not None
    }
    actual_codes.update(
        str(item["actual_code"])
        for item in injection_results
        if item.get("actual_code") is not None
    )
    report = {
        "schema_version": "1.0",
        "report_kind": "gemm_bmm_precision_observability",
        "determinism_policy": {
            "clock_fields": "omitted",
            "ordering": "explicit stable keys",
            "thread_collection": "submission order, not completion order",
            "json": "allow_nan=false, sort_keys=true, atomic replace",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_pool": "concurrent.futures.ThreadPoolExecutor",
            "serial_workers": int(config.serial_workers),
            "parallel_workers": int(config.max_workers),
        },
        "configuration": config.as_dict(),
        "reference_policy": {
            "path": "float64_serial_k_ascending",
            "product_accumulation": "float64",
            "alpha_beta": "float64",
            "per_output_reduction": "k ascending, zero initialization",
        },
        "loop_order_semantics": {
            "ijk": "intra-chunk ijk traversal; K-chunk partials merged forward sequentially",
            "ikj": "intra-chunk ikj traversal; K-chunk partials merged reverse sequentially",
            "jik": "intra-chunk jik traversal; K-chunk partials merged as a balanced binary tree",
        },
        "samples": sample_reports,
        "comparison_tables": comparisons,
        "combinations": combinations,
        "layer_exceedance_summary": layer_exceedance_summary,
        "concurrency_checks": concurrency_checks,
        "injections": injection_results,
        "failures": failures,
        "business_codes": {
            "observed": sorted(actual_codes),
            "catalog": dict(sorted(BUSINESS_CODE_CATALOG.items())),
        },
        "summary": {
            "sample_count": len(config.samples),
            "combination_count": len(combinations),
            "expected_combination_count": sum(
                len(config.block_shapes) * len(config.loop_orders) for _ in config.samples
            ),
            "injection_count": len(injection_results),
            "injections_hit": sum(1 for item in injection_results if item["hit"]),
            "failure_count": len(failures),
            "concurrency_checks": len(concurrency_checks),
            "all_concurrency_checks_equal": all(
                item["equal"] for item in concurrency_checks
            ),
            "layer_exceedance_count": len(layer_exceedance_summary),
        },
        "report_digest": "",
    }
    digest_payload = {key: value for key, value in report.items() if key != "report_digest"}
    report["report_digest"] = payload_sha256(digest_payload)
    if write:
        write_report(report, config.report_path)
    return report
