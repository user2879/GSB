from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import E_INTERNAL, BusinessError
from .kernel import bmm, gemm
from .model import AccuracyTolerance, BlockShape, ExperimentConfig, InjectionSpec
from .samples import SampleData


@dataclass(frozen=True)
class InjectionOutcome:
    result: dict[str, Any]
    failure: dict[str, Any] | None


def _as_value(text: str | None) -> float:
    lowered = (text or "nan").lower()
    if lowered in {"nan", "+nan", "-nan"}:
        return float("nan")
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if lowered in {"-inf", "-infinity"}:
        return float("-inf")
    return float(text)  # type: ignore[arg-type]


def _copy_with_nonfinite(
    data: SampleData,
    injection: InjectionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = {"A": data.A.copy(), "B": data.B.copy(), "C": data.C.copy()}
    target = injection.target or "A"
    if target not in arrays:
        raise BusinessError(
            code=E_INTERNAL,
            stage="injection_setup",
            message=f"Unknown non-finite injection target {target}.",
            injection_id=injection.injection_id,
        )
    index = tuple(injection.index or (0, 0))
    value = _as_value(injection.value)
    arrays[target][index] = value
    location = {
        "matrix": target,
        "index": list(index),
        "value": "nan" if np.isnan(value) else ("inf" if value > 0 else ("-inf" if value < 0 else value)),
    }
    return arrays["A"], arrays["B"], arrays["C"], location


def _shape_mismatch_arrays(
    data: SampleData,
    injection: InjectionSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    A = data.A
    B = data.B
    C = data.C
    target = injection.target or "B"
    if not data.spec.is_bmm:
        raise BusinessError(
            code=E_INTERNAL,
            stage="injection_setup",
            message="Shape-mismatch injection requires a BMM sample.",
            injection_id=injection.injection_id,
        )
    if target == "A":
        A = A[:-1]
    elif target == "B":
        B = B[:-1]
    elif target == "C":
        C = C[:-1]
    else:
        raise BusinessError(
            code=E_INTERNAL,
            stage="injection_setup",
            message=f"Unknown shape injection target {target}.",
            injection_id=injection.injection_id,
        )
    return A, B, C, {
        "target": target,
        "action": "drop_last_batch",
        "original_shapes": {
            "A": list(data.A.shape),
            "B": list(data.B.shape),
            "C": list(data.C.shape),
        },
    }


def _failure_from_error(
    injection: InjectionSpec,
    error: BusinessError,
    combo_id: str,
) -> dict[str, Any]:
    record = {
        "case_id": injection.injection_id,
        "sample": injection.sample,
        "combo_id": combo_id,
        "injection_id": injection.injection_id,
        "expected_code": injection.expected_code,
        **error.to_record(),
    }
    return record


def run_injection(
    injection: InjectionSpec,
    config: ExperimentConfig,
    samples: dict[str, SampleData],
) -> InjectionOutcome:
    data = samples[injection.sample]
    block = config.block_shapes[0]
    order = config.loop_orders[0]
    evidence: dict[str, Any] = {}
    error: BusinessError | None = None
    combo_id = f"injection:{injection.injection_id}"

    try:
        if injection.kind == "nonfinite":
            A, B, C, evidence = _copy_with_nonfinite(data, injection)
            kernel = bmm if data.spec.is_bmm else gemm
            kernel(
                A,
                B,
                C,
                alpha=data.spec.alpha,
                beta=data.spec.beta,
                block_shape=block,
                loop_order=order,
                max_workers=1,
            )
        elif injection.kind == "shape_mismatch":
            A, B, C, evidence = _shape_mismatch_arrays(data, injection)
            bmm(
                A,
                B,
                C,
                alpha=data.spec.alpha,
                beta=data.spec.beta,
                block_shape=block,
                loop_order=order,
                max_workers=1,
            )
        elif injection.kind == "block_invalid":
            block_value = BlockShape.from_any(injection.block_shape or (0, 1, 1))
            evidence = {"block_shape": block_value.as_list()}
            block_value.validate(stage="kernel_block_validation")
            raise BusinessError(
                code=E_INTERNAL,
                stage="injection_did_not_hit",
                message="Invalid block injection unexpectedly validated.",
                injection_id=injection.injection_id,
            )
        elif injection.kind == "tolerance_invalid":
            tolerance = AccuracyTolerance(
                float(injection.atol if injection.atol is not None else 0.0),
                float(injection.rtol if injection.rtol is not None else 0.0),
            )
            evidence = {"tolerance": tolerance.as_dict()}
            tolerance.validate(stage="error_analysis_tolerance_validation")
            raise BusinessError(
                code=E_INTERNAL,
                stage="injection_did_not_hit",
                message="Invalid tolerance injection unexpectedly validated.",
                injection_id=injection.injection_id,
            )
        else:
            raise BusinessError(
                code=E_INTERNAL,
                stage="injection_setup",
                message=f"Unsupported injection kind: {injection.kind}",
                injection_id=injection.injection_id,
            )
    except BusinessError as exc:
        exc.injection_id = injection.injection_id
        error = exc
    except Exception as exc:
        error = BusinessError(
            code=E_INTERNAL,
            stage=f"injection_{injection.kind}",
            message="Unexpected exception during injection.",
            details={"exception_type": type(exc).__name__, "error": str(exc)},
            injection_id=injection.injection_id,
        )

    actual_code = error.code if error is not None else None
    hit = error is not None and actual_code == injection.expected_code
    result = {
        "injection_id": injection.injection_id,
        "kind": injection.kind,
        "sample": injection.sample,
        "expected_code": injection.expected_code,
        "actual_code": actual_code,
        "hit": hit,
        "matched_expected_code": actual_code == injection.expected_code,
        "stage": error.stage if error is not None else None,
        "shape": error.shape if error is not None else None,
        "batch_index": error.batch_index if error is not None else None,
        "evidence": evidence,
        "error_details": dict(error.details) if error is not None else None,
    }
    failure = _failure_from_error(injection, error, combo_id) if error is not None else None
    return InjectionOutcome(result=result, failure=failure)


def run_injections(
    config: ExperimentConfig,
    samples: dict[str, SampleData],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes = [run_injection(injection, config, samples) for injection in config.injections]
    results = sorted((outcome.result for outcome in outcomes), key=lambda item: item["injection_id"])
    failures = sorted(
        (outcome.failure for outcome in outcomes if outcome.failure is not None),
        key=lambda item: item["case_id"],
    )
    return results, failures
