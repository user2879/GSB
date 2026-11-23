from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .config import SpMMConfig
from .csr import CSRMatrix
from .diagnostics import Diagnostic, sort_diagnostics
from .engine import validate_bundle


@dataclass(frozen=True)
class InjectionItem:
    injection_id: str
    kind: str
    row: int | None = None
    column: int | None = None
    flat_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"id": self.injection_id, "kind": self.kind}
        if self.row is not None:
            value["row"] = self.row
        if self.column is not None:
            value["column"] = self.column
        if self.flat_index is not None:
            value["flat_index"] = self.flat_index
        return value


def default_injection_manifest(seed: int = 20261123) -> list[InjectionItem]:
    rng = np.random.default_rng(seed)
    offsets = [int(rng.integers(0, 6)), int(rng.integers(0, 6))]
    return [
        InjectionItem(
            injection_id="INJ-01-CSR-COLUMN",
            kind="csr_column_out_of_range",
            row=offsets[0],
            column=offsets[1],
        ),
        InjectionItem(
            injection_id="INJ-02-NONFINITE-B",
            kind="dense_nonfinite",
            row=int(rng.integers(0, 6)),
            column=int(rng.integers(0, 4)),
        ),
        InjectionItem(
            injection_id="INJ-03-ACC-WIDTH",
            kind="accumulator_width_invalid",
        ),
        InjectionItem(
            injection_id="INJ-04-BLOCK-SIZE",
            kind="block_size_zero",
        ),
    ]


def apply_injection(
    item: InjectionItem,
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    config: SpMMConfig,
) -> tuple[CSRMatrix, np.ndarray, SpMMConfig, dict[str, Any]]:
    injected_csr = csr
    injected_b = np.asarray(b_matrix, dtype=np.float64).copy()
    injected_config = config
    trace: dict[str, Any] = {}

    if item.kind == "csr_column_out_of_range":
        if csr.nnz == 0:
            raise ValueError("cannot inject a column index into an empty CSR matrix")
        row = int((item.row or 0) % csr.rows)
        start = int(csr.row_offsets[row])
        end = int(csr.row_offsets[row + 1])
        if start == end:
            row = next(r for r in range(csr.rows) if csr.row_nnz(r) > 0)
            start = int(csr.row_offsets[row])
            end = int(csr.row_offsets[row + 1])
        position = start + int((item.column or 0) % (end - start))
        columns = csr.col_indices.copy()
        original = int(columns[position])
        columns[position] = csr.cols + 7
        injected_csr = replace(csr, col_indices=columns)
        trace = {
            "path": f"csr.col_indices[{position}]",
            "row": row,
            "before": original,
            "after": int(columns[position]),
        }
    elif item.kind == "dense_nonfinite":
        if injected_b.size == 0:
            raise ValueError("cannot inject a non-finite value into an empty B matrix")
        row = int((item.row or 0) % injected_b.shape[0])
        column = int((item.column or 0) % injected_b.shape[1])
        injected_b[row, column] = np.inf
        trace = {
            "path": f"B[{row},{column}]",
            "before": "finite",
            "after": "inf",
        }
    elif item.kind == "accumulator_width_invalid":
        injected_config = config.with_changes(accumulator_bits=0)
        trace = {
            "path": "config.accumulator_bits",
            "before": config.accumulator_bits,
            "after": 0,
        }
    elif item.kind == "block_size_zero":
        injected_config = config.with_changes(row_block_size=0)
        trace = {
            "path": "config.row_block_size",
            "before": config.row_block_size,
            "after": 0,
        }
    else:
        raise ValueError(f"unknown injection kind: {item.kind}")
    return injected_csr, injected_b, injected_config, trace


def run_injection_suite(
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
    items: list[InjectionItem] | None = None,
) -> dict[str, Any]:
    manifest = list(items or default_injection_manifest())
    original_csr_fingerprint = csr.fingerprint()
    original_b = np.asarray(b_matrix, dtype=np.float64).copy()
    original_config = config.to_dict()
    results: list[dict[str, Any]] = []
    all_failures: list[Diagnostic] = []
    for item in manifest:
        injected_csr, injected_b, injected_config, trace = apply_injection(
            item, csr, b_matrix, config
        )
        failures = validate_bundle(injected_csr, injected_b, c_matrix, injected_config)
        tagged = [replace(failure, injection_id=item.injection_id) for failure in failures]
        all_failures.extend(tagged)
        results.append(
            {
                "injection_id": item.injection_id,
                "kind": item.kind,
                "trace": trace,
                "failure_codes": sorted({failure.code for failure in tagged}),
                "failures": [failure.to_dict() for failure in sort_diagnostics(tagged)],
                "isolated": bool(tagged),
            }
        )
    isolation = {
        "csr_unchanged": csr.fingerprint() == original_csr_fingerprint,
        "b_unchanged": np.array_equal(original_b, np.asarray(b_matrix, dtype=np.float64)),
        "config_unchanged": config.to_dict() == original_config,
    }
    return {
        "manifest": [item.to_dict() for item in manifest],
        "results": results,
        "failures": [failure.to_dict() for failure in sort_diagnostics(all_failures)],
        "isolation": isolation,
        "all_injections_detected": all(item["isolated"] for item in results),
    }

