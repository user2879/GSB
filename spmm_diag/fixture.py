from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import MODE_FIXED, MODE_FLOAT32, ORDER_ASCENDING, SpMMConfig
from .csr import CSRMatrix, DistributionSpec, generate_csr
from .injections import InjectionItem, default_injection_manifest


@dataclass(frozen=True)
class DemoFixture:
    seed: int
    csr: CSRMatrix
    b_matrix: np.ndarray
    c_matrix: np.ndarray
    config: SpMMConfig
    single_thread_config: SpMMConfig
    distributions: dict[str, DistributionSpec]
    block_size_candidates: tuple[tuple[int, int], ...]
    precision_candidates: tuple[dict[str, Any], ...]
    injections: tuple[InjectionItem, ...]
    append_row_columns: tuple[int, ...]
    append_row_values: tuple[float, ...]
    append_c_row: tuple[float, ...]


def make_demo_fixture(seed: int = 20261123) -> DemoFixture:
    rng = np.random.default_rng(seed)
    rows, cols, dense_cols = 30, 18, 6
    distributions = {
        "uniform": DistributionSpec("uniform", 0, 10, power=1.0),
        "powerlaw": DistributionSpec("powerlaw", 0, 14, power=2.4),
        "extreme": DistributionSpec(
            "extreme", 0, 18, zero_probability=0.28, extreme_fraction=0.12
        ),
    }
    active_kind = str(rng.choice(["uniform", "powerlaw", "extreme"]))
    csr = generate_csr(rows, cols, distributions[active_kind], seed=seed + 101)
    b_matrix = rng.normal(0.0, 1.25, size=(cols, dense_cols))
    c_matrix = rng.normal(0.0, 0.45, size=(rows, dense_cols))

    block_size_candidates = ((2, 3), (3, 2), (4, 3), (5, 4))
    block_index = int(rng.integers(0, len(block_size_candidates)))
    row_block_size, nnz_chunk_size = block_size_candidates[block_index]
    precision_candidates = (
        {"precision_mode": MODE_FLOAT32, "fractional_bits": 8, "accumulator_bits": 24},
        {"precision_mode": MODE_FIXED, "fractional_bits": 6, "accumulator_bits": 18},
        {"precision_mode": MODE_FIXED, "fractional_bits": 7, "accumulator_bits": 20},
    )
    precision = precision_candidates[int(rng.integers(0, len(precision_candidates)))]
    alpha_choices = np.array([0.75, 1.0, 1.25], dtype=np.float64)
    beta_choices = np.array([0.0, 0.2, 0.35], dtype=np.float64)
    total_nnz = csr.nnz
    budget_candidates = np.array(
        [max(1, total_nnz // 2), max(1, total_nnz), max(1, total_nnz * 2)]
    )
    config = SpMMConfig(
        alpha=float(rng.choice(alpha_choices)),
        beta=float(rng.choice(beta_choices)),
        row_block_size=row_block_size,
        nnz_chunk_size=nnz_chunk_size,
        order=ORDER_ASCENDING,
        precision_mode=str(precision["precision_mode"]),
        fractional_bits=int(precision["fractional_bits"]),
        accumulator_bits=int(precision["accumulator_bits"]),
        max_workers=int(rng.integers(2, 5)),
        recompute_budget=int(rng.choice(budget_candidates)),
        hot_capacity=int(rng.choice([6, 8, 10])),
    )
    append_columns = tuple(
        sorted(int(value) for value in rng.choice(cols, size=4, replace=False))
    )
    append_values = tuple(float(value) for value in rng.uniform(-1.5, 1.5, size=4))
    append_c_row = tuple(float(value) for value in rng.normal(0.0, 0.2, size=dense_cols))
    return DemoFixture(
        seed=seed,
        csr=csr,
        b_matrix=b_matrix,
        c_matrix=c_matrix,
        config=config,
        single_thread_config=config.with_changes(max_workers=1),
        distributions=distributions,
        block_size_candidates=block_size_candidates,
        precision_candidates=precision_candidates,
        injections=tuple(default_injection_manifest(seed)),
        append_row_columns=append_columns,
        append_row_values=append_values,
        append_c_row=append_c_row,
    )

