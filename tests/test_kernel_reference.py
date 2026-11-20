from __future__ import annotations

import numpy as np
import pytest

from gemm_precision.errors import (
    E_BLOCK_INVALID,
    E_NONFINITE_INPUT,
    E_SHAPE_MISMATCH,
    BusinessError,
)
from gemm_precision.kernel import bmm, gemm
from gemm_precision.model import BlockShape
from gemm_precision.reference import reference_bmm, reference_gemm


def _gemm_data(seed: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(5, 257)).astype(np.float32),
        rng.normal(size=(257, 4)).astype(np.float32),
        rng.normal(size=(5, 4)).astype(np.float32),
    )


def test_gemm_matches_reference_all_loop_orders_and_reports_metadata() -> None:
    A, B, C = _gemm_data()
    alpha, beta = 1.25, -0.5
    reference = reference_gemm(A, B, C, alpha=alpha, beta=beta)
    hashes: set[str] = set()
    for block in (BlockShape(2, 3, 4), BlockShape(4, 2, 7)):
        for order in ("ijk", "ikj", "jik"):
            result = gemm(
                A,
                B,
                C,
                alpha=alpha,
                beta=beta,
                block_shape=block,
                loop_order=order,
                max_workers=3,
            )
            assert result.c.dtype == np.float32
            assert result.c.shape == (5, 4)
            assert np.max(np.abs(result.c.astype(np.float64) - reference.c)) < 2.0e-5
            assert len(result.element_metadata) == 20
            first = result.element_metadata[0]
            chunks = len(range(0, 257, block.bk))
            assert first["loop_order"] == order
            assert first["chain_length"] == 257
            assert first["k_chunk_count"] == chunks
            assert first["rounding_step_count"] == 257 + (chunks - 1) + 1 + 1
            assert f"merge=" in first["accumulation_order_id"]
            assert [task.key for task in result.tasks] == sorted(task.key for task in result.tasks)
            hashes.add(result.output_sha256)
    assert len(hashes) >= 2


def test_beta_zero_ignores_initial_c_but_beta_nonzero_reads_it() -> None:
    A, B, C = _gemm_data(seed=42)
    C_other = C * np.float32(-3.5)
    block = BlockShape(2, 2, 5)
    zero_beta_first = gemm(A, B, C, beta=0.0, block_shape=block, loop_order="ijk")
    zero_beta_second = gemm(A, B, C_other, beta=0.0, block_shape=block, loop_order="ijk")
    nonzero_beta_first = gemm(A, B, C, beta=0.5, block_shape=block, loop_order="ijk")
    nonzero_beta_second = gemm(A, B, C_other, beta=0.5, block_shape=block, loop_order="ijk")
    assert np.array_equal(zero_beta_first.c, zero_beta_second.c)
    assert not np.array_equal(nonzero_beta_first.c, nonzero_beta_second.c)


def test_bmm_matches_per_batch_reference_and_has_stable_task_order() -> None:
    rng = np.random.default_rng(43)
    A = rng.normal(size=(4, 6, 37)).astype(np.float32)
    B = rng.normal(size=(4, 37, 5)).astype(np.float32)
    C = rng.normal(size=(4, 6, 5)).astype(np.float32)
    reference = reference_bmm(A, B, C, alpha=0.75, beta=-0.25)
    for order in ("ijk", "ikj", "jik"):
        result = bmm(
            A,
            B,
            C,
            alpha=0.75,
            beta=-0.25,
            block_shape=(3, 4, 6),
            loop_order=order,
            max_workers=4,
        )
        assert np.max(np.abs(result.c.astype(np.float64) - reference.c)) < 1.0e-5
        assert result.c.shape == (4, 6, 5)
        assert [task.key for task in result.tasks] == sorted(task.key for task in result.tasks)
        assert {task.key[0] for task in result.tasks} == {0, 1, 2, 3}
        assert len(result.element_metadata) == 4 * 6 * 5


def test_bmm_batch_and_matrix_shape_errors_are_structured() -> None:
    rng = np.random.default_rng(44)
    A = rng.normal(size=(3, 5, 7)).astype(np.float32)
    B = rng.normal(size=(2, 7, 4)).astype(np.float32)
    C = rng.normal(size=(3, 5, 4)).astype(np.float32)
    with pytest.raises(BusinessError) as exc:
        bmm(A, B, C, block_shape=(2, 2, 2))
    error = exc.value
    assert error.code == E_SHAPE_MISMATCH
    assert error.stage == "bmm_batch_validation"
    assert error.batch_index == 2
    assert error.to_record()["shape"]["A"] == [3, 5, 7]

    B_bad = rng.normal(size=(3, 6, 4)).astype(np.float32)
    with pytest.raises(BusinessError) as exc2:
        bmm(A, B_bad, C, block_shape=(2, 2, 2))
    error2 = exc2.value
    assert error2.code == E_SHAPE_MISMATCH
    assert error2.stage == "bmm_shape_validation"
    assert error2.batch_index == 0


@pytest.mark.parametrize("bad_block", [BlockShape(0, 1, 1), BlockShape(-1, 2, 2), BlockShape(1, 1, 0)])
def test_block_validation_codes(bad_block: BlockShape) -> None:
    A, B, C = _gemm_data(seed=45)
    with pytest.raises(BusinessError) as exc:
        gemm(A, B, C, block_shape=bad_block)
    assert exc.value.code == E_BLOCK_INVALID
    assert exc.value.stage == "kernel_block_validation"
    assert exc.value.shape["block_shape"] == bad_block.as_list()


def test_nonfinite_input_and_parameters_are_rejected() -> None:
    A, B, C = _gemm_data(seed=46)
    A_bad = A.copy()
    A_bad[2, 3] = np.nan
    with pytest.raises(BusinessError) as exc:
        gemm(A_bad, B, C)
    assert exc.value.code == E_NONFINITE_INPUT
    assert exc.value.stage == "input_validation"
    assert exc.value.details["matrix"] == "A"
    assert exc.value.details["index"] == [2, 3]

    with pytest.raises(BusinessError) as exc2:
        gemm(A, B, C, alpha=float("inf"))
    assert exc2.value.code == E_NONFINITE_INPUT
    assert exc2.value.details["matrix"] == "alpha"
