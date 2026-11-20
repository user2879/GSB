from __future__ import annotations

import math

import numpy as np
import pytest

from gemm_precision.analysis import analyse_combination
from gemm_precision.errors import E_TOLERANCE_INVALID, BusinessError
from gemm_precision.kernel import gemm
from gemm_precision.model import (
    AccuracyTolerance,
    BlockShape,
    ExperimentConfig,
    SampleSpec,
)
from gemm_precision.reference import reference_gemm


def _analysis_fixture(order: str = "ijk") -> tuple[dict, dict, SampleSpec]:
    rng = np.random.default_rng(71)
    A = rng.normal(size=(5, 257)).astype(np.float32)
    B = rng.normal(size=(257, 4)).astype(np.float32)
    C = rng.normal(size=(5, 4)).astype(np.float32)
    sample = SampleSpec("analysis", "small", 71, 5, 4, 257, alpha=1.0, beta=0.0)
    config = ExperimentConfig(
        seed=71,
        samples=(sample,),
        block_shapes=(BlockShape(2, 3, 5), BlockShape(4, 2, 16)),
        loop_orders=("ijk", "ikj", "jik"),
        tolerance=AccuracyTolerance(1.0e-9, 1.0e-9),
        layer_exceed_limit=0.0,
        layer_count=4,
        top_k_contributors=3,
        max_workers=3,
    )
    main = gemm(
        A,
        B,
        C,
        alpha=sample.alpha,
        beta=sample.beta,
        block_shape=config.block_shapes[0],
        loop_order=order,  # type: ignore[arg-type]
        max_workers=3,
    )
    reference = reference_gemm(A, B, C, alpha=sample.alpha, beta=sample.beta)
    combo = analyse_combination(
        A,
        B,
        C,
        main,
        reference,
        config.tolerance,
        config,
        sample,
    )
    return combo, {"A": A, "B": B, "C": C, "main": main, "reference": reference}, sample


def test_exceeded_elements_have_sorted_nonzero_top_contributors() -> None:
    combo, _, _ = _analysis_fixture("jik")
    assert combo["error_summary"]["exceed_count"] > 0
    for element in combo["exceeded_elements"]:
        contributions = element["top_contributors"]
        assert 0 < len(contributions) <= 3
        expected_order = sorted(
            contributions,
            key=lambda item: (-item["absolute_rounding_contribution"], item["accumulation_position"]),
        )
        assert contributions == expected_order
        for item in contributions:
            assert item["a_value"] != 0.0
            assert item["b_value"] != 0.0
            assert item["accumulation_position"] == item["k"]
        merge_stages = [
            stage for stage in element["propagation_stages"] if stage["stage"] == "chunk_merge"
        ]
        assert merge_stages
        assert merge_stages[0]["merge_schedule"] == "k-chunk-balanced-tree"
        assert merge_stages[0]["merge_count"] > 0
        assert merge_stages[0]["worst_merges"]


def test_layers_cover_elements_and_share_partitions_sum_to_one() -> None:
    combo, _, _ = _analysis_fixture("ijk")
    total_elements = combo["error_summary"]["element_count"]
    for axis in ("M", "N", "K", "BATCH"):
        assert sum(row["element_count"] for row in combo["layer_tables"][axis]) == total_elements
    m_share = sum(row["attribution_share"] for row in combo["layer_tables"]["M"])
    n_share = sum(row["attribution_share"] for row in combo["layer_tables"]["N"])
    k_share = sum(row["attribution_share"] for row in combo["layer_tables"]["K"])
    batch_share = sum(row["attribution_share"] for row in combo["layer_tables"]["BATCH"])
    assert math.isclose(m_share, 1.0, rel_tol=1e-12)
    assert math.isclose(n_share, 1.0, rel_tol=1e-12)
    assert math.isclose(k_share, 1.0, rel_tol=1e-12)
    assert math.isclose(batch_share, 1.0, rel_tol=1e-12)
    assert combo["layer_exceedances"]


def test_loop_orders_change_numeric_outputs_and_audit_digests() -> None:
    combos = [_analysis_fixture(order)[0] for order in ("ijk", "ikj", "jik")]
    output_hashes = {combo["kernel"]["output_sha256"] for combo in combos}
    audit_hashes = {combo["audit_digest"] for combo in combos}
    assert len(output_hashes) == 3
    assert len(audit_hashes) == 3
    schedules = {
        next(
            stage["merge_schedule"]
            for stage in combo["exceeded_elements"][0]["propagation_stages"]
            if stage["stage"] == "chunk_merge"
        )
        for combo in combos
    }
    assert schedules == {
        "k-chunk-forward-sequential",
        "k-chunk-reverse-sequential",
        "k-chunk-balanced-tree",
    }


def test_zero_products_and_zero_reference_do_not_break_relative_metrics() -> None:
    A = np.zeros((2, 17), dtype=np.float32)
    B = np.zeros((17, 3), dtype=np.float32)
    C = np.zeros((2, 3), dtype=np.float32)
    sample = SampleSpec("zeros", "small", 72, 2, 3, 17)
    config = ExperimentConfig(
        72,
        (sample,),
        (BlockShape(1, 2, 4), BlockShape(2, 1, 5)),
        ("ijk", "jik"),
        AccuracyTolerance(1.0e-8, 1.0e-8),
    )
    main = gemm(A, B, C, block_shape=config.block_shapes[0], loop_order="ijk")
    reference = reference_gemm(A, B, C)
    combo = analyse_combination(A, B, C, main, reference, config.tolerance, config, sample)
    assert combo["error_summary"]["max_relative_error"] == 0.0
    assert combo["error_summary"]["exceed_count"] == 0
    assert all(row["attribution_share"] == 0.0 for row in combo["layer_tables"]["K"])


@pytest.mark.parametrize("atol,rtol", [(0.0, 0.0), (-1.0, 1.0), (float("nan"), 1.0)])
def test_invalid_tolerances_raise_business_code(atol: float, rtol: float) -> None:
    with pytest.raises(BusinessError) as exc:
        AccuracyTolerance(atol, rtol).validate(stage="error_analysis")
    assert exc.value.code == E_TOLERANCE_INVALID
    assert exc.value.stage == "error_analysis"
