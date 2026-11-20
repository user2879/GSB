from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gemm_precision.errors import E_INTERNAL, BusinessError
from gemm_precision.model import (
    AccuracyTolerance,
    BlockShape,
    ExperimentConfig,
    InjectionSpec,
    SampleSpec,
)
from gemm_precision.runner import run_experiment
from gemm_precision.samples import generate_sample, save_sample_npz


def _small_config(report_path: str, *, injections: bool = True) -> ExperimentConfig:
    sample = SampleSpec(
        name="tiny_bmm",
        category="special_values",
        seed=9001,
        M=4,
        N=5,
        K=39,
        batch_size=2,
        alpha=0.75,
        beta=-0.25,
        special_fraction=0.15,
        description="Tiny deterministic BMM pipeline sample.",
    )
    injection_specs = (
        (
            InjectionSpec(
                "INJ-1",
                "nonfinite",
                sample.name,
                target="A",
                index=(0, 0, 0),
                value="nan",
                expected_code="E_NONFINITE_INPUT",
            ),
            InjectionSpec(
                "INJ-2",
                "shape_mismatch",
                sample.name,
                target="B",
                expected_code="E_SHAPE_MISMATCH",
            ),
            InjectionSpec(
                "INJ-3",
                "block_invalid",
                sample.name,
                block_shape=(0, 2, 3),
                expected_code="E_BLOCK_INVALID",
            ),
            InjectionSpec(
                "INJ-4",
                "tolerance_invalid",
                sample.name,
                atol=0.0,
                rtol=0.0,
                expected_code="E_TOLERANCE_INVALID",
            ),
        )
        if injections
        else ()
    )
    return ExperimentConfig(
        seed=9001,
        samples=(sample,),
        block_shapes=(BlockShape(2, 3, 5), BlockShape(3, 2, 8)),
        loop_orders=("ijk", "ikj", "jik"),
        tolerance=AccuracyTolerance(1.0e-9, 1.0e-9),
        layer_exceed_limit=0.0,
        layer_count=3,
        top_k_contributors=3,
        max_workers=3,
        serial_workers=1,
        report_path=report_path,
        injections=injection_specs,
    )


def test_special_sample_generation_is_reproducible_and_traceable(tmp_path: Path) -> None:
    spec = SampleSpec(
        name="special",
        category="special_values",
        seed=12345,
        M=7,
        N=6,
        K=11,
        batch_size=2,
        special_fraction=0.5,
    )
    first = generate_sample(spec)
    second = generate_sample(spec)
    assert np.array_equal(first.A, second.A)
    assert np.array_equal(first.B, second.B)
    assert np.array_equal(first.C, second.C)
    assert first.special_positions == second.special_positions
    assert first.digest == second.digest
    kinds = {item["kind"] for item in first.special_positions}
    assert "zero" in kinds
    assert "positive_subnormal" in kinds or "negative_subnormal" in kinds
    for item in first.special_positions:
        matrix = item["matrix"]
        array = first.A if matrix == "A" else first.B
        assert float(array[tuple(item["index"])]) == item["value"]
        assert item["float32_bits"].startswith("0x")
    path = tmp_path / "special.npz"
    save_sample_npz(first, path)
    assert path.exists()
    with np.load(path) as loaded:
        assert np.array_equal(loaded["A"], first.A)
        assert np.array_equal(loaded["B"], first.B)
        assert np.array_equal(loaded["C"], first.C)


def test_explicit_batched_true_supports_batch_size_one() -> None:
    spec = SampleSpec(
        name="batch_one_bmm",
        category="small",
        seed=77,
        M=3,
        N=4,
        K=5,
        batch_size=1,
        batched=True,
    )
    sample = generate_sample(spec)
    assert spec.is_bmm is True
    assert sample.A.shape == (1, 3, 5)
    assert sample.B.shape == (1, 5, 4)
    assert sample.C.shape == (1, 3, 4)


def test_full_pipeline_injections_concurrency_and_atomic_deterministic_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    config = _small_config(str(path))
    first = run_experiment(config, write=True)
    first_bytes = path.read_bytes()
    markdown = path.with_suffix(".md")
    assert markdown.exists()
    assert not list(tmp_path.glob(".*.tmp"))

    second = run_experiment(config, write=True)
    second_bytes = path.read_bytes()
    assert first_bytes == second_bytes
    assert first["report_digest"] == second["report_digest"]

    report = json.loads(first_bytes)
    assert report["summary"]["combination_count"] == 6
    assert report["summary"]["injections_hit"] == 4
    assert report["summary"]["all_concurrency_checks_equal"] is True
    assert len(report["concurrency_checks"]) == 6
    assert all(item["equal"] for item in report["concurrency_checks"])
    assert {item["actual_code"] for item in report["injections"]} == {
        "E_NONFINITE_INPUT",
        "E_SHAPE_MISMATCH",
        "E_BLOCK_INVALID",
        "E_TOLERANCE_INVALID",
    }
    assert report["business_codes"]["observed"] == [
        "E_BLOCK_INVALID",
        "E_NONFINITE_INPUT",
        "E_SHAPE_MISMATCH",
        "E_TOLERANCE_INVALID",
    ]
    assert report["layer_exceedance_summary"]
    for combo in report["combinations"]:
        assert len(combo["element_trace_identifiers"]) == 2 * 4 * 5
        assert combo["reference"]["path"] == "float64_serial_k_ascending"
        assert combo["exceeded_elements"]
        assert combo["layer_tables"].keys() == {"M", "N", "K", "BATCH"}
    for sample in report["samples"]:
        assert sample["special_positions"]
        assert sample["generation"]["seed"] == 9001
    assert report["report_digest"]
    assert report["failures"]


def test_failure_isolation_continues_remaining_combinations(
    monkeypatch,
) -> None:
    import gemm_precision.runner as runner_module

    config = replace(
        _small_config("unused.json", injections=False),
        samples=(
            SampleSpec(
                name="tiny_gemm",
                category="small",
                seed=9002,
                M=4,
                N=5,
                K=31,
                alpha=1.0,
                beta=0.0,
            ),
        ),
        block_shapes=(BlockShape(2, 2, 4), BlockShape(3, 2, 7)),
        loop_orders=("ijk", "jik"),
    )
    original = runner_module._evaluate_one

    def flaky(data, reference, block, order, workers, experiment_config):
        if workers == config.max_workers and block == config.block_shapes[0] and order == "ijk":
            raise BusinessError(
                code=E_INTERNAL,
                stage="combination_execution",
                message="Synthetic isolated failure.",
                shape={"A": list(data.A.shape)},
            )
        return original(data, reference, block, order, workers, experiment_config)

    monkeypatch.setattr(runner_module, "_evaluate_one", flaky)
    report = run_experiment(config, write=False)
    assert report["summary"]["expected_combination_count"] == 4
    assert report["summary"]["combination_count"] == 3
    assert report["summary"]["failure_count"] == 1
    failure = report["failures"][0]
    assert failure["business_code"] == E_INTERNAL
    assert failure["stage"] == "combination_execution"
    assert failure["block_shape"] == {"BM": 2, "BN": 2, "BK": 4}
