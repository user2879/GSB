import numpy as np

from qcalib import distill as dist
from qcalib.models import DistillConfig, WeightLayer


def test_distillation_reduces_or_flags_increase():
    rng = np.random.default_rng(7)
    weights = rng.normal(size=(3, 4))
    layer = WeightLayer("l", (3, 4), weights, 1, 12)
    quant_weights = weights + rng.normal(scale=0.05, size=weights.shape)
    inputs = rng.normal(size=(6, 4))
    original = dist.forward(inputs, weights)
    cfg = DistillConfig(temperature=1.5, steps=5, soft_weight=0.3,
                        learning_rate=0.05, tolerance=0.2)
    result = dist.distill_layer(layer, quant_weights, original, inputs, cfg)
    assert result.post_deviation <= result.pre_deviation
    assert result.increased is False
    assert len(result.per_sample_pre) == 6
    assert result.correction_norm >= 0.0


def test_stage_attribution_first_breach():
    original = np.zeros((2, 3))
    good = np.zeros((2, 3))
    bad = np.ones((2, 3))
    attribution = dist.stage_attribution(
        original_output=original,
        calibration_output=good,
        requant_output=good,
        saturated_output=bad,
        corrected_output=bad,
        reference_output=good,
        tolerance=0.5)
    assert attribution["first_stage_over_tolerance"] == dist.STAGE_SATURATION
    assert attribution["stage_rmse"][dist.STAGE_SATURATION] > 0.5


def test_distillation_increase_is_flagged_with_judgement_numbers():
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(3, 4))
    layer = WeightLayer("l", (3, 4), weights, 1, 12)
    quant_weights = weights + 0.01
    inputs = rng.normal(size=(2, 4))
    original = dist.forward(inputs, weights)
    cfg = DistillConfig(temperature=1.0, steps=3, soft_weight=1.0,
                        learning_rate=5.0, tolerance=0.1)
    result = dist.distill_layer(layer, quant_weights, original, inputs, cfg)
    assert result.increased is True
    assert result.increase_judgement["post_minus_pre"] > 0.0
    assert (result.increase_judgement["post_rmse"]
            == result.post_deviation)
    assert (result.increase_judgement["pre_rmse"]
            == result.pre_deviation)
