from __future__ import annotations

from fcdiag.fixedpoint import FixedConfig, quantize
from fcdiag.pipeline import FixedPipeline
from fcdiag.runner import Runner


def test_quantization_records_step_and_residual():
    value, residual = quantize(17, gear=5)
    assert value == 15 and residual == 2
    value, residual = quantize(18, gear=5)
    assert value == 20 and residual == -2


def test_fixed_point_scaling_rounds_at_configured_width():
    from fcdiag.fixedpoint import scaled

    assert scaled(1.234567, 4) == 12346
    assert scaled(-1.234567, 4) == -12346
    assert scaled(1.25, 4) == 12500


def test_gear_and_frac_bits_change_payload_values(context):
    state = context["state"]
    receipts = context["receipts"][3]
    baseline = Runner(state, receipts).run()
    gear_state = state.copy()
    gear_state.config = FixedConfig(
        state.config.fixed.frac_bits, 10, state.config.budget, state.config.policy, state.config.seed
    ) if False else state.config
    # Build explicit config variants through the same immutable model.
    from fcdiag.models import Config

    gear_config = Config(
        FixedConfig(state.config.fixed.frac_bits, 10),
        state.config.budget,
        state.config.policy,
        state.config.seed,
    )
    gear_state = state.copy()
    object.__setattr__(gear_state, "config", gear_config)
    gear_run = Runner(gear_state, receipts).run()
    frac_config = Config(
        FixedConfig(6, state.config.fixed.gear),
        state.config.budget,
        state.config.policy,
        state.config.seed,
    )
    frac_state = state.copy()
    object.__setattr__(frac_state, "config", frac_config)
    frac_run = Runner(frac_state, receipts).run()
    base_view = baseline.artifacts["train_view:global"].payload
    gear_view = gear_run.artifacts["train_view:global"].payload
    frac_view = frac_run.artifacts["train_view:global"].payload
    assert base_view != gear_view
    assert base_view != frac_view
    assert all(log["step"] == 10 for log in gear_run.quant_logs["interact"])
    assert any(log["residual"] != 0 for log in gear_run.quant_logs["interact"])
    assert any(log["step"] == 1 for log in frac_run.quant_logs["interact"])


def test_every_artifact_has_digest_and_cache_key(context):
    result = Runner(context["state"], context["receipts"][3]).run()
    for artifact_id, artifact in result.artifacts.items():
        assert len(artifact.digest) == 16
        assert artifact_id in result.cache_keys
        assert len(result.cache_keys[artifact_id]) == 24


def test_deterministic_twice(context):
    left = Runner(context["state"], context["receipts"][3]).run()
    right = Runner(context["state"], context["receipts"][3]).run()
    assert left.artifacts == right.artifacts
    assert left.cache_keys == right.cache_keys
    assert left.ledger == right.ledger
