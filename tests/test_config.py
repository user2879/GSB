from fablock.config import OperatorConfig
from fablock.errors import Code


def test_default_config_valid(cfg):
    assert cfg.validate() == []


def test_frac_bits_out_of_range():
    issues = OperatorConfig(fixed_point_frac_bits=40).validate()
    assert any(i.code == Code.E_FRAC_BITS_OUT_OF_RANGE for i in issues)


def test_quant_level_invalid():
    issues = OperatorConfig(quantization_level_bits=12).validate()
    assert any(i.code == Code.E_QUANT_LEVEL_INVALID for i in issues)


def test_workers_out_of_range():
    issues = OperatorConfig(thread_pool_workers=64).validate()
    assert any(i.code == Code.E_WORKERS_OUT_OF_RANGE for i in issues)


def test_config_hash_sensitive_to_tolerance_and_frac_bits(cfg):
    assert cfg.config_hash() != cfg.with_overrides(tol_score=0.01).config_hash()
    assert cfg.config_hash() != cfg.with_overrides(
        fixed_point_frac_bits=10).config_hash()
