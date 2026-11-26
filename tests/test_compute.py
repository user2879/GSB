from edge_cache.compute import compute_shard, generate_samples, quantize, shard_content_fingerprint
from edge_cache.config import load_config_dir


def _config(sample_config_dir):
    return load_config_dir(sample_config_dir)


def test_samples_are_deterministic(sample_config_dir):
    config = _config(sample_config_dir)
    shard = config.shard("s1")
    first = generate_samples(shard)
    second = generate_samples(shard)
    assert first.shape == second.shape
    assert (first == second).all()


def test_quantization_uses_integer_fixed_point(sample_config_dir):
    import numpy as np

    config = _config(sample_config_dir)
    shard = config.shard("s1")
    samples = generate_samples(shard)
    quantized = quantize(samples, fixed_point=4, quant_level=2)
    assert quantized.dtype == np.int64
    assert np.all(quantized % 2 == 0)
    assert np.all((quantized >= 0) & (quantized < 10_000))


def test_digest_changes_with_node_profile(sample_config_dir):
    config = _config(sample_config_dir)
    shard = config.shard("s1")
    node = config.node("node-a")
    first = compute_shard(shard, node.node_id, node.quota_version, 4, 1)
    second = compute_shard(shard, node.node_id, node.quota_version, 2, 1)
    assert first.digest != second.digest
    assert first.residual >= 0 and first.check_rate_permille in range(0, 1001)


def test_shard_content_fingerprint_stable(sample_config_dir):
    config = _config(sample_config_dir)
    shard = config.shard("s2")
    assert shard_content_fingerprint(shard) == shard_content_fingerprint(shard)
