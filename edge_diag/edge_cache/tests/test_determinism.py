"""确定性与定点校验测试。"""

import numpy as np

from edge_cache import determinism as dt


def test_shard_bytes_order_independent():
    a = dt.generate_shard_bytes("k", 7, 2, 512)
    b = dt.generate_shard_bytes("k", 7, 2, 512)
    assert a == b
    # 不同分片序号/种子必须产生不同字节
    c = dt.generate_shard_bytes("k", 7, 3, 512)
    assert a != c


def test_digest_binds_node_and_fields():
    payload = dt.generate_shard_bytes("video", 9, 1, 256)
    d1 = dt.compute_digest("node-a", 11, "video", 1, payload, 1000)
    d2 = dt.compute_digest("node-a", 11, "video", 1, payload, 1000)
    assert d1["digest"] == d2["digest"]
    # 节点不同 -> 摘要不同（绑定节点标识与种子）
    d3 = dt.compute_digest("node-b", 22, "video", 1, payload, 1000)
    assert d1["digest"] != d3["digest"]
    # 全部定点结果为整数
    assert isinstance(d1["fp_sum"], int)
    assert isinstance(d1["fp_sqsum"], int)
    assert d1["session_scale"] == 1000


def test_fixed_point_uses_numpy_integer_quantization():
    payload = bytes(256)
    fp = dt.fixed_point_check(5, "node-a", "k", 0, payload, 1000)
    assert fp["sample_count"] == 256
    assert 0 <= fp["fp_sum"] <= 256 * 999
    # 量化结果可由相同种子复现
    fp2 = dt.fixed_point_check(5, "node-a", "k", 0, payload, 1000)
    assert fp == fp2


def test_mix_seed_order_sensitive():
    assert dt.mix_seed([1, "x"]) != dt.mix_seed(["x", 1])
