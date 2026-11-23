import numpy as np

from fablock.cache import BlockCache, params_fingerprint
from fablock.pipeline import Pipeline


def _recompute(report, kind):
    return {tuple(e["block"]) if isinstance(e["block"], list) else e["block"]
            for e in report["cache"]["recompute"] if e["kind"] == kind}


def test_single_block_data_replace_invalidates_only_downstream(cfg, dataset):
    p = Pipeline(cfg, dataset)
    p.run(1)
    dataset.set_burst_query_blocks(())  # 排除突发块干扰
    # 替换查询块 2 的单块张量数据
    dataset.Q[16:24] += 0.5
    r = p.run_batch(1)
    assert _recompute(r, "score") == {(2, 0), (2, 1), (2, 2)}
    assert _recompute(r, "state") == {2}
    assert _recompute(r, "out") == {2}


def test_params_change_invalidates_everything(cfg, dataset):
    p = Pipeline(cfg, dataset)
    p.run(1)
    # 修改定点位宽 -> 参数指纹变化 -> 全部失效
    p.cfg = cfg.with_overrides(fixed_point_frac_bits=10)
    p.cache = BlockCache(p.cfg)
    r = p.run_batch(1)
    assert len(r["cache"]["recompute"]) == 15
    # 修改容差同样改变参数指纹
    assert params_fingerprint(cfg) != params_fingerprint(
        cfg.with_overrides(tol_output=0.01))
    # 修改块长亦然
    assert params_fingerprint(cfg) != params_fingerprint(
        cfg.with_overrides(query_block_len=4))


def test_out_deps_recorded(cfg, dataset):
    p = Pipeline(cfg, dataset)
    p.run(1)
    assert p.cache.deps, "应记录输出块引用的键块与状态集合"
    for deps in p.cache.deps.values():
        assert deps["key_blocks"] == [0, 1, 2]
        assert len(deps["state_keys"]) == 1
