from fablock.pipeline import Pipeline


def _recompute_blocks(report, kind):
    return {entry["block"] for entry in report["cache"]["recompute"]
            if entry["kind"] == kind}


def _reuse_blocks(report, kind):
    return {tuple(e["block"]) if isinstance(e["block"], list) else e["block"]
            for e in report["cache"]["reuse"] if e["kind"] == kind}


def test_burst_blocks_recomputed_stable_reused(cfg, dataset):
    p = Pipeline(cfg, dataset)
    r0, r1 = p.run(2)
    # 批次 0：全量计算
    assert len(r0["cache"]["recompute"]) == 15  # 9 score + 3 state + 3 out
    # 批次 1：仅突发查询块 1 及其键块乘积进入重算集合
    assert _recompute_blocks(r1, "score") == {(1, 0), (1, 1), (1, 2)}
    assert _recompute_blocks(r1, "state") == {1}
    assert _recompute_blocks(r1, "out") == {1}
    # 稳定查询块产物直接复用并记录命中
    assert _reuse_blocks(r1, "score") == {(0, 0), (0, 1), (0, 2),
                                          (2, 0), (2, 1), (2, 2)}
    assert _reuse_blocks(r1, "state") == {0, 2}
    assert _reuse_blocks(r1, "out") == {0, 2}


def test_burst_list_change_invalidates_only_new_blocks(cfg, dataset):
    p = Pipeline(cfg, dataset)
    p.run(2)
    dataset.set_burst_query_blocks((2,))
    r2 = p.run_batch(2)
    # 新增突发块 2 失效重算；旧突发块 1 恢复稳定并复用已有产物
    assert _recompute_blocks(r2, "score") == {(2, 0), (2, 1), (2, 2)}
    assert _recompute_blocks(r2, "state") == {2}
    assert (1, 0) in _reuse_blocks(r2, "score")
    assert 1 in _reuse_blocks(r2, "state")
    assert 1 in _reuse_blocks(r2, "out")
