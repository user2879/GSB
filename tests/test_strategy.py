from pe_diag.strategy import POLICY_TABLE, TierMetrics, select_strategy


def _metrics(n, dim, budget=100000, mem=64.0):
    return TierMetrics("t", n, dim, 8.0, 8, 64, budget, mem)


def test_small_prefers_full():
    d = select_strategy(_metrics(300, 32, mem=64.0))
    assert d["selected"] == "S_FULL"
    assert d["decision"]["block_method"] == "single"
    assert d["decision"]["active_dim"] == 32


def test_memory_pressure_forces_chunk_then_stream():
    # full batch exceeds memory but 75% batch fits -> chunk
    d = select_strategy(_metrics(500, 96, mem=0.3))
    assert d["selected"] == "S_CHUNK"
    assert d["decision"]["active_dim"] == 72
    # tight memory: only the single-key streaming path fits
    d2 = select_strategy(_metrics(800, 192, mem=0.5))
    assert d2["selected"] == "S_STREAM"
    assert d2["decision"]["block_method"] == "stream"
    assert d2["decision"]["layer_precision"] == "float32"


def test_audit_record_contains_scores_and_metrics():
    d = select_strategy(_metrics(10, 16))
    assert len(d["candidate_scores"]) == len(POLICY_TABLE)
    for row in d["candidate_scores"]:
        assert "eligible" in row and "score" in row
        assert row["dim_range"] == [0, row["active_dim"]]
    assert d["tier_metrics"]["dim"] == 16


def test_tier_switch_changes_compute_path():
    small = select_strategy(_metrics(300, 32, mem=64.0))["decision"]
    large = select_strategy(_metrics(800, 192, mem=0.5))["decision"]
    assert (small["active_dim"], small["block_method"]) != \
           (large["active_dim"], large["block_method"])
