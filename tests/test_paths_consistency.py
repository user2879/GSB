"""三路径逐字节一致、批次统计与默认路径选择确定性。"""
import numpy as np

from quota_arbiter import ALL_PATHS
from quota_arbiter.consistency import (
    final_state_matrix,
    outcome_matrix,
    queue_trace_matrix,
)
from quota_arbiter.generator import generate_dataset
from quota_arbiter.loader import load_requests, load_spec
from quota_arbiter.paths import run_path
from quota_arbiter.runner import run_all_paths
from quota_arbiter.selector import select_default_path


def _load(workdir):
    tenants, pools = load_spec(f"{workdir}/spec.json")
    requests = load_requests(f"{workdir}/requests.ndjson", tenants, pools)
    return tenants, pools, requests


def test_three_paths_byte_identical(tmp_path):
    info = generate_dataset(str(tmp_path), tenant_count=12, pool_count=4, request_count=800)
    tenants, pools, requests = _load(info["spec"].rsplit("/", 1)[0])
    results, report = run_all_paths(tenants, pools, requests)
    assert report["identical"] is True
    base = outcome_matrix(results[ALL_PATHS[0]])
    for name in ALL_PATHS[1:]:
        assert np.array_equal(base, outcome_matrix(results[name]))
        assert np.array_equal(
            final_state_matrix(results[ALL_PATHS[0]]),
            final_state_matrix(results[name]),
        )
        assert np.array_equal(
            queue_trace_matrix(results[ALL_PATHS[0]]),
            queue_trace_matrix(results[name]),
        )


def test_merged_batch_stats_deterministic(tmp_path):
    info = generate_dataset(str(tmp_path), tenant_count=6, pool_count=2, request_count=300)
    tenants, pools, requests = _load(info["spec"].rsplit("/", 1)[0])
    result = run_path("merged", tenants, pools, requests)
    # 合并次序在每个批次中一致且为确定序列
    orders = {tuple(b["merge_order"]) for b in result.batches}
    assert len(orders) == 1
    total_requests = sum(b["request_count"] for b in result.batches)
    assert total_requests == len(requests)
    assert sum(b["skipped_count"] for b in result.batches) >= 0


def test_selector_deterministic_and_threshold_flag():
    s1 = select_default_path(8, 300, 3)
    s2 = select_default_path(8, 300, 3)
    assert s1["selected"] == s2["selected"]
    assert set(s1["estimates"]) == set(ALL_PATHS)
    assert isinstance(s1["threshold_influenced"], bool)
    assert s1["reasons"]
    # 规模变化会改变选择空间，但结论始终确定
    big = select_default_path(300, 50000, 16)
    assert big["selected"] in ALL_PATHS
    assert big == select_default_path(300, 50000, 16)
