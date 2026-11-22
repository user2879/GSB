"""并发调度确定性与元素预算 (业务要求 8)."""

from __future__ import annotations

import json

from cqdiag.errors import E_BUDGET_EXCEEDED
from cqdiag.pipeline import Pipeline
from cqdiag.store import SnapshotStore

from conftest import make_config, make_samples


def _normalize(result):
    return json.loads(json.dumps({
        "columns": result.columns,
        "stratify": result.stratify,
        "failures": result.failures.to_list(),
        "budget_log": result.budget_log,
    }, sort_keys=True))


def test_parallel_matches_single_thread(tmp_path):
    config1 = make_config(max_workers=1)
    config4 = make_config(max_workers=4)
    samples1 = make_samples(config1)
    samples4 = make_samples(config4)
    r1 = Pipeline(config1, samples1, SnapshotStore(tmp_path / "w1")).run()
    r4 = Pipeline(config4, samples4, SnapshotStore(tmp_path / "w4")).run()
    assert _normalize(r1) == _normalize(r4)


def test_budget_exceeded_deterministic(tmp_path):
    # 列长 6, 预算 14 -> 按列标识顺序扣减, t:2 起预算耗尽
    config1 = make_config(max_workers=1, element_budget=14)
    config4 = make_config(max_workers=4, element_budget=14)
    r1 = Pipeline(config1, make_samples(config1), SnapshotStore(tmp_path / "b1")).run()
    r4 = Pipeline(config4, make_samples(config4), SnapshotStore(tmp_path / "b4")).run()
    budget_failures = [f for f in r1.failures.to_list()
                       if f["code"] == E_BUDGET_EXCEEDED]
    assert [f["context"]["column"] for f in budget_failures] == ["t:2", "t:3"]
    assert r1.budget_spent == 12
    assert sorted(r1.columns.keys()) == ["t:0", "t:1"]
    assert _normalize(r1) == _normalize(r4)


def test_budget_log_sorted_by_column(tmp_path):
    config = make_config(max_workers=4)
    result = Pipeline(config, make_samples(config), SnapshotStore(tmp_path)).run()
    charged = [e["column"] for e in result.budget_log if e["action"] == "charged"]
    assert charged == sorted(charged)
    assert result.budget_spent == 24
