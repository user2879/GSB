from fablock.config import OperatorConfig
from fablock.errors import Code
from fablock.injection import Injection, apply_load_time
from fablock.pipeline import Pipeline
from fablock.scheduler import BudgetScheduler


def test_node_budget_exhausted(cfg, dataset):
    # 单查询块配额 < 单块所需点数 -> E_NODE_BUDGET_EXHAUSTED
    small = cfg.with_overrides(point_budget_per_batch=3 * 64 - 1)
    p = Pipeline(small, dataset)
    r = p.run(1)[0]
    codes = {e["code"] for e in r["budget_events"]}
    assert Code.E_NODE_BUDGET_EXHAUSTED in codes


def test_global_budget_pending_and_next_batch_priority(cfg, dataset):
    # 总需求 9*64=576 点；预算 575 使配额 192 足够但全局预算差 1 点
    limited = cfg.with_overrides(point_budget_per_batch=9 * 64 - 1)
    p = Pipeline(limited, dataset)
    r0, r1 = p.run(2)
    codes = {e["code"] for e in r0["budget_events"]}
    assert Code.E_POINT_BUDGET_PENDING in codes
    assert r0["pending_tasks"], "预算耗尽应保留待办队列"
    # 下一批优先处理待办：批次 1 的重算集合即上批待办
    carried = {(t["query_block"], t["key_block"]) for t in r0["pending_tasks"]}
    done_b1 = {(e["block"][0], e["block"][1])
               for e in r1["cache"]["recompute"] if e["kind"] == "score"}
    assert carried <= done_b1  # 待办在下一批被优先执行


def test_budget_below_min_block_injection(cfg, dataset):
    inj = [Injection(id="INJ-B", type="budget_below_min_block")]
    tensors, meta, cfg_b, hits = apply_load_time(
        dataset.tensors_for_batch(0), dataset.meta, cfg, inj)
    assert hits and cfg_b.point_budget_per_batch == 63
    p = Pipeline(cfg, dataset, injections=inj)
    r = p.run(1)[0]
    assert r["injection_hits"][0]["id"] == "INJ-B"
    assert any(e["code"] == Code.E_NODE_BUDGET_EXHAUSTED
               for e in r["budget_events"])


def test_max_concurrency_within_limit(cfg, dataset):
    p = Pipeline(cfg, dataset)
    r = p.run(1)[0]
    assert 1 <= r["peak_concurrency"] <= cfg.thread_pool_workers


def test_scheduler_stable_order(cfg):
    sched = BudgetScheduler(cfg, num_query_blocks=3, num_key_blocks=3)
    tasks = sched.plan([(2, 0), (0, 1), (1, 0), (0, 0)])
    order = [(t.query_block, t.key_block) for t in tasks]
    assert order == sorted(order)  # 按查询块序号稳定顺序
