from gpu_diag.loader import load_declaration
from gpu_diag.models import ShardState
from gpu_diag.sim.engine import SchedulerEngine
from gpu_diag.sim.propagation import (propagate_shard_invalidation,
                                      propagate_task_invalidation)


def test_shard_invalidation_is_stable_and_order_independent(sample_dir):
    decl = load_declaration(sample_dir / "pools.json",
                            sample_dir / "tasks.json",
                            sample_dir / "params.json")
    ordered_a, rationale_a = propagate_shard_invalidation(decl, ["T1#0"])
    decl_b = load_declaration(sample_dir / "pools.json",
                              sample_dir / "tasks.json",
                              sample_dir / "params.json")
    ordered_b, rationale_b = propagate_shard_invalidation(
        decl_b, ["T1#0"])
    assert ordered_a == ordered_b
    assert ordered_a[0] == "T1#0"
    assert "T1#1" in ordered_a and "T4#0" in ordered_a
    assert rationale_a and all("reason" in r for r in rationale_a)


def test_task_graph_includes_cross_task_shard_edges(sample_dir):
    decl = load_declaration(sample_dir / "pools.json",
                            sample_dir / "tasks.json",
                            sample_dir / "params.json")
    affected, _ = propagate_task_invalidation(decl, ["T1"])
    assert "T4" in affected


def test_failure_propagation_through_engine(sample_dir):
    decl = load_declaration(sample_dir / "pools.json",
                            sample_dir / "tasks.json",
                            sample_dir / "params.json")
    engine = SchedulerEngine(decl)
    engine.fail_inject = {"T1#1"}
    engine.run_all()
    assert engine.decl.shards["T1#1"].state == ShardState.FAILED
    assert engine.decl.shards["T1#2"].state == ShardState.AFFECTED
    assert engine.decl.shards["T4#1"].state == ShardState.AFFECTED
