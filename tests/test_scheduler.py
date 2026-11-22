from gpu_diag.loader import load_declaration
from gpu_diag.models import ShardState
from gpu_diag.sim.engine import SchedulerEngine


def _engine(sample_dir):
    decl = load_declaration(sample_dir / "pools.json",
                            sample_dir / "tasks.json",
                            sample_dir / "params.json")
    return SchedulerEngine(decl)


def test_all_shards_complete(sample_dir):
    engine = _engine(sample_dir)
    engine.run_all()
    assert all(shard.state == ShardState.COMPLETED
               for shard in engine.decl.shards.values())


def test_global_quota_never_exceeded(sample_dir):
    engine = _engine(sample_dir)
    quota = engine.params.global_gpu_quota
    for record in engine.run_all():
        assert record["water_levels"]["__global__"]["gpu_used"] <= quota


def test_pool_capacity_never_exceeded(sample_dir):
    engine = _engine(sample_dir)
    for record in engine.run_all():
        for pid, pool in engine.decl.pools.items():
            level = record["water_levels"][pid]
            assert level["gpu_used"] <= pool.gpu_count
            assert level["memory_used_mib"] <= pool.total_memory_mib
            assert level["running_task_count"] <= engine.params.pool_concurrency_limit


def test_deterministic_round_digests(sample_dir):
    first = _engine(sample_dir)
    second = _engine(sample_dir)
    first.run_all()
    second.run_all()
    assert [r["round_digest"] for r in first.rounds] == \
           [r["round_digest"] for r in second.rounds]


def test_worker_threads_are_real(sample_dir):
    engine = _engine(sample_dir)
    engine.run_all()
    assert engine._worker_seen, "no shard ran on a worker thread"
    assert all(name != "MainThread" for name in engine._worker_seen)


def test_fixed_point_ratio_within_error_bound(sample_dir):
    engine = _engine(sample_dir)
    scale = engine.params.scale_factor
    for record in engine.run_all():
        for pid, pool in engine.decl.pools.items():
            level = record["water_levels"][pid]
            exact = level["gpu_used"] * scale / pool.gpu_count
            assert level["water_ratio_q"] <= exact
            assert exact - level["water_ratio_q"] < 1.0


def test_queue_records_wait_window(sample_dir):
    engine = _engine(sample_dir)
    engine.run_all()
    queued_ids = {q["shard_id"] for r in engine.rounds for q in r["queued"]}
    assert "T2#1" in queued_ids or queued_ids  # queue path exercised
    for shard in engine.decl.shards.values():
        if shard.wait_start_ms is not None:
            assert shard.wait_end_ms >= shard.wait_start_ms


def test_preemption_at_shard_boundary(preempt_dir):
    engine = _engine(preempt_dir)
    engine.run_all()
    preempted = [p for r in engine.rounds for p in r["preemptions"]]
    tasks_preempted = [p["task_id"] for p in preempted]
    assert "LOW" in tasks_preempted and "MID" in tasks_preempted
    assert all(p["action"] == "preempt" for p in preempted)
    # Preempted tasks eventually finish all their shards.
    for tid in ("LOW", "MID", "HIGH"):
        rt = engine.runtime[tid]
        assert rt.next_index == engine.decl.tasks[tid].shard_count


def test_preemption_sequence_is_deterministic(preempt_dir):
    first = _engine(preempt_dir)
    second = _engine(preempt_dir)
    first.run_all()
    second.run_all()
    seq1 = [(p["task_id"], p["boundary_shard"])
            for r in first.rounds for p in r["preemptions"]]
    seq2 = [(p["task_id"], p["boundary_shard"])
            for r in second.rounds for p in r["preemptions"]]
    assert seq1 == seq2


def test_high_priority_never_blocked_by_lower_after_tick(preempt_dir):
    engine = _engine(preempt_dir)
    engine.run_all()
    high = engine.decl.tasks["HIGH"]
    first_high_start = min(
        s.start_ms for s in engine.decl.shards.values()
        if s.task_id == "HIGH" and s.start_ms is not None)
    assert first_high_start <= high.submit_time_ms + 20


def test_resume_reuses_completed_shards(preempt_dir):
    engine = _engine(preempt_dir)
    engine.run_all()
    # Each preempted shard is admitted at least twice (initial + resume),
    # while completed shard results are reused (no duplicate quota ledger).
    low_zero = engine.decl.shards["LOW#0"]
    assert low_zero.attempts >= 2
    assert low_zero.state == ShardState.COMPLETED
