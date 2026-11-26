from decchain.cache import ConclusionCache
from decchain.config import DecodeConfig
from decchain.pipeline import DecodePipeline
from decchain.scheduler import Scheduler, Task
from decchain.models import stable_task_key


def _scheduler(requests, config, budget, quota, cap=5, workers=1):
    cache = ConclusionCache()
    order = {r.request_id: i + 1 for i, r in
             enumerate(sorted(requests, key=lambda r: r.request_id))}
    pipeline = DecodePipeline(config, cache, order)
    return Scheduler(pipeline, requests, cache, budget, quota, cap, workers)


def test_stable_task_keys(sample_requests, config):
    scheduler = _scheduler(sample_requests, config, 1000, 1000)
    tasks = scheduler.build_tasks()
    keys = [t.key for t in tasks]
    assert keys == sorted(keys)
    assert keys[0] == stable_task_key(tasks[0].request_id, tasks[0].seq_id,
                                      tasks[0].step_index)


def test_tight_budget_retains_todos_then_drains(sample_requests, config):
    scheduler = _scheduler(sample_requests, config, budget=13, quota=1000)
    batches = scheduler.run_to_completion()
    assert len(batches) > 1
    assert batches[0].todos
    assert batches[-1].todos == []
    total = sum(len(b.outcomes) for b in batches)
    expected = sum(1 for r in sample_requests for _, _ in r.all_steps())
    assert total == expected


def test_budget_split_matches_single_run(sample_requests, config):
    single = _scheduler([_clone(r) for r in sample_requests], config,
                        10_000, 10_000)
    # Budget above the largest single-task cost (R003 steps have 8 tokens)
    # but small enough to force several batches.
    split = _scheduler([_clone(r) for r in sample_requests], config,
                       13, 10_000)
    single_rows = sorted(
        (o.task_key, o.conclusion.selected_token)
        for b in single.run_to_completion() for o in b.outcomes)
    split_rows = sorted(
        (o.task_key, o.conclusion.selected_token)
        for b in split.run_to_completion() for o in b.outcomes)
    assert single_rows == split_rows


def test_per_request_quota_aborts_batch(sample_requests, config):
    # Feasible per-batch quota: a couple of steps fit, then the batch aborts
    # and remaining work carries over to the next batch (quota resets).
    scheduler = _scheduler(sample_requests, config, budget=10_000, quota=12)
    batches = scheduler.run_to_completion()
    codes = {f.code for b in batches for f in b.failures}
    from decchain.errors import BizCode
    assert BizCode.REQUEST_QUOTA_EXCEEDED in codes
    assert len(batches) > 1
    assert batches[-1].todos == []


def test_recompute_cap(simple_request, config):
    scheduler = _scheduler([simple_request], config, 10_000, 10_000, cap=0)
    result = scheduler.run_batch()
    from decchain.errors import BizCode
    assert any(f.code == BizCode.STEP_RECOMPUTE_CAP for f in result.failures)
    assert result.todos


def _clone(request):
    from decchain.models import Request, CandidateSequence, Step
    return Request(
        request_id=request.request_id,
        sequences=[CandidateSequence(
            seq_id=s.seq_id, seed_token=s.seed_token, channel=s.channel,
            temperature=s.temperature,
            steps=[Step(index=st.index, logprobs=dict(st.logprobs))
                   for st in s.steps]) for s in request.sequences])
