from __future__ import annotations

from fcdiag.errors import E_ARTIFACT_BUDGET_PENDING, E_STAGE_BUDGET_EXCEEDED
from fcdiag.fixedpoint import FixedConfig
from fcdiag.models import Budget, Config, Policy
from fcdiag.runner import Runner
from fcdiag.scheduler import Scheduler, Task
from fcdiag.state import WorkState


class DummyDecision:
    reuse = False
    reason = "test"
    receipt_version = None
    digest_match = None
    degraded = False
    degradation_basis = None


def make_task(index: int, stage: str, points: int) -> Task:
    scope = "global" if index % 5 == 0 else f"sample:s{index:03d}"
    return Task(
        stage_index=index,
        stage_name=stage,
        scope=scope,
        artifact_id=f"{stage}:{index}",
        points=points,
        sort_key=f"{index:04d}:{scope}",
        params={},
        upstream=(),
        expected_param_version=0,
        candidate=None,
        decision=DummyDecision(),
        plan=lambda: object(),
    )


def test_pending_tasks_have_priority_and_stable_deduction_sequence():
    tasks = tuple(make_task(index, "standardize", 2) for index in range(10))
    scheduler = Scheduler(batch_points=5, stage_quotas={"standardize": 100})
    first = scheduler.run_batch(tasks)
    assert first.batch_points_used == 4
    assert [event["code"] for event in first.events].count(E_ARTIFACT_BUDGET_PENDING) == 8
    second = scheduler.run_batch((), carry_pending=first.pending)
    assert second.pending[0].artifact_id == "standardize:4"
    accepted = [outcome.task.artifact_id for outcome in second.outcomes if outcome.status == "REUSED"]
    assert accepted[:2] == ["standardize:2", "standardize:3"]


def test_stage_quota_aborts_stage_with_business_code():
    tasks = tuple(make_task(index, "bucketize", 3) for index in range(6))
    result = Scheduler(batch_points=100, stage_quotas={"bucketize": 4}).run_batch(tasks)
    assert E_STAGE_BUDGET_EXCEEDED in [event["code"] for event in result.events]
    assert "bucketize" in result.aborted_stages
    assert all(task.stage_name == "bucketize" for task in result.pending)


def test_runner_budget_config_produces_reviewable_events(context):
    state = context["state"]
    config = Config(
        state.config.fixed,
        Budget(3, {"standardize": 100, "bucketize": 100, "interact": 100, "train_view": 100}),
        state.config.policy,
        state.config.seed,
    )
    budget_state = state.copy()
    object.__setattr__(budget_state, "config", config)
    result = Runner(budget_state, context["receipts"][3]).run()
    assert result.events
    assert all(event["sort_key"] for event in result.events)
    assert [event["sort_key"] for event in result.events] == sorted(
        event["sort_key"] for event in result.events
    )


def test_concurrent_execution_preserves_order(context):
    left = Runner(context["state"], context["receipts"][3], max_workers=1).run()
    right = Runner(context["state"], context["receipts"][3], max_workers=4).run()
    assert [entry.sort_key for entry in left.ledger] == [entry.sort_key for entry in right.ledger]
    assert left.artifacts == right.artifacts
