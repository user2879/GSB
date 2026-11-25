from __future__ import annotations

from pathlib import Path

from inference_attribution.engine import execute_plan
from inference_attribution.invalidation import diagnose_invalidation
from inference_attribution.jsonio import canonical_json
from inference_attribution.ledger import Ledger
from inference_attribution.planner import build_plan
from inference_attribution.spec import load_merge_rules, load_requests, load_workers
from helpers import SAMPLES, dynamic_workers


def _load_world(workers_path: Path):
    workers = load_workers(workers_path)
    requests = load_requests(SAMPLES / "requests.json", workers)
    rules = load_merge_rules(SAMPLES / "merge_rules.json")
    plan = build_plan(workers, requests)
    return workers, requests, rules, plan


def test_parameter_version_invalidates_only_owning_worker(tmp_path: Path):
    old_path = tmp_path / "workers-old.json"
    old_document = dynamic_workers(old_path)
    ports = (
        old_document["workers"][0]["port"],
        old_document["workers"][1]["port"],
    )
    new_path = tmp_path / "workers-new.json"
    dynamic_workers(
        new_path,
        ports=ports,
        worker_set_version=4,
        alpha_param_version=3,
    )
    old_workers, old_requests, rules, old_plan = _load_world(old_path)
    new_workers, new_requests, _, new_plan = _load_world(new_path)
    report = diagnose_invalidation(
        old_workers, old_plan, new_workers, new_plan
    )
    old_alpha = {
        batch.batch_id for batch in old_plan.batches if batch.worker_id == "worker-alpha"
    }
    old_beta = {
        batch.batch_id for batch in old_plan.batches if batch.worker_id == "worker-beta"
    }
    assert set(report.direct_batch_ids) == old_alpha
    assert report.indirect_batch_ids == ()
    assert set(report.unchanged_batch_ids) == old_beta
    assert set(report.indirect_request_ids) == {
        "req-001",
        "req-002",
        "req-004",
        "req-006",
    }

    ledger_path = tmp_path / "ledger.sqlite"
    with Ledger(ledger_path) as ledger:
        execute_plan(
            plan=old_plan,
            request_set=old_requests,
            worker_set=old_workers,
            rules=rules,
            ledger=ledger,
        )
        before = {row["batch_id"]: row for row in ledger.rows()}
        beta_id = min(old_beta)
        beta_before = canonical_json(before[beta_id])
        alpha_before = {
            batch_id: before[batch_id]["receipt_digest"] for batch_id in old_alpha
        }

        invalid = report.invalid_new_plan_batch_ids(new_plan)
        outcome = execute_plan(
            plan=new_plan,
            request_set=new_requests,
            worker_set=new_workers,
            rules=rules,
            ledger=ledger,
            invalid_batch_ids=invalid,
            invalidation_report=report,
        )
        after = {row["batch_id"]: row for row in ledger.rows()}
        assert canonical_json(after[beta_id]) == beta_before
        assert set(outcome.skipped_batch_ids) == old_beta
        for batch_id in old_alpha:
            assert after[batch_id]["receipt_digest"] != alpha_before[batch_id]
            assert after[batch_id]["invalidation_trace"]["direct"] is True
            assert after[batch_id]["param_version_seq"] == 3


def test_batch_size_change_marks_old_and_new_batches(tmp_path: Path):
    old_path = tmp_path / "workers-old.json"
    old_document = dynamic_workers(old_path)
    new_path = tmp_path / "workers-repartitioned.json"
    dynamic_workers(
        new_path,
        ports=(
            old_document["workers"][0]["port"],
            old_document["workers"][1]["port"],
        ),
        worker_set_version=5,
        beta_batch_size=1,
    )
    old_workers, _, _, old_plan = _load_world(old_path)
    new_workers, _, _, new_plan = _load_world(new_path)
    report = diagnose_invalidation(
        old_workers, old_plan, new_workers, new_plan
    )
    old_beta = {
        batch.batch_id for batch in old_plan.batches if batch.worker_id == "worker-beta"
    }
    new_beta = {
        batch.batch_id for batch in new_plan.batches if batch.worker_id == "worker-beta"
    }
    alpha = {
        batch.batch_id for batch in new_plan.batches if batch.worker_id == "worker-alpha"
    }
    assert set(report.direct_batch_ids) == old_beta | new_beta
    assert set(report.unchanged_batch_ids) == alpha
    assert set(report.indirect_request_ids) == {"req-003", "req-005"}
    assert report.worker_changes == (
        {
            "worker_id": "worker-beta",
            "changed_fields": ["batch_size"],
        },
    )
