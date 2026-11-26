import threading

from arg_review.diagnostics import run_injection_suite
from arg_review.engine import ReviewEngine, projections_equal
from arg_review.samplegen import generate_sample_tree
from arg_review.scheduler import BudgetScheduler
from arg_review.snapshots import SnapshotStore
from arg_review.state import load_workspace


def test_budget_order_exceeded_pending_and_next_batch_priority():
    scheduler = BudgetScheduler(
        {
            "budget": {
                "max_workers": 3,
                "global_arg_score_points": 5,
                "per_event_arg_quota": 3,
                "max_trigger_matches_per_batch": 2,
            }
        }
    )
    scheduler.enqueue_argument_units(
        [
            {"event_id": "E2", "role": "x", "candidate_count": 2},
            {"event_id": "E1", "role": "b", "candidate_count": 1},
            {"event_id": "E1", "role": "a", "candidate_count": 2},
            {"event_id": "E3", "role": "a", "candidate_count": 1},
        ]
    )
    assert [u["unit_key"] for u in scheduler.arg_queue] == [
        "E1::a",
        "E1::b",
        "E2::x",
        "E3::a",
    ]
    first = scheduler.run_argument_batch(lambda unit: {"ok": unit["unit_key"]})
    assert first["admitted_unit_keys"] == ["E1::a", "E1::b", "E2::x"]
    assert first["budget_exceeded"]["code"] == "E_ARG_BUDGET_EXCEEDED"
    assert [u["unit_key"] for u in first["pending_units"]] == ["E3::a"]
    assert first["pending_units"][0]["code"] == "E_ARG_PENDING"
    assert first["peak_concurrency"] <= 3

    scheduler.global_limit = 10
    second = scheduler.run_argument_batch(lambda unit: {"ok": True})
    assert second["admitted_unit_keys"] == ["E3::a"]
    assert second["pending_units"] == []


def test_thread_pool_never_exceeds_configured_concurrency():
    scheduler = BudgetScheduler({"budget": {"max_workers": 2}})
    barrier = threading.Barrier(2, timeout=2)

    def job(unit):
        barrier.wait()
        return True

    scheduler.enqueue_argument_units(
        [
            {"event_id": f"E{i}", "role": "r", "candidate_count": 1}
            for i in range(4)
        ]
    )
    report = scheduler.run_argument_batch(job)
    assert report["admitted_count"] == 4
    assert report["peak_concurrency"] == 2
    assert report["concurrency_limit_respected"] is True


def test_trigger_batch_capacity_and_pending_order():
    scheduler = BudgetScheduler(
        {"budget": {"max_workers": 2, "max_trigger_matches_per_batch": 2}}
    )
    scheduler.enqueue_trigger_events(["E4", "E2", "E1", "E3", "E0"])
    first = scheduler.run_trigger_batch(lambda event_id: event_id)
    assert first["processed_event_ids"] == ["E0", "E1"]
    assert first["pending_event_ids"] == ["E2", "E3", "E4"]
    second = scheduler.run_trigger_batch(lambda event_id: event_id)
    assert second["processed_event_ids"] == ["E2", "E3"]
    assert second["pending_event_ids"] == ["E4"]


def test_snapshot_atomic_replace_and_staging_recovery(tmp_path):
    root = tmp_path / "snapshots"
    store = SnapshotStore(root)
    (root / "tmp-stale-interrupted").mkdir(parents=True)
    recovery = store.recover()
    assert recovery["decision"] == "full_rebuild_required"
    assert recovery["discarded_temp_dirs"] == ["tmp-stale-interrupted"]

    first = store.commit({"validation": {"ok": 1}}, 1, "hash-a")
    recovered = store.recover()
    assert recovered["decision"] == "use_current_snapshot"
    assert recovered["commit"]["global_version"] == 1

    second = store.commit({"validation": {"ok": 2}}, 2, "hash-b")
    assert second["version_changed"] is True
    assert second["snapshot_id"] != first["snapshot_id"]
    assert store.current_link.resolve().name == second["snapshot_id"]


def test_all_injections_hit_and_isolate_other_events(tmp_path):
    sample_root = tmp_path / "samples"
    generate_sample_tree(sample_root)
    state = load_workspace(sample_root)
    suite = run_injection_suite(state, tmp_path / "injections")
    assert suite["all_injections_hit"] is True
    assert suite["hit_count"] == 5
    by_id = {case["injection_id"]: case for case in suite["cases"]}
    assert len(by_id["INJ-OFFSET-001"]["surviving_decision_event_ids"]) == 6
    assert len(by_id["INJ-CONF-002"]["surviving_decision_event_ids"]) == 6
    assert len(by_id["INJ-ROLE-004"]["surviving_decision_event_ids"]) == 6
    assert by_id["INJ-REL-003"]["hits"][0]["business_code"] == "E_SCHEMA_INVALID"
    assert by_id["INJ-CHAIN-005"]["hits"][0]["stage_code"] == (
        "STAGE_REVISION_CHAIN_INVALID"
    )


def test_persistent_state_resumes_after_injection_removed(tmp_path):
    sample_root = tmp_path / "samples"
    generate_sample_tree(sample_root)
    state = load_workspace(sample_root)
    state_dir = tmp_path / "resume"
    injection = next(
        item for item in state.injections if item["injection_id"] == "INJ-CONF-002"
    )
    failed = ReviewEngine(state, state_dir, injection=injection).run()
    assert failed["report"]["status"] == "partial"
    assert "decision:E003" not in failed["manifest"]

    resumed = ReviewEngine(state, state_dir).run()
    clean = ReviewEngine(state, tmp_path / "clean").run()
    assert resumed["report"]["status"] == "ok"
    assert "decision:E003" in resumed["manifest"]
    assert projections_equal(resumed["projection"], clean["projection"])
