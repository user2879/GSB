from __future__ import annotations

import unittest
from pathlib import Path

from isolation_lab.models import SchedulerParams
from isolation_lab.scheduler import Engine, compare_run_outcomes
from isolation_lab.scripts import load_script, normalize_document
from isolation_lab.storage import Storage


ROOT = Path(__file__).resolve().parent.parent


def run_document(document: dict, limit: int = 2):
    script = normalize_document(document)
    storage = Storage(":memory:")
    result = Engine(storage, SchedulerParams(global_concurrency_limit=limit)).run(script)
    return result, storage


class SchedulerTests(unittest.TestCase):
    def test_normal_concurrent_editing_commit_versions(self) -> None:
        result, storage = run_document(load_script(ROOT / "samples" / "current-v3.json").to_dict(include_source=False))
        self.assertEqual(result.global_data_version, 3)
        self.assertEqual(result.final_rows["acct:alpha"], {"value": 11, "version": 1})
        self.assertEqual(result.final_rows["acct:beta"], {"value": 22, "version": 2})
        self.assertEqual(result.final_rows["acct:gamma"], {"value": 33, "version": 3})
        self.assertEqual([commit.session_id for commit in result.commits], ["N1", "N2", "N3"])
        self.assertEqual(result.lock_state["rows"], [])
        self.assertEqual(storage.integrity_check(), "ok")
        storage.close()

    def test_global_concurrency_limit_is_never_exceeded(self) -> None:
        script = load_script(ROOT / "samples" / "current-v3.json")
        storage = Storage(":memory:")
        result = Engine(storage, SchedulerParams(global_concurrency_limit=2)).run(script)
        active = 0
        maximum = 0
        for event in result.trace:
            if event.event_type == "session.start":
                active += 1
                maximum = max(maximum, active)
            elif event.event_type in ("session.commit", "session.abort"):
                active -= 1
            self.assertLessEqual(active, 2)
        self.assertEqual(maximum, 2)
        starts = [event.payload["start_seq"] for event in result.trace if event.event_type == "session.start"]
        self.assertEqual(starts, [1, 2, 3])
        storage.close()

    def test_lock_wait_budget_timeout_rolls_back_waiter(self) -> None:
        script = load_script(ROOT / "samples" / "lock-timeout-v3.json")
        storage = Storage(":memory:")
        result = Engine(storage, SchedulerParams(global_concurrency_limit=2)).run(script)
        outcomes = result.outcome_by_session()
        self.assertEqual(outcomes["W"].outcome, "E_LOCKTIMEOUT")
        self.assertEqual(outcomes["W"].business_code, "E_LOCKTIMEOUT")
        self.assertEqual(outcomes["W"].wait_used_ms, 3)
        self.assertEqual(outcomes["W"].committed_data_version, None)
        self.assertEqual(outcomes["H"].outcome, "COMMITTED")
        self.assertEqual(result.final_rows["acct:locked"], {"value": 100, "version": 1})
        self.assertEqual(result.global_data_version, 1)
        wait_ops = [
            operation
            for operation in outcomes["W"].operations
            if operation.status == "failed"
        ]
        self.assertEqual(len(wait_ops), 1)
        self.assertEqual(wait_ops[0].wait_ms, 3)
        self.assertIsNone(wait_ops[0].wait.granted_virtual_ms)
        self.assertEqual(result.lock_state["rows"], [])
        storage.close()

    def test_operation_timeout_wins_when_shorter_than_budget(self) -> None:
        document = {
            "format_version": 3,
            "script_id": "operation-timeout",
            "initial_rows": {"r": 0, "other": 0},
            "sessions": [
                {
                    "session_id": "H",
                    "isolation_level": "read_committed",
                    "lock_wait_budget_ms": 100,
                    "concurrency": 1,
                    "operation_timeout_ms": 100,
                    "operations": [
                        {"op_no": 1, "type": "write", "row_key": "r", "value": 1},
                        {"op_no": 2, "type": "read", "row_key": "other"},
                        {"op_no": 3, "type": "read", "row_key": "other"},
                    ],
                },
                {
                    "session_id": "W",
                    "isolation_level": "read_committed",
                    "lock_wait_budget_ms": 100,
                    "concurrency": 1,
                    "operation_timeout_ms": 2,
                    "operations": [{"op_no": 1, "type": "write", "row_key": "r", "value": 9}],
                },
            ],
        }
        result, storage = run_document(document)
        timeout_events = [
            event for event in result.trace if event.event_type == "session.lock_timeout"
        ]
        self.assertEqual(timeout_events[0].payload["reason"], "operation_timeout")
        self.assertEqual(result.outcome_by_session()["W"].business_code, "E_LOCKTIMEOUT")
        storage.close()

    def test_deadlock_victim_is_deterministic_and_locks_are_clean(self) -> None:
        script = load_script(ROOT / "samples" / "deadlock-v3.json")
        victims = []
        final_states = []
        for _ in range(5):
            storage = Storage(":memory:")
            result = Engine(storage, SchedulerParams(global_concurrency_limit=2)).run(script)
            victim = next(
                outcome.session_id
                for outcome in result.outcomes
                if outcome.business_code == "E_DEADLOCK"
            )
            victims.append(victim)
            final_states.append((result.final_rows, result.global_data_version))
            self.assertEqual(result.lock_state["rows"], [])
            deadlock_events = [
                event for event in result.trace if event.event_type == "deadlock.detected"
            ]
            self.assertEqual(deadlock_events[0].payload["victim"], victim)
            self.assertEqual(deadlock_events[0].payload["arbitration_rule"], "min(start_seq, session_id)")
            storage.close()
        self.assertEqual(victims, ["D1"] * 5)
        self.assertEqual(final_states, [final_states[0]] * 5)

    def test_repeatable_read_vs_read_committed_semantics(self) -> None:
        def document(isolation: str) -> dict:
            return {
                "format_version": 3,
                "script_id": f"isolation-{isolation}",
                "initial_rows": {"r": 0},
                "sessions": [
                    {
                        "session_id": "R",
                        "isolation_level": isolation,
                        "lock_wait_budget_ms": 20,
                        "concurrency": 1,
                        "operation_timeout_ms": 20,
                        "operations": [
                            {"op_no": 1, "type": "read", "row_key": "r"},
                            {"op_no": 2, "type": "read", "row_key": "r"},
                        ],
                    },
                    {
                        "session_id": "W",
                        "isolation_level": "read_committed",
                        "lock_wait_budget_ms": 20,
                        "concurrency": 1,
                        "operation_timeout_ms": 20,
                        "operations": [
                            {"op_no": 1, "type": "write", "row_key": "r", "value": 9}
                        ],
                    },
                ],
            }

        rr_result, rr_storage = run_document(document("repeatable_read"))
        rr_outcome = rr_result.outcome_by_session()["R"]
        self.assertEqual(
            [(op.value, op.data_version) for op in rr_outcome.operations],
            [(0, 0), (0, 0)],
        )
        self.assertEqual(rr_result.final_rows["r"], {"value": 9, "version": 1})
        rr_storage.close()

        rc_result, rc_storage = run_document(document("read_committed"))
        rc_outcome = rc_result.outcome_by_session()["R"]
        self.assertEqual(
            [(op.value, op.data_version) for op in rc_outcome.operations],
            [(0, 0), (9, 1)],
        )
        rc_storage.close()

    def test_legacy_versions_upgrade_to_identical_outcomes(self) -> None:
        source_script = load_script(ROOT / "samples" / "legacy-v1.json")
        upgraded_document = source_script.to_dict(include_source=True)
        source_storage = Storage(":memory:")
        source_result = Engine(source_storage, SchedulerParams(global_concurrency_limit=2)).run(source_script)
        upgraded_script = normalize_document(upgraded_document)
        upgraded_storage = Storage(":memory:")
        upgraded_result = Engine(upgraded_storage, SchedulerParams(global_concurrency_limit=2)).run(upgraded_script)
        self.assertEqual(source_result.source_version, 1)
        self.assertEqual(upgraded_result.source_version, 1)
        self.assertEqual(source_result.run_id, upgraded_result.run_id)
        self.assertEqual(compare_run_outcomes(source_result, upgraded_result), [])
        source_storage.close()
        upgraded_storage.close()

        v2_script = load_script(ROOT / "samples" / "legacy-v2.json")
        v2_upgraded = normalize_document(v2_script.to_dict(include_source=True))
        v2_a_storage = Storage(":memory:")
        v2_b_storage = Storage(":memory:")
        first = Engine(v2_a_storage, SchedulerParams(global_concurrency_limit=2)).run(v2_script)
        second = Engine(v2_b_storage, SchedulerParams(global_concurrency_limit=2)).run(v2_upgraded)
        self.assertEqual(compare_run_outcomes(first, second), [])
        self.assertEqual(first.outcome_by_session()["V2W"].wait_used_ms, 2)
        self.assertEqual(second.outcome_by_session()["V2W"].wait_used_ms, 2)
        v2_a_storage.close()
        v2_b_storage.close()


if __name__ == "__main__":
    unittest.main()
