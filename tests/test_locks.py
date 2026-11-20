from __future__ import annotations

import unittest

from isolation_lab.locks import LockManager


class LockManagerTests(unittest.TestCase):
    def test_shared_locks_are_compatible_and_exclusive_waits(self) -> None:
        manager = LockManager()
        first = manager.acquire(
            session_id="A",
            row_key="r",
            mode="S",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=100,
            operation_timeout_ms=100,
        )
        second = manager.acquire(
            session_id="B",
            row_key="r",
            mode="S",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=100,
            operation_timeout_ms=100,
        )
        self.assertEqual(first.status, "granted")
        self.assertEqual(second.status, "granted")
        self.assertNotEqual(first.grant_seq, second.grant_seq)

        third = manager.acquire(
            session_id="C",
            row_key="r",
            mode="X",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=100,
            operation_timeout_ms=100,
        )
        self.assertEqual(third.status, "waiting")
        self.assertEqual(third.blocked_by, ("A", "B"))
        self.assertEqual(third.request.request_seq, 1)

    def test_fifo_queue_is_stable_and_does_not_bypass(self) -> None:
        manager = LockManager()
        manager.acquire(
            session_id="OWNER",
            row_key="r",
            mode="X",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=100,
            operation_timeout_ms=100,
        )
        wait_x = manager.acquire(
            session_id="W1",
            row_key="r",
            mode="X",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=100,
            operation_timeout_ms=100,
        )
        wait_s = manager.acquire(
            session_id="W2",
            row_key="r",
            mode="S",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=100,
            operation_timeout_ms=100,
        )
        self.assertEqual(wait_x.request.request_seq, 1)
        self.assertEqual(wait_s.request.request_seq, 2)
        queue = manager.state_dict()["rows"][0]["queue"]
        self.assertEqual([item["request_seq"] for item in queue], [1, 2])

        manager.release_session("OWNER", 1, "commit")
        notifications = manager.drain_notifications()
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0].request.session_id, "W1")
        self.assertEqual(manager.owner_count(), 1)
        self.assertEqual(manager.waiter_count(), 1)

        manager.release_session("W1", 2, "commit")
        notifications = manager.drain_notifications()
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0].request.session_id, "W2")
        self.assertEqual(manager.entry_count(), 1)

        manager.release_session("W2", 3, "commit")
        self.assertEqual(manager.entry_count(), 0)

    def test_lock_events_record_wait_basis_and_queue(self) -> None:
        events: list[tuple[str, dict]] = []

        def callback(event_type: str, session_id: str | None, op_no: int | None, payload: dict, now: int) -> None:
            events.append((event_type, payload))

        manager = LockManager(callback)
        manager.acquire(
            session_id="A",
            row_key="r",
            mode="X",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=100,
            operation_timeout_ms=100,
        )
        manager.acquire(
            session_id="B",
            row_key="r",
            mode="S",
            isolation_level="read_committed",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=5,
            operation_timeout_ms=4,
        )
        wait_event = next(payload for event_type, payload in events if event_type == "lock.waiting")
        self.assertEqual(wait_event["blocked_by"], ["A"])
        self.assertEqual(wait_event["wait_limit_ms"], 4)
        self.assertEqual(wait_event["compatibility_basis"], "mode_conflict")
        self.assertEqual(wait_event["queue"][0]["request_seq"], 1)

    def test_upgrade_waits_for_other_shared_holders(self) -> None:
        manager = LockManager()
        manager.acquire(
            session_id="A",
            row_key="r",
            mode="S",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=50,
            operation_timeout_ms=50,
        )
        manager.acquire(
            session_id="B",
            row_key="r",
            mode="S",
            isolation_level="repeatable_read",
            now=0,
            op_no=1,
            op_started_virtual_ms=0,
            remaining_budget_ms=50,
            operation_timeout_ms=50,
        )
        upgrade = manager.acquire(
            session_id="A",
            row_key="r",
            mode="X",
            isolation_level="repeatable_read",
            now=1,
            op_no=2,
            op_started_virtual_ms=1,
            remaining_budget_ms=50,
            operation_timeout_ms=50,
        )
        self.assertEqual(upgrade.status, "waiting")
        self.assertEqual(upgrade.blocked_by, ("B",))
        manager.release_session("B", 2, "commit")
        notifications = manager.drain_notifications()
        self.assertEqual(notifications[0].request.session_id, "A")
        owner = manager.state_dict()["rows"][0]["owners"][0]
        self.assertEqual(owner["session_id"], "A")
        self.assertEqual(owner["mode"], "X")


if __name__ == "__main__":
    unittest.main()
