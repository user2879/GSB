import unittest

from hashlot.clock import FakeClock
from hashlot.errors import BorrowCycleError, DeadlineError
from hashlot.slots import (
    SlotManager, ST_ACTIVE, ST_IDLE, ST_RELEASED,
)
from hashlot.spec import (
    Topology, PartitionSpec, Borrow,
)


def topology(global_max=3, partitions=None, borrows=()):
    if partitions is None:
        partitions = [
            PartitionSpec(i, max_connections=1, key_cap_per_connection=2,
                          idle_recycle_ms=10)
            for i in range(3)
        ]
    return Topology(len(partitions), tuple(partitions), tuple(borrows), global_max)


class SlotStateTest(unittest.TestCase):
    def test_establish_active_idle_released(self):
        clock = FakeClock()
        manager = SlotManager(topology(), clock)
        key = ((0x11, (1).to_bytes(8, "little")),)
        conn = manager.acquire(0, key)
        self.assertEqual(conn.state, ST_ACTIVE)
        manager.mark_idle(conn.conn_id)
        self.assertEqual(manager.connections[conn.conn_id].state, ST_IDLE)
        clock.advance(10)
        manager.recycle_locked()
        self.assertEqual(manager.connections[conn.conn_id].state, ST_RELEASED)

    def test_key_cap_forces_release(self):
        clock = FakeClock()
        manager = SlotManager(topology(), FakeClock())
        keys = [((0x11, i.to_bytes(8, "little")),) for i in range(5)]
        conns = []
        for key in keys:
            conn = manager.acquire(0, key)
            manager.mark_idle(conn.conn_id)
            conns.append(conn)
        released = [c for c in manager.connections.values() if c.state == ST_RELEASED]
        self.assertTrue(released)
        for conn in manager.connections.values():
            self.assertLessEqual(conn.peak_keys, conn.key_cap)

    def test_global_max_never_exceeded(self):
        clock = FakeClock()
        manager = SlotManager(topology(global_max=2), clock)
        keys = [((0x11, (100 + i).to_bytes(8, "little")),) for i in range(6)]
        for i, key in enumerate(keys):
            conn = manager.acquire(i % 3, key)
            self.assertLessEqual(manager._active_count_locked(), 2)
            manager.mark_idle(conn.conn_id)
            clock.advance(10)
        self.assertLessEqual(manager.peak_active, 2)

    def test_borrow_connection_registered(self):
        clock = FakeClock()
        borrows = (Borrow(lender=2, borrower=0, borrowed_key_cap=1, deadline_ms=100),)
        parts = [
            PartitionSpec(0, 0, 1, 10),
            PartitionSpec(1, 1, 2, 10),
            PartitionSpec(2, 1, 2, 10),
        ]
        manager = SlotManager(topology(global_max=3, partitions=parts, borrows=borrows), clock)
        key = ((0x11, (7).to_bytes(8, "little")),)
        conn = manager.acquire(0, key)
        self.assertTrue(conn.is_borrowed)
        self.assertEqual(conn.lender, 2)
        self.assertEqual(conn.home_partition, 2)
        self.assertEqual(conn.borrow_deadline_ms, 100)

    def test_borrow_deadline_expiry_releases(self):
        clock = FakeClock()
        borrows = (Borrow(lender=2, borrower=0, borrowed_key_cap=1, deadline_ms=50),)
        parts = [
            PartitionSpec(0, 0, 1, 1000),
            PartitionSpec(1, 1, 2, 1000),
            PartitionSpec(2, 1, 2, 1000),
        ]
        manager = SlotManager(topology(global_max=3, partitions=parts, borrows=borrows), clock)
        key = ((0x11, (7).to_bytes(8, "little")),)
        conn = manager.acquire(0, key)
        clock.advance(50)
        manager.recycle_locked()
        self.assertEqual(conn.state, ST_RELEASED)

    def test_borrow_cycle_injection(self):
        clock = FakeClock()
        with self.assertRaises(BorrowCycleError) as ctx:
            SlotManager(topology(), clock, inject_cycle=True)
        self.assertEqual(ctx.exception.code, "E_BORROW")

    def test_early_deadline_injection(self):
        clock = FakeClock()
        borrows = (Borrow(lender=2, borrower=0, borrowed_key_cap=1, deadline_ms=100),)
        parts = [
            PartitionSpec(0, 0, 1, 10),
            PartitionSpec(1, 1, 2, 10),
            PartitionSpec(2, 1, 2, 10),
        ]
        manager = SlotManager(topology(global_max=3, partitions=parts, borrows=borrows),
                              clock, inject_deadline=True)
        key = ((0x11, (7).to_bytes(8, "little")),)
        with self.assertRaises(DeadlineError) as ctx:
            manager.acquire(0, key)
        self.assertEqual(ctx.exception.code, "E_DEADLINE")

    def test_wait_event_recorded(self):
        clock = FakeClock()
        parts = [
            PartitionSpec(0, max_connections=1, key_cap_per_connection=2,
                          idle_recycle_ms=5),
            PartitionSpec(1, max_connections=1, key_cap_per_connection=2,
                          idle_recycle_ms=5),
        ]
        manager = SlotManager(Topology(2, tuple(parts), (), 1), clock)
        first = ((0x11, (10).to_bytes(8, "little")),)
        conn = manager.acquire(0, first)
        self.assertEqual(conn.state, ST_ACTIVE)

        original_sleep = clock.sleep_ms

        def sleep_and_release(milliseconds):
            original_sleep(milliseconds)
            manager.release(conn.conn_id)

        clock.sleep_ms = sleep_and_release
        second = ((0x11, (20).to_bytes(8, "little")),)
        manager.acquire(1, second)
        self.assertTrue(manager.wait_events)
        for event in manager.wait_events:
            self.assertLessEqual(event["wait_start_ms"], event["wait_end_ms"])

    def test_mark_unknown(self):
        clock = FakeClock()
        manager = SlotManager(topology(), FakeClock())
        key = ((0x11, (1).to_bytes(8, "little")),)
        conn = manager.acquire(0, key)
        manager.mark_idle(conn.conn_id)
        clock.advance(10)
        manager.recycle_locked()
        manager.mark_unknown()
        self.assertTrue(any(c.unknown for c in manager.connections.values()))


if __name__ == "__main__":
    unittest.main()
