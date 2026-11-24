from __future__ import annotations

import unittest

from h2burst.engine import Engine
from h2burst.errors import WorkbenchError
from h2burst.model import MAX_WINDOW
from h2burst.state_machine import CLOSED, HALF_CLOSED_LOCAL, OPEN
from tests.helpers import goaway, raw_session, rst, spec_session, wu


class EngineLedgerAndStateTests(unittest.TestCase):
    def test_bootstrap_sends_headers_and_opens_streams(self) -> None:
        spec = spec_session(
            streams=[
                {"stream_id": 1, "priority": 1, "request_body_bytes": 10},
                {"stream_id": 3, "priority": 2, "request_body_bytes": 0},
            ]
        )
        engine = Engine(spec)
        self.assertEqual([frame.frame_type for frame in engine.frames], ["HEADERS", "HEADERS"])
        self.assertEqual(engine.streams[1].state, OPEN)
        self.assertEqual(engine.streams[3].state, HALF_CLOSED_LOCAL)
        self.assertEqual(engine.transitions[-1].reason, "request_body_complete")

    def test_data_frame_deducts_both_windows_with_before_after(self) -> None:
        spec = spec_session(
            streams=[{"stream_id": 1, "priority": 1, "request_body_bytes": 20000}],
            initial_window=30000,
            initial_stream_window=25000,
            scheduling_quantum=1,
            events=[wu(0, 0)],
        )
        engine = Engine(spec)
        before_conn = engine.connection_window
        before_stream = engine.streams[1].window
        engine.run(drain=False)
        data = [frame for frame in engine.frames if frame.frame_type == "DATA"]
        self.assertEqual(len(data), 1)
        frame = data[0]
        self.assertEqual(frame.data_bytes, 16384)
        self.assertEqual(frame.connection_window_before, before_conn)
        self.assertEqual(frame.connection_window_after, before_conn - 16384)
        self.assertEqual(frame.stream_window_before, before_stream)
        self.assertEqual(frame.stream_window_after, before_stream - 16384)
        self.assertEqual(engine.connection_window, before_conn - 16384)

    def test_connection_window_update_increases_only_connection(self) -> None:
        spec = spec_session(events=[wu(0, 1000)], scheduling_quantum=1)
        engine = Engine(spec)
        stream_before = engine.streams[1].window
        conn_before = engine.connection_window
        engine.apply_event(spec.events[0])
        self.assertEqual(engine.connection_window, conn_before + 1000)
        self.assertEqual(engine.streams[1].window, stream_before)

    def test_stream_window_update_increases_only_stream(self) -> None:
        spec = spec_session(events=[wu(0, 500, stream_id=1)], scheduling_quantum=1)
        engine = Engine(spec)
        conn_before = engine.connection_window
        stream_before = engine.streams[1].window
        engine.apply_event(spec.events[0])
        self.assertEqual(engine.connection_window, conn_before)
        self.assertEqual(engine.streams[1].window, stream_before + 500)

    def test_connection_window_overflow_is_e_flow_and_atomic(self) -> None:
        spec = spec_session(initial_window=MAX_WINDOW - 5, events=[wu(0, 10)])
        engine = Engine(spec)
        before_digest = engine.run_digest()
        with self.assertRaises(WorkbenchError) as ctx:
            engine.step()
        self.assertEqual(ctx.exception.code, "E_FLOW")
        self.assertEqual(ctx.exception.field, "increment")
        self.assertEqual(ctx.exception.event_seq, 0)
        self.assertEqual(engine.run_digest(), before_digest)

    def test_stream_window_overflow_is_e_flow(self) -> None:
        spec = spec_session(initial_stream_window=MAX_WINDOW - 1, events=[wu(0, 10, stream_id=1)])
        engine = Engine(spec)
        with self.assertRaises(WorkbenchError) as ctx:
            engine.step()
        self.assertEqual(ctx.exception.code, "E_FLOW")
        self.assertEqual(ctx.exception.event_seq, 0)

    def test_rst_closes_stream_and_second_rst_is_e_state_without_ledger_change(self) -> None:
        spec = spec_session(events=[rst(0, 1), rst(1, 1)])
        engine = Engine(spec)
        engine.step()
        self.assertEqual(engine.streams[1].state, CLOSED)
        digest_before = engine.run_digest()
        with self.assertRaises(WorkbenchError) as ctx:
            engine.step()
        self.assertEqual(ctx.exception.code, "E_STATE")
        self.assertEqual(ctx.exception.field, "stream_state")
        self.assertEqual(ctx.exception.event_seq, 1)
        self.assertEqual(engine.run_digest(), digest_before)

    def test_window_update_on_closed_stream_is_e_state(self) -> None:
        spec = spec_session(events=[rst(0, 1), wu(1, 100, stream_id=1)])
        engine = Engine(spec)
        engine.step()
        before = engine.run_digest()
        with self.assertRaises(WorkbenchError) as ctx:
            engine.step()
        self.assertEqual(ctx.exception.code, "E_STATE")
        self.assertEqual(engine.run_digest(), before)

    def test_body_completion_moves_to_half_closed_local(self) -> None:
        spec = spec_session(
            streams=[{"stream_id": 1, "priority": 1, "request_body_bytes": 100}],
            scheduling_quantum=1,
        )
        engine = Engine(spec).run()
        self.assertEqual(engine.streams[1].state, HALF_CLOSED_LOCAL)
        self.assertEqual(engine.streams[1].body_remaining, 0)
        completed = [item for item in engine.transitions if item.reason == "request_body_complete"]
        self.assertTrue(completed)
        self.assertTrue(engine.frames[-1].end_stream)

    def test_goaway_closes_only_streams_above_last_stream_id(self) -> None:
        spec = spec_session(
            streams=[
                {"stream_id": 1, "priority": 1, "request_body_bytes": 100},
                {"stream_id": 3, "priority": 2, "request_body_bytes": 100},
                {"stream_id": 5, "priority": 3, "request_body_bytes": 100},
            ],
            events=[goaway(0, 3)],
        )
        engine = Engine(spec)
        engine.step()
        self.assertNotEqual(engine.streams[1].state, CLOSED)
        self.assertNotEqual(engine.streams[3].state, CLOSED)
        self.assertEqual(engine.streams[5].state, CLOSED)
        stream_5_frames = len([frame for frame in engine.frames if frame.stream_id == 5])
        engine.run()
        self.assertEqual(
            stream_5_frames,
            len([frame for frame in engine.frames if frame.stream_id == 5]),
        )
        self.assertTrue(all(item.event_seq == 0 for item in engine.transitions if item.stream_id == 5))

    def test_max_concurrent_streams_limits_headers(self) -> None:
        spec = spec_session(
            streams=[
                {"stream_id": 1, "priority": 3, "request_body_bytes": 100},
                {"stream_id": 3, "priority": 1, "request_body_bytes": 100},
                {"stream_id": 5, "priority": 2, "request_body_bytes": 100},
            ],
            max_concurrent_streams=1,
        )
        engine = Engine(spec)
        self.assertEqual(sum(1 for frame in engine.frames if frame.frame_type == "HEADERS"), 1)
        self.assertEqual(engine.streams[3].state, OPEN)
        self.assertEqual(engine.streams[1].state, "idle")

    def test_priority_change_is_reflected_in_runtime(self) -> None:
        spec = spec_session(
            streams=[{"stream_id": 1, "priority": 8, "request_body_bytes": 100}],
            events=[
                {"seq": 0, "arrival_ms": 0, "type": "priority_change", "stream_id": 1, "priority": 2}
            ],
        )
        engine = Engine(spec).run()
        self.assertEqual(engine.streams[1].priority, 2)


if __name__ == "__main__":
    unittest.main()
