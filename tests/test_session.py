from __future__ import annotations

import unittest
from pathlib import Path

from h2mux.declaration import load_declaration, load_script, parse_declaration, parse_script
from h2mux.priority import PriorityNode, PriorityTree
from h2mux.session import H2Session
from h2mux.util import digest_value


ROOT = Path(__file__).resolve().parents[1]


class SessionTests(unittest.TestCase):
    def sample_session(self) -> H2Session:
        declaration = load_declaration(ROOT / "samples" / "session.json")
        script = load_script(ROOT / "samples" / "frames.json", declaration)
        return H2Session(declaration, script).run()

    def test_sample_completes_and_closes_streams(self) -> None:
        session = self.sample_session()
        completion = session.completion_report()
        self.assertTrue(completion["complete"])
        self.assertTrue(completion["goaway_received"])
        self.assertEqual({stream.state for stream in session.streams.values()}, {"closed"})

    def test_two_layer_windows_are_decremented_together_and_never_negative(self) -> None:
        session = self.sample_session()
        data_records = [
            record
            for record in session.ledger.records
            if record["direction"] == "send" and record["frame_type"] == "DATA"
        ]
        self.assertTrue(data_records)
        declaration = session.declaration
        for record in data_records:
            amount = record["payload_bytes"]
            self.assertLessEqual(amount, declaration.max_frame_size)
            self.assertEqual(
                record["conn_send_window_before"] - amount,
                record["conn_send_window_after"],
            )
            self.assertEqual(
                record["stream_send_window_before"] - amount,
                record["stream_send_window_after"],
            )
            self.assertGreaterEqual(record["conn_send_window_after"], 0)
            self.assertGreaterEqual(record["stream_send_window_after"], 0)

    def test_window_wait_has_monotonic_interval(self) -> None:
        session = self.sample_session()
        self.assertTrue(session.ledger.empty_waits)
        wait = session.ledger.empty_waits[0]
        self.assertEqual(wait["reason"], "connection_send_window_exhausted")
        self.assertGreaterEqual(wait["end_ms"], wait["start_ms"])

    def test_concurrency_queue_and_released_slot_are_recorded(self) -> None:
        session = self.sample_session()
        self.assertTrue(session.ledger.queue_waits)
        first = session.ledger.queue_waits[0]
        self.assertEqual(first["reason"], "MAX_CONCURRENT_STREAMS")
        self.assertIsNotNone(first["end_ms"])
        self.assertIn("max_concurrent", str(first["closed_by"]))

    def test_state_trace_covers_lifecycle(self) -> None:
        session = self.sample_session()
        transitions = session.ledger.transitions
        self.assertTrue(any(item["from"] == "idle" and item["to"] == "open" for item in transitions))
        self.assertTrue(any(item["to"] == "half_closed_local" for item in transitions))
        self.assertTrue(any(item["to"] == "half_closed_remote" for item in transitions))
        self.assertTrue(any(item["to"] == "closed" for item in transitions))

    def test_priority_change_and_parent_rst_rehang_are_recorded(self) -> None:
        session = self.sample_session()
        events = session.ledger.priority_events
        self.assertTrue(any(item["stream_id"] == 5 and item["new_weight"] == 40 for item in events))
        rehang = [item for item in events if item["stream_id"] == 7]
        self.assertEqual(len(rehang), 1)
        self.assertEqual(rehang[0]["new_parent_id"], 0)
        self.assertEqual(rehang[0]["reason"], "parent_rst_stream")

    def test_closed_stream_error_does_not_mutate_windows_or_state(self) -> None:
        session = self.sample_session()
        error = [item for item in session.ledger.records if item["code"] == "E_STREAM"][0]
        self.assertEqual(error["target_stream_id"], 1)
        self.assertEqual(error["conn_send_window_before"], error["conn_send_window_after"])
        self.assertEqual(error["stream_send_window_before"], error["stream_send_window_after"])
        self.assertEqual(error["state_before"], error["state_after"])
        self.assertEqual(error["state_after"], "closed")

    def test_repeated_runs_have_identical_ledger_digest(self) -> None:
        first = self.sample_session()
        second = self.sample_session()
        self.assertEqual(
            digest_value(first.ledger.to_dict()),
            digest_value(second.ledger.to_dict()),
        )
        self.assertEqual(first.summary()["frame_count"], second.summary()["frame_count"])

    def test_window_priority_can_starve_low_window_stream(self) -> None:
        declaration = parse_declaration(
            {
                "schema_version": 1,
                "declaration_version": 1,
                "client": {"connection_window": 65535, "stream_window": 65535},
                "server": {"connection_window": 500000, "stream_window": 1000},
                "max_concurrent_streams": 2,
                "max_frame_size": 16384,
                "requests": [
                    {"stream_id": 1, "weight": 16, "parent_id": 0, "body_bytes": 100000},
                    {"stream_id": 3, "weight": 16, "parent_id": 0, "body_bytes": 5000},
                ],
            }
        )
        script = parse_script(
            {
                "script_version": 1,
                "declaration_version": 1,
                "events": [
                    {"event": "schedule", "max_frames": 2},
                    {
                        "event": "recv",
                        "frame": {
                            "type": "WINDOW_UPDATE",
                            "stream_id": 1,
                            "increment": 200000,
                        },
                    },
                    {"event": "schedule", "max_frames": 1},
                    {"event": "schedule", "max_frames": 1},
                    {"event": "schedule", "max_frames": 1},
                ],
            },
            declaration=declaration,
        )
        session = H2Session(declaration, script, strategy="window_priority").run()
        starvation = session.diagnostics()["starvation"]
        self.assertTrue(any(item["stream_id"] == 3 and item["round_count"] >= 3 for item in starvation))

    def test_exclusive_priority_reparents_previous_siblings(self) -> None:
        tree = PriorityTree(
            [
                PriorityNode(1, 0, 16),
                PriorityNode(3, 0, 16),
                PriorityNode(5, 0, 16),
            ]
        )
        change = tree.update(5, parent_id=0, weight=32, exclusive=True)
        self.assertEqual(change["reparented_child_ids"], [1, 3])
        self.assertEqual(tree.parent_of(1), 5)
        self.assertEqual(tree.parent_of(3), 5)
        self.assertEqual(tree.parent_of(5), 0)

    def test_invalid_settings_frame_is_atomic(self) -> None:
        declaration = load_declaration(ROOT / "samples" / "session.json")
        script = parse_script(
            {
                "script_version": 1,
                "declaration_version": 7,
                "events": [
                    {
                        "event": "recv",
                        "frame": {
                            "type": "SETTINGS",
                            "stream_id": 0,
                            "settings": [
                                {"id": 4, "value": 9000},
                                {"id": 5, "value": 100},
                            ],
                        },
                    }
                ],
            },
            declaration=declaration,
        )
        session = H2Session(declaration, script).run()
        self.assertEqual(session.peer_initial_stream_window, 18000)
        self.assertEqual(session.peer_max_frame_size, 16384)
        self.assertTrue(all(stream.send_window == 18000 for stream in session.streams.values()))
        self.assertEqual(session.ledger.records[-1]["code"], "E_FRAME")


if __name__ == "__main__":
    unittest.main()
