from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from h2burst.errors import WorkbenchError
from h2burst.model import MAX_WINDOW, load_session, parse_session
from tests.helpers import goaway, raw_session, rst, wu


class SpecValidationTests(unittest.TestCase):
    def assert_spec_error(self, raw: dict, field: str, event_seq: int | None = None) -> None:
        with self.assertRaises(WorkbenchError) as ctx:
            parse_session(raw)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, field)
        if event_seq is not None:
            self.assertEqual(ctx.exception.event_seq, event_seq)

    def test_valid_session_and_digest_is_stable(self) -> None:
        raw = raw_session(
            streams=[{"stream_id": 1, "priority": 2, "request_body_bytes": 10}],
            events=[wu(0, 10)],
        )
        first = parse_session(raw)
        second = parse_session(json.loads(json.dumps(raw)))
        self.assertEqual(first.content_digest, second.content_digest)
        self.assertEqual(first.next_event_seq, 1)

    def test_duplicate_stream_id_rejected(self) -> None:
        raw = raw_session(
            streams=[
                {"stream_id": 1, "priority": 1, "request_body_bytes": 1},
                {"stream_id": 1, "priority": 2, "request_body_bytes": 2},
            ]
        )
        self.assert_spec_error(raw, "streams[1].stream_id")

    def test_even_stream_id_rejected(self) -> None:
        raw = raw_session(streams=[{"stream_id": 2, "priority": 1, "request_body_bytes": 1}])
        self.assert_spec_error(raw, "streams[0].stream_id")

    def test_priority_range_rejected(self) -> None:
        for priority in (0, 257):
            raw = raw_session(streams=[{"stream_id": 1, "priority": priority, "request_body_bytes": 1}])
            self.assert_spec_error(raw, "streams[0].priority")

    def test_negative_body_rejected(self) -> None:
        raw = raw_session(streams=[{"stream_id": 1, "priority": 1, "request_body_bytes": -1}])
        self.assert_spec_error(raw, "streams[0].request_body_bytes")

    def test_initial_window_range_rejected(self) -> None:
        raw = raw_session(initial_window=MAX_WINDOW + 1)
        self.assert_spec_error(raw, "connection.initial_window")

    def test_unsupported_frame_size_rejected(self) -> None:
        raw = raw_session(max_frame_size=20000)
        self.assert_spec_error(raw, "connection.max_frame_size")

    def test_event_sequence_must_be_continuous(self) -> None:
        raw = raw_session(events=[wu(0, 1), wu(2, 1)])
        self.assert_spec_error(raw, "events[1].seq", 2)

    def test_arrival_time_must_be_monotonic(self) -> None:
        raw = raw_session(events=[wu(0, 1, arrival_ms=5), wu(1, 1, arrival_ms=4)])
        self.assert_spec_error(raw, "events[1].arrival_ms", 1)

    def test_negative_window_increment_rejected(self) -> None:
        raw = raw_session(events=[wu(0, -1)])
        self.assert_spec_error(raw, "events[0].increment", 0)

    def test_unknown_stream_reference_rejected(self) -> None:
        raw = raw_session(events=[rst(0, 99)])
        self.assert_spec_error(raw, "events[0].stream_id", 0)

    def test_unknown_event_type_rejected(self) -> None:
        raw = raw_session(events=[{"seq": 0, "arrival_ms": 0, "type": "mystery"}])
        self.assert_spec_error(raw, "events[0].type", 0)

    def test_unknown_top_level_field_rejected(self) -> None:
        raw = raw_session()
        raw["surprise"] = True
        self.assert_spec_error(raw, "$.surprise")

    def test_non_array_streams_rejected_as_e_spec(self) -> None:
        raw = raw_session()
        raw["streams"] = 3
        self.assert_spec_error(raw, "streams")

    def test_boolean_integer_rejected(self) -> None:
        raw = raw_session()
        raw["connection"]["initial_window"] = True
        self.assert_spec_error(raw, "connection.initial_window")

    def test_goaway_aliases_and_one_based_events(self) -> None:
        raw = raw_session(
            events=[
                {"seq": 1, "arrival_ms": 0, "type": "goaway", "last_stream_id": 1, "error_code": 0}
            ]
        )
        spec = parse_session(raw)
        self.assertEqual(spec.events[0].kind, "goaway")
        self.assertEqual(spec.event_offset, 1)
        self.assertEqual(spec.next_event_seq, 2)

    def test_load_session_wraps_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(WorkbenchError) as ctx:
                load_session(path)
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertEqual(ctx.exception.field, "$")


if __name__ == "__main__":
    unittest.main()
