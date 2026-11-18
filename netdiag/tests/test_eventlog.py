import json
import unittest

from netdiag.eventlog import parse_event_log
from netdiag.errors import EventFormatError


class EventLogTests(unittest.TestCase):
    def test_v1_normalized_keeps_version(self):
        raw = {"format": "loss-event-list", "version": "v1",
               "flow": "tcp:h1-h2",
               "events": [{"event_id": 1, "loss_count": 2}]}
        log = parse_event_log(raw)
        self.assertEqual(log.version, "v1")
        self.assertEqual(log.events[0].source_version, "v1")
        self.assertIsNone(log.events[0].link_id)

    def test_v2_requires_link_and_window(self):
        raw = {"format": "loss-event-list", "version": "v2",
               "events": [{"event_id": 1, "flow": "f", "loss_count": 1}]}
        with self.assertRaises(EventFormatError) as ctx:
            parse_event_log(raw)
        self.assertEqual(ctx.exception.code, "E_FORMAT")
        self.assertTrue(
            ctx.exception.context["field_path"].startswith("events[0]"))

    def test_v3_requires_retransmitted_and_depth(self):
        raw = {"format": "loss-event-list", "version": "v3", "events": []}
        log = parse_event_log(raw)
        self.assertEqual(log.events, [])
        bad = {"format": "loss-event-list", "version": "v3",
               "events": [{"event_id": 1, "flow": "f", "loss_count": 1,
                           "link_id": "L1", "window_ms": [0, 10],
                           "retransmitted": True}]}
        with self.assertRaises(EventFormatError) as ctx:
            parse_event_log(bad)
        self.assertEqual(ctx.exception.context["field_path"],
                         "events[0].ooo_depth")

    def test_unknown_version_rejected(self):
        raw = {"format": "loss-event-list", "version": "v9",
               "events": []}
        with self.assertRaises(EventFormatError) as ctx:
            parse_event_log(raw)
        self.assertEqual(ctx.exception.context["field_path"], "version")

    def test_negative_count_rejected_with_path(self):
        raw = {"format": "loss-event-list", "version": "v1",
               "flow": "f",
               "events": [{"event_id": 3, "loss_count": -1}]}
        with self.assertRaises(EventFormatError) as ctx:
            parse_event_log(raw)
        self.assertEqual(ctx.exception.context["field_path"],
                         "events[0].loss_count")


if __name__ == "__main__":
    unittest.main()
