import unittest

from swa.generator import (
    build_round_robin_plan,
    event_stream_digest,
    generate_all_events,
    generate_source_events,
)
from swa.spec import parse_spec


def make_spec(counts=(8, 6)):
    return parse_spec(
        {
            "value_min": 0,
            "value_max": 50,
            "group_keys": ["a", "b", "c"],
            "sources": [
                {"source_id": "alpha", "seed": 123, "event_count": counts[0]},
                {"source_id": "beta", "seed": 456, "event_count": counts[1]},
            ],
            "rules": [
                {
                    "rule_id": "r",
                    "window_seconds": 2,
                    "group_key": "group_key",
                    "aggregations": ["count", "sum", "max"],
                    "concurrency": 1,
                    "memory_budget": 4,
                }
            ],
        }
    )


class GeneratorTests(unittest.TestCase):
    def test_seeded_generation_is_byte_stable_and_monotonic(self):
        spec = make_spec()
        first = generate_all_events(spec)
        second = generate_all_events(spec)
        self.assertEqual(event_stream_digest(first), event_stream_digest(second))
        for events in first.values():
            timestamps = [event.timestamp_ms for event in events]
            self.assertEqual(timestamps, sorted(timestamps))
            self.assertTrue(all(later > earlier for earlier, later in zip(timestamps, timestamps[1:])))

    def test_round_robin_plan_interleaves_sources(self):
        spec = make_spec()
        streams = generate_all_events(spec)
        plan = build_round_robin_plan(streams)
        self.assertEqual(plan[0].source_id, "alpha")
        self.assertEqual(plan[1].source_id, "beta")
        # Per-source order is preserved.
        for source_id, events in streams.items():
            planned = [event for event in plan if event.source_id == source_id]
            self.assertEqual([event.source_seq for event in planned], [event.source_seq for event in events])

    def test_sample_override_changes_planned_event(self):
        spec = make_spec()
        baseline = generate_source_events(spec.sources[0], spec)
        overrides = {("alpha", 2): {"value": 7, "group_key": "c"}}
        changed = generate_source_events(spec.sources[0], spec, overrides)
        self.assertNotEqual(baseline[1].value, changed[1].value)
        self.assertEqual(changed[1].value, 7)
        self.assertEqual(changed[1].group_key, "c")
        # Other events stay byte-identical.
        self.assertEqual(baseline[0].to_dict(), changed[0].to_dict())
        self.assertEqual(baseline[2:], changed[2:])


if __name__ == "__main__":
    unittest.main()
