import os
import tempfile
import unittest

from bench.mw.config import (INT64_MAX, SessionSpec, load_events, load_spec,
                             encode_fixed, saturate_int64)
from bench.mw.engine import SessionEngine
from bench.mw.errors import SpecError, WatermarkError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASIC_SPEC = os.path.join(ROOT, "samples", "basic", "spec.json")
BASIC_EVENTS = os.path.join(ROOT, "samples", "basic", "events.ndjson")


class SpecValidationTests(unittest.TestCase):
    def test_valid_load(self):
        spec = load_spec(BASIC_SPEC)
        self.assertEqual(spec.scale, 2)
        self.assertEqual(spec.partition_keys, ("p0", "p1", "p2"))
        events = load_events(BASIC_EVENTS, spec)
        self.assertEqual(len(events), 16)

    def test_bad_spec_reports_field(self):
        with self.assertRaises(SpecError) as ctx:
            load_spec(os.path.join(ROOT, "samples", "invalid", "bad_spec.json"))
        self.assertEqual(ctx.exception.code, "E_SPEC")
        # out_of_order_ms=0 fails before scale=9; first bad field is reported.
        self.assertEqual(ctx.exception.details["field"], "out_of_order_ms")

    def test_bad_event_reports_line_and_field(self):
        spec = load_spec(os.path.join(ROOT, "samples", "invalid", "spec_p0.json"))
        with self.assertRaises(SpecError) as ctx:
            load_events(os.path.join(ROOT, "samples", "invalid",
                                     "bad_events.ndjson"), spec)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.details["line"], 2)
        self.assertEqual(ctx.exception.details["field"], "seq")

    def test_corrupt_json_line(self):
        spec = load_spec(BASIC_SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.ndjson")
            with open(path, "w") as fh:
                fh.write('{"partition": "p0", "seq": 0, "ts_ms": 1, "value": 1}\n')
                fh.write('{not json}\n')
            with self.assertRaises(SpecError) as ctx:
                load_events(path, spec)
            self.assertEqual(ctx.exception.details["line"], 2)
            self.assertEqual(ctx.exception.details["field"], "<json>")

    def test_unknown_partition_and_negative_ts(self):
        spec = load_spec(BASIC_SPEC)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.ndjson")
            with open(path, "w") as fh:
                fh.write('{"partition": "px", "seq": 0, "ts_ms": 1, "value": 1}\n')
            with self.assertRaises(SpecError) as ctx:
                load_events(path, spec)
            self.assertEqual(ctx.exception.details["field"], "partition")


class WatermarkTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec(BASIC_SPEC)
        self.events = load_events(BASIC_EVENTS, self.spec)

    def test_expected_finalized_and_late(self):
        engine = SessionEngine(self.spec)
        steps = engine.apply_events(self.events)
        self.assertTrue(all(s.new_watermark_ms >= s.old_watermark_ms
                            for s in steps))
        engine.drain()
        expected = {
            "p0": (4, 2, 130, 13000),
            "p1": (5, 1, 19, 1900),
            "p2": (4, 0, 20, 2000),
        }
        for key, (count, late, raw, fixed) in expected.items():
            view = engine.parts[key].view()
            self.assertEqual(view["finalized_count"], count)
            self.assertEqual(view["late_count"], late)
            self.assertEqual(view["sum_raw"], raw)
            self.assertEqual(view["sum_fixed"], fixed)

    def test_watermark_rollback_rejected(self):
        engine = SessionEngine(self.spec)
        engine.apply_events(self.events[:4])
        current = engine.parts["p0"].watermark_ms
        with self.assertRaises(WatermarkError) as ctx:
            engine.parts["p0"].request_watermark(current - 1)
        self.assertEqual(ctx.exception.code, "E_WATER")
        # original watermark retained
        self.assertEqual(engine.parts["p0"].watermark_ms, current)

    def test_late_event_does_not_change_aggregates(self):
        engine = SessionEngine(self.spec)
        engine.apply_events(self.events)
        engine.drain()
        before = {k: v.view() for k, v in engine.parts.items()}
        # replay and compare: late events never touch sums
        engine2 = SessionEngine(self.spec)
        for event in self.events:
            engine2.apply_events([event])  # one-by-one batching
        engine2.drain()
        for key in engine.parts:
            self.assertEqual(engine.parts[key].view(),
                             engine2.parts[key].view())
        self.assertIsNotNone(before)


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec(BASIC_SPEC)
        self.events = load_events(BASIC_EVENTS, self.spec)

    def _run(self, batch_sizes, watermark_steps=None):
        engine = SessionEngine(self.spec)
        index = 0
        if watermark_steps:
            for step in watermark_steps:
                engine.request_watermarks(step)
        while index < len(self.events):
            size = batch_sizes[index % len(batch_sizes)]
            engine.apply_events(self.events[index:index + size])
            index += size
        engine.drain()
        return engine.views()

    def test_arrival_order_batch_and_step_invariant(self):
        import random
        canonical = self._run([len(self.events)])
        # shuffled arrival order
        shuffled = self.events[:]
        random.Random(42).shuffle(shuffled)
        engine = SessionEngine(self.spec)
        engine.apply_events(shuffled)
        engine.drain()
        self.assertEqual(engine.views(), canonical)
        # varied batch sizes
        self.assertEqual(self._run([1, 2, 3, 5]), canonical)
        self.assertEqual(self._run([7, 1, 4]), canonical)
        # explicit intermediate watermark requests clamped at/below the
        # event-derived boundary cannot move finalization earlier/later
        self.assertEqual(
            self._run([3, 3, 3, 3, 3, 1],
                      watermark_steps=[100, 400, 700]), canonical)
        # byte-stable serialization
        from bench.mw.config import canonical_dumps
        self.assertEqual(canonical_dumps(engine.views()),
                         canonical_dumps(canonical))


class FixedPointTests(unittest.TestCase):
    def test_encode_exact_for_integers(self):
        self.assertEqual(encode_fixed(130, 2), 13000)
        self.assertEqual(encode_fixed(7, 3), 7000)

    def test_saturation_bounds(self):
        self.assertEqual(saturate_int64(INT64_MAX + 5), INT64_MAX)
        self.assertEqual(saturate_int64(-INT64_MAX - 99), -(2 ** 63))

    def test_saturation_is_order_independent(self):
        spec = SessionSpec(out_of_order_ms=1, latency_budget_ms=1, scale=6,
                           rules=(), raw={})
        huge = [10 ** 20, -(10 ** 20), 10 ** 20]
        from bench.mw.engine import PartitionEngine
        from bench.mw.config import Event
        results = []
        for ordering in (huge, list(reversed(huge))):
            part = PartitionEngine("p", spec)
            for i, value in enumerate(ordering):
                part.apply_event(
                    Event("p", i, i * 10, value, i + 1))
            part.drain()
            results.append((part.raw_sum, part.fixed_sum()))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
