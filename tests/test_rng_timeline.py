import unittest

from chaoskit.rng import SplitMix64, mix_u64
from chaoskit.spec import load_spec
from chaoskit.timeline import build_timeline, timeline_digest, verify_timeline
from chaoskit.errors import TimelineError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class RngTest(unittest.TestCase):
    def test_deterministic_sequence(self):
        first = [SplitMix64(42).next_u64() for _ in range(8)]
        second = [SplitMix64(42).next_u64() for _ in range(8)]
        self.assertEqual(first, second)
        self.assertNotEqual(first, [SplitMix64(43).next_u64() for _ in range(8)])

    def test_bounds(self):
        rng = SplitMix64(7)
        for _ in range(1000):
            value = rng.next_below(13)
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 13)
        self.assertEqual(SplitMix64(1).next_below(1), 0)

    def test_mix_stable(self):
        self.assertEqual(mix_u64(["a", 1, b"b"]), mix_u64(["a", 1, b"b"]))
        self.assertNotEqual(mix_u64(["a", 1]), mix_u64(["a", 2]))


class TimelineTest(unittest.TestCase):
    def test_order_independent_digest(self):
        spec_a = load_spec(str(ROOT / "examples" / "sample_experiment.json"))
        spec_b = load_spec(str(ROOT / "examples" / "sample_experiment_shuffled.json"))
        entries_a = build_timeline(spec_a)
        entries_b = build_timeline(spec_b)
        self.assertEqual(timeline_digest(entries_a), timeline_digest(entries_b))
        self.assertEqual([e.to_dict() for e in entries_a], [e.to_dict() for e in entries_b])

    def test_stable_across_repeated_generation(self):
        spec = load_spec(str(ROOT / "examples" / "sample_experiment.json"))
        d1 = timeline_digest(build_timeline(spec))
        d2 = timeline_digest(build_timeline(spec))
        self.assertEqual(d1, d2)

    def test_seed_changes_digest(self):
        spec = load_spec(str(ROOT / "examples" / "sample_experiment.json"))
        object.__setattr__(spec, "seed", spec.seed + 1)
        self.assertNotEqual(
            timeline_digest(build_timeline(spec)),
            timeline_digest(build_timeline(load_spec(str(ROOT / "examples" / "sample_experiment.json")))),
        )

    def test_verify_reports_first_deviation(self):
        spec = load_spec(str(ROOT / "examples" / "sample_experiment.json"))
        expected = build_timeline(spec)
        actual = build_timeline(spec)
        from dataclasses import replace
        actual[2] = replace(actual[2], target_id="orders")
        with self.assertRaises(TimelineError) as ctx:
            verify_timeline(expected, actual)
        self.assertEqual(ctx.exception.first_diverging_index, 2)
        self.assertEqual(ctx.exception.code, "E_TIMELINE")


if __name__ == "__main__":
    unittest.main()
