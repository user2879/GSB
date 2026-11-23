import unittest
from pathlib import Path

from arcore.errors import SpecError
from arcore.model import Alert
from arcore.spec import load_declaration
from arcore.stream import generate_source, generate_stream, stable_sort

SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "declaration.json"


class StreamTest(unittest.TestCase):
    def setUp(self):
        self.declaration = load_declaration(SAMPLE)

    def test_byte_identical_repeated_generation(self):
        first = generate_stream(self.declaration)
        second = generate_stream(self.declaration)
        self.assertEqual(first.global_digest, second.global_digest)
        self.assertEqual(
            [a.to_dict() for a in first.alerts],
            [a.to_dict() for a in second.alerts])
        self.assertEqual(
            [s.to_dict() for s in first.source_summaries],
            [s.to_dict() for s in second.source_summaries])

    def test_source_timestamps_monotonic_and_fingerprints_unique(self):
        for spec in self.declaration.sources:
            alerts = generate_source(spec)
            timestamps = [a.timestamp_ms for a in alerts]
            self.assertEqual(timestamps, sorted(timestamps))
            fingerprints = [a.fingerprint for a in alerts]
            self.assertEqual(len(fingerprints), len(set(fingerprints)))

    def test_stable_sort_ignores_insertion_order(self):
        alerts = generate_stream(self.declaration).alerts
        shuffled = list(reversed(alerts))
        self.assertEqual(
            [(a.alert_id) for a in stable_sort(list(alerts))],
            [(a.alert_id) for a in stable_sort(shuffled)])

    def test_validate_stream_detects_regressing_timestamp(self):
        spec = self.declaration.sources[0]
        alerts = generate_source(spec)
        broken = list(alerts)
        broken[1] = Alert(**{**broken[1].__dict__,
                             "timestamp_ms": broken[0].timestamp_ms - 1})
        from arcore.stream import validate_stream
        with self.assertRaises(SpecError) as ctx:
            validate_stream(spec, broken)
        self.assertEqual(ctx.exception.source_id, spec.source_id)
        self.assertEqual(ctx.exception.field, "timestamp_ms")

    def test_validate_stream_detects_duplicate_fingerprint(self):
        spec = self.declaration.sources[0]
        alerts = generate_source(spec)
        broken = list(alerts)
        broken[1] = Alert(**{**broken[1].__dict__,
                             "fingerprint": broken[0].fingerprint})
        from arcore.stream import validate_stream
        with self.assertRaises(SpecError) as ctx:
            validate_stream(spec, broken)
        self.assertEqual(ctx.exception.field, "fingerprint")


if __name__ == "__main__":
    unittest.main()
