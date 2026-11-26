import json
import unittest
from pathlib import Path

from gsb.errors import SpecError
from gsb.gen_sample import generate
from gsb.spec import load_spec


class SpecValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("/tmp/gsb_spec_test")
        generate(self.tmp, n_events=12)
        self.spec_path = self.tmp / "spec.json"

    def test_loads_valid_sample(self):
        spec = load_spec(self.spec_path, self.tmp / "events.ndjson")
        self.assertEqual(len(spec.subscribers), 3)
        self.assertTrue(all(s.host == "127.0.0.1" for s in spec.subscribers))
        self.assertEqual(len({s.port for s in spec.subscribers}), 3)

    def _reload(self, mutate):
        doc = json.loads(self.spec_path.read_text())
        mutate(doc)
        path = self.tmp / "bad.json"
        path.write_text(json.dumps(doc))
        return path

    def test_duplicate_subscriber_id(self):
        path = self._reload(
            lambda d: d["subscribers"][1].__setitem__(
                "subscriber_id", d["subscribers"][0]["subscriber_id"]
            )
        )
        with self.assertRaises(SpecError) as ctx:
            load_spec(path, self.tmp / "events.ndjson")
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "subscriber_id")

    def test_host_literal_and_unique_port(self):
        path = self._reload(
            lambda d: d["subscribers"][0].__setitem__("host", "0.0.0.0")
        )
        with self.assertRaises(SpecError) as ctx:
            load_spec(path, self.tmp / "events.ndjson")
        self.assertEqual(ctx.exception.field, "host")

        path = self._reload(
            lambda d: d["subscribers"][1].__setitem__(
                "port", d["subscribers"][0]["port"]
            )
        )
        with self.assertRaises(SpecError) as ctx:
            load_spec(path, self.tmp / "events.ndjson")
        self.assertEqual(ctx.exception.field, "port")

    def test_positive_integer_constraints(self):
        path = self._reload(lambda d: d["channels"][0].__setitem__("batch_size", 0))
        with self.assertRaises(SpecError) as ctx:
            load_spec(path, self.tmp / "events.ndjson")
        self.assertEqual(ctx.exception.field, "batch_size")

    def test_event_ordering(self):
        events = self.tmp / "bad_events.ndjson"
        lines = [
            json.loads(line)
            for line in (self.tmp / "events.ndjson").read_text().splitlines()
        ]
        lines[2]["seq"] = lines[1]["seq"] - 5
        events.write_text("\n".join(json.dumps(e) for e in lines))
        with self.assertRaises(SpecError) as ctx:
            load_spec(self.spec_path, events)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.field, "seq")

        lines = [
            json.loads(line)
            for line in (self.tmp / "events.ndjson").read_text().splitlines()
        ]
        lines[2]["timestamp_ms"] = lines[1]["timestamp_ms"] - 1
        events.write_text("\n".join(json.dumps(e) for e in lines))
        with self.assertRaises(SpecError) as ctx:
            load_spec(self.spec_path, events)
        self.assertEqual(ctx.exception.field, "timestamp_ms")


if __name__ == "__main__":
    unittest.main()
