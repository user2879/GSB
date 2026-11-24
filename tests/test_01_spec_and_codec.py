import json
import unittest
from pathlib import Path

from batchdiag.canonical import dump_bytes
from batchdiag.errors import ChainError, FieldError, SpecError
from batchdiag.jsonlocate import loads
from batchdiag.receipts import (decode_receipt, decode_receipts_file,
                                encode_receipt)
from batchdiag.samples import generate
from batchdiag.spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "run" / "test_samples"


class SpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate(SAMPLES)
        cls.spec = load_spec(SAMPLES / "requests.json",
                             SAMPLES / "workers.json",
                             SAMPLES / "merge_rules.json")

    def test_spec_loaded(self):
        self.assertEqual(len(self.spec.requests), 12)
        self.assertEqual(len(self.spec.workers), 3)
        self.assertEqual(self.spec.model_dim, 4)

    def test_parser_tracks_lines(self):
        text = '{"a": 1,\n "b": [2, 3]}'
        value, lines = loads(text)
        self.assertEqual(value, {"a": 1, "b": [2, 3]})
        self.assertEqual(lines.line_of("b", 1), 2)
        self.assertEqual(lines.line_of("a"), 1)

    def test_espec_negative_arrival_has_line(self):
        with self.assertRaises(SpecError) as ctx:
            load_spec(SAMPLES / "bad_requests_negative_arrival.json",
                      SAMPLES / "workers.json",
                      SAMPLES / "merge_rules.json")
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIsNotNone(ctx.exception.line)
        self.assertIn("arrival", ctx.exception.field)

    def test_espec_duplicate_port(self):
        workers = json.loads((SAMPLES / "workers.json").read_text())
        workers[1]["port"] = workers[0]["port"]
        path = SAMPLES / "dup_port.json"
        path.write_text(json.dumps(workers, indent=2))
        with self.assertRaises(SpecError) as ctx:
            load_spec(SAMPLES / "requests.json", path,
                      SAMPLES / "merge_rules.json")
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("port", ctx.exception.field)

    def test_espec_bad_host(self):
        workers = json.loads((SAMPLES / "workers.json").read_text())
        workers[0]["host"] = "10.0.0.1"
        path = SAMPLES / "bad_host.json"
        path.write_text(json.dumps(workers, indent=2))
        with self.assertRaises(SpecError) as ctx:
            load_spec(SAMPLES / "requests.json", path,
                      SAMPLES / "merge_rules.json")
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_espec_dimension_mismatch(self):
        requests = json.loads((SAMPLES / "requests.json").read_text())
        requests[0]["vector"] = [1.0, 2.0]
        path = SAMPLES / "bad_dim.json"
        path.write_text(json.dumps(requests, indent=2))
        with self.assertRaises(SpecError) as ctx:
            load_spec(path, SAMPLES / "workers.json",
                      SAMPLES / "merge_rules.json")
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("vector", ctx.exception.field)


class ReceiptCodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate(SAMPLES)
        cls.spec = load_spec(SAMPLES / "requests.json",
                             SAMPLES / "workers.json",
                             SAMPLES / "merge_rules.json")
        from batchdiag.planner import build_plan
        cls.plan = build_plan(cls.spec)
        cls.known = {r.request_id for r in cls.spec.requests}
        cls.sizes = cls.plan.batch_sizes

    def test_three_formats_normalize_to_same_core(self):
        per_format = {}
        for fmt in ("r1", "r2", "r3"):
            records = decode_receipts_file(
                SAMPLES / f"receipts_{fmt}.json",
                known_request_ids=self.known, batch_sizes=self.sizes)
            self.assertEqual(len(records), 12)
            keyed = {(r.batch_id, r.request_id): r for r in records}
            per_format[fmt] = keyed
        key = next(iter(per_format["r3"]))
        for fmt in ("r1", "r2", "r3"):
            record = per_format[fmt][key]
            self.assertEqual(record.batch_id, key[0])
            self.assertEqual(record.request_id, key[1])
        r1 = per_format["r1"][key]
        self.assertEqual({f["field"] for f in r1.filled_fields},
                         {"worker_id", "param_version", "duration_ms",
                          "rank", "duplicate_ref", "param_digest"})
        self.assertTrue(any(f["basis"].startswith("derived:batch_id_prefix")
                            for f in r1.filled_fields))
        self.assertEqual(per_format["r3"][key].filled_fields, [])

    def test_encode_roundtrip(self):
        records = decode_receipts_file(SAMPLES / "receipts_r3.json",
                                       known_request_ids=self.known,
                                       batch_sizes=self.sizes)
        raw = encode_receipt(records[0], "r1")
        self.assertEqual(set(raw), {"format", "batch_id", "request_id",
                                    "output_digest"})
        again = decode_receipt(raw, known_request_ids=self.known,
                               batch_sizes=self.sizes)
        self.assertEqual(again.source_format, "r1")
        self.assertTrue(again.filled_fields)

    def test_efield_negative_duration(self):
        with self.assertRaises(FieldError) as ctx:
            decode_receipts_file(SAMPLES / "bad_receipts_field.json",
                                 known_request_ids=self.known,
                                 batch_sizes=self.sizes)
        self.assertEqual(ctx.exception.code, "E_FIELD")
        self.assertIn("duration_ms", ctx.exception.field)

    def test_echain_missing_request(self):
        with self.assertRaises(ChainError) as ctx:
            decode_receipts_file(SAMPLES / "bad_receipts_chain.json",
                                 known_request_ids=self.known,
                                 batch_sizes=self.sizes)
        self.assertEqual(ctx.exception.code, "E_CHAIN")
        self.assertIn("duplicate_ref", ctx.exception.field)

    def test_rank_out_of_bounds_and_duplicate(self):
        good = json.loads((SAMPLES / "receipts_r3.json").read_text())
        batch_id = good[0]["batch_id"]
        size = self.sizes[batch_id]
        bad = json.loads(json.dumps(good))
        bad[0]["rank"] = size + 5
        path = SAMPLES / "bad_rank_oob.json"
        path.write_text(json.dumps(bad, indent=2))
        with self.assertRaises(FieldError) as ctx:
            decode_receipts_file(path, known_request_ids=self.known,
                                 batch_sizes=self.sizes)
        self.assertEqual(ctx.exception.code, "E_FIELD")
        self.assertIn("rank", ctx.exception.field)

        dup = json.loads(json.dumps(good))
        same_batch = [r for r in dup if r["batch_id"] == batch_id]
        same_batch[1]["rank"] = same_batch[0]["rank"]
        path2 = SAMPLES / "bad_rank_dup.json"
        path2.write_text(json.dumps(dup, indent=2))
        with self.assertRaises(FieldError) as ctx:
            decode_receipts_file(path2, known_request_ids=self.known,
                                 batch_sizes=self.sizes)
        self.assertEqual(ctx.exception.code, "E_FIELD")

    def test_canonical_order_independence(self):
        a = {"x": [3, 1, 2], "y": {"b": 2, "a": 1}}
        b = {"y": {"a": 1, "b": 2}, "x": [3, 1, 2]}
        self.assertEqual(dump_bytes(a), dump_bytes(b))


if __name__ == "__main__":
    unittest.main()
