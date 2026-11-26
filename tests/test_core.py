import unittest

import numpy as np

from src import core


def _cfg():
    return {
        "sample_seed": 7,
        "request_count": 6,
        "input_dim": 4,
        "hidden_dim": 6,
        "output_dim": 4,
        "batch_size": 2,
        "fixed_point": {"frac_bits": 8, "word_bits": 32,
                        "rounding": "floor", "saturation": "saturate"},
        "workers": [{"worker_id": "w0", "seed": 101},
                    {"worker_id": "w1", "seed": 202}],
        "workers_by_id": {"w0": {"worker_id": "w0", "seed": 101},
                          "w1": {"worker_id": "w1", "seed": 202}},
        "scheduler": {"global_limit": 2, "per_worker_limit": 1, "timeout_ms": 1000},
    }


class TestFixedPoint(unittest.TestCase):
    def test_floor_rounding_toward_negative_infinity(self):
        values = np.array([255, 256, -1, -255, -256, -257], dtype=np.int64)
        shifted = core.floor_shift(values, 8)
        self.assertEqual(shifted.tolist(), [0, 1, -1, -1, -1, -2])

    def test_saturation_clips_to_int32(self):
        values = np.array([2 ** 40, -(2 ** 40), 0, 2 ** 31 - 1, -(2 ** 31)],
                          dtype=np.int64)
        sat = core.saturate_int32(values)
        self.assertEqual(sat.dtype, np.int32)
        self.assertEqual(sat.tolist(),
                         [core.INT32_MAX, core.INT32_MIN, 0,
                          core.INT32_MAX, core.INT32_MIN])

    def test_spec_declares_rules(self):
        spec = core.FixedPointSpec.from_config(_cfg()["fixed_point"])
        self.assertEqual(spec.rounding, "floor")
        self.assertEqual(spec.saturation, "saturate")
        self.assertIn("rounding", spec.to_dict())


class TestDeterminism(unittest.TestCase):
    def test_param_digest_independent_of_dict_order(self):
        cfg = _cfg()
        spec1 = core.param_spec_dict("w0", 101, cfg)
        spec2 = dict(reversed(list(spec1.items())))
        self.assertEqual(core.spec_digest(spec1), core.spec_digest(spec2))
        params = core.generate_params(spec1)
        self.assertEqual(core.param_digest(spec1, params),
                         core.param_digest(spec2, params))

    def test_batching_independent_of_input_order(self):
        cfg = _cfg()
        reqs = core.gen_requests(cfg["sample_seed"], cfg["request_count"])
        forward = core.chunk_batches(reqs, cfg["batch_size"])
        shuffled = core.chunk_batches(list(reversed(reqs)), cfg["batch_size"])
        self.assertEqual(forward, shuffled)

    def test_output_digest_independent_of_execution_order(self):
        cfg = _cfg()
        reqs = core.gen_requests(cfg["sample_seed"], cfg["request_count"])
        batches = core.chunk_batches(reqs, cfg["batch_size"])
        digests_forward = [core.compute_batch(cfg, "w0", 1, bid, list(rids))[0]
                           for bid, rids in batches]
        digests_reverse = [core.compute_batch(cfg, "w0", 1, bid, list(rids))[0]
                           for bid, rids in reversed(batches)]
        self.assertEqual(digests_forward, list(reversed(digests_reverse)))

    def test_output_digest_binds_request_set_and_version(self):
        cfg = _cfg()
        rids = ["req-0000", "req-0001"]
        d1, _ = core.compute_batch(cfg, "w0", 1, "batch-0000", rids)
        d2, _ = core.compute_batch(cfg, "w0", 2, "batch-0000", rids)
        d3, _ = core.compute_batch(cfg, "w0", 1, "batch-0000", ["req-0000"])
        self.assertNotEqual(d1, d2)
        self.assertNotEqual(d1, d3)
        d4, _ = core.compute_batch(cfg, "w1", 1, "batch-0000", rids)
        self.assertNotEqual(d1, d4)

    def test_receipt_lines_stable_rank(self):
        cfg = _cfg()
        lines = core.compute_receipt_lines(cfg, "w0", 1, "batch-0000",
                                           ["req-0001", "req-0000"])
        parts = [line.split("|") for line in lines]
        self.assertEqual(parts[0][1], "batch-0000")
        self.assertEqual(parts[0][2], "req-0000")
        self.assertEqual(parts[0][6], "0")
        self.assertEqual(parts[1][2], "req-0001")
        self.assertEqual(parts[1][6], "1")


if __name__ == "__main__":
    unittest.main()
