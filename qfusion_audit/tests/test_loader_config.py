import json
import os
import tempfile
import unittest

from qfusion.config import load_config
from qfusion.errors import E_CONFIG_INVALID, E_RECORD_INVALID
from qfusion.loader import load_stream
from qfusion.sample_generator import generate


class LoaderConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        info = generate(self.tmp)
        self.cfg, cfg_failures = load_config(info["config"])
        self.assertEqual(cfg_failures, [])
        self.stream_path = info["stream"]
        self.invalid_path = info["invalid"]

    def test_valid_stream_loads_without_failure(self):
        stream = load_stream(self.stream_path, self.cfg)
        self.assertEqual(stream.failures, [])
        self.assertTrue(stream.request_ids)
        # 通道内名次为正整数且唯一（加载成功即已校验）。
        for req in stream.requests.values():
            for chan, recs in req.items():
                ranks = [r.channel_rank for r in recs]
                self.assertEqual(len(ranks), len(set(ranks)))
                self.assertTrue(all(r > 0 for r in ranks))

    def test_invalid_records_collected_not_raised(self):
        stream = load_stream(self.invalid_path, self.cfg)
        codes = {f.code for f in stream.failures}
        self.assertEqual(codes, {E_RECORD_INVALID})
        self.assertTrue(len(stream.failures) >= 2)

    def test_duplicate_item_and_rank_detected(self):
        cfg = self.cfg
        base = {
            "request_id": "r", "channel": cfg.channel_names_sorted[0],
            "raw_score": 0.5, "features": [0.0] * cfg.feature_dim,
            "shard": 0}
        rows = [
            dict(base, item_id="a", channel_rank=1),
            dict(base, item_id="a", channel_rank=2),
            dict(base, item_id="b", channel_rank=1),
            dict(base, item_id="c", channel_rank=-1),
            dict(base, item_id="d", channel_rank=3,
                 features=[0.0] * (cfg.feature_dim - 1)),
        ]
        path = os.path.join(self.tmp, "dup.jsonl")
        with open(path, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        stream = load_stream(path, cfg)
        # 仅首行 (a, rank=1) 合法，其余因重复/非法名次/特征维度被拒。
        self.assertEqual(len(stream.records), 1)
        self.assertTrue(all(f.code == E_RECORD_INVALID
                            for f in stream.failures))
        self.assertEqual(len(stream.failures), 4)

    def test_config_validation_monotonic_sets(self):
        bad = dict(self.cfg.raw)
        bad["gears"] = [4, 2, 1]
        path = os.path.join(self.tmp, "bad_config.json")
        with open(path, "w") as fh:
            json.dump(bad, fh)
        cfg, failures = load_config(path)
        self.assertIsNone(cfg)
        self.assertTrue(any(f.code == E_CONFIG_INVALID for f in failures))

    def test_config_empty_sets_rejected(self):
        bad = dict(self.cfg.raw)
        bad["frac_bits_choices"] = []
        path = os.path.join(self.tmp, "empty_fb.json")
        with open(path, "w") as fh:
            json.dump(bad, fh)
        cfg, failures = load_config(path)
        self.assertIsNone(cfg)
        self.assertTrue(any(f.code == E_CONFIG_INVALID for f in failures))


if __name__ == "__main__":
    unittest.main()
