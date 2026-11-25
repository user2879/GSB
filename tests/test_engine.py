import random
import unittest

from arbplatform.engine import ArbitrationEngine
from arbplatform.spec import parse_manifest


def manifest(policy="latest_revision", tolerance=0, scale=2):
    return parse_manifest({
        "scale_power": scale,
        "global_concurrency": 3,
        "environments": [
            {"env_id": "e1", "host": "127.0.0.1", "port": 49011,
             "lease_capacity": 2, "seed": 1, "weight": 3},
            {"env_id": "e2", "host": "127.0.0.1", "port": 49012,
             "lease_capacity": 1, "seed": 2, "weight": 2},
            {"env_id": "e3", "host": "127.0.0.1", "port": 49013,
             "lease_capacity": 2, "seed": 3, "weight": 1},
        ],
        "instances": [
            {"instance_id": "i1", "env_ref": "e1", "seq_low": 1,
             "seq_high": 3, "time_base_ms": 1},
            {"instance_id": "i2", "env_ref": "e2", "seq_low": 1,
             "seq_high": 3, "time_base_ms": 2},
            {"instance_id": "i3", "env_ref": "e3", "seq_low": 1,
             "seq_high": 3, "time_base_ms": 3},
        ],
        "views": [
            {"view_id": "v", "env_refs": ["e1", "e2", "e3"],
             "policies": {"k": policy}, "tolerance": tolerance}
        ],
    })


def obs(env, instance, seq, value, key="k"):
    return {"obs_id": f"{env}:{instance}:{seq}:{key}", "env_id": env,
            "instance_id": instance, "seq": seq,
            "gen_time_ms": seq * 10, "key": key, "value": value}


def scalar_set():
    return [
        obs("e1", "i1", 1, 10), obs("e1", "i1", 2, 12), obs("e1", "i1", 3, 12),
        obs("e2", "i2", 1, 12), obs("e2", "i2", 2, 12), obs("e2", "i2", 3, 5),
        obs("e3", "i3", 1, 9), obs("e3", "i3", 2, 9), obs("e3", "i3", 3, 9),
    ]


class EnginePolicyTest(unittest.TestCase):
    def test_latest_revision_winner(self):
        m = manifest("latest_revision")
        engine = ArbitrationEngine(m)
        result = engine.arbitrate_view(m.view("v"), 1, scalar_set())
        verdict = result.verdicts["k"]
        # Highest revision seq=3: e1=12, e2=5, e3=9 -> smallest q tie => 5 (e2)
        self.assertEqual(verdict.authority_text, "5.00")
        self.assertIn("e2", verdict.adopted)

    def test_weighted_majority_winner(self):
        m = manifest("weighted_majority")
        engine = ArbitrationEngine(m)
        result = engine.arbitrate_view(m.view("v"), 1, scalar_set())
        verdict = result.verdicts["k"]
        # seq3 representatives: e1=12 (w3), e2=5 (w2), e3=9 (w1): 12 wins weight 3
        self.assertEqual(verdict.authority_text, "12.00")
        self.assertEqual(verdict.adopted, ("e1",))
        self.assertEqual(verdict.rejected, ("e2", "e3"))

    def test_interval_intersection_midpoint(self):
        m = manifest("interval_midpoint")
        engine = ArbitrationEngine(m)
        data = [
            obs("e1", "i1", 1, [0, 10]),
            obs("e2", "i2", 1, [4, 12]),
            obs("e3", "i3", 1, [6, 20]),
        ]
        result = engine.arbitrate_view(m.view("v"), 1, data)
        verdict = result.verdicts["k"]
        # intersection [6,10], midpoint 8
        self.assertEqual(verdict.interval, [6, 10])
        self.assertEqual(verdict.authority_text, "8.00")
        self.assertIn("e3", verdict.adopted)

    def test_interval_conflict_flagged(self):
        m = manifest("interval_midpoint")
        engine = ArbitrationEngine(m)
        data = [
            obs("e1", "i1", 1, [0, 4]),
            obs("e2", "i2", 1, [10, 20]),
        ]
        result = engine.arbitrate_view(m.view("v"), 1, data)
        verdict = result.verdicts["k"]
        self.assertIn("policy_conflict", verdict.causes)
        self.assertEqual(verdict.adopted, ())

    def test_order_and_batch_independent_byte_identical(self):
        m = manifest("weighted_majority")
        engine = ArbitrationEngine(m)
        base = scalar_set()
        digests = set()
        rng = random.Random(99)
        for _ in range(8):
            shuffled = base[:]
            rng.shuffle(shuffled)
            result = engine.arbitrate_view(m.view("v"), 1, shuffled)
            digests.add(result.content_digest)
        self.assertEqual(len(digests), 1)

    def test_revision_chain_uses_latest_per_instance(self):
        m = manifest("latest_revision")
        engine = ArbitrationEngine(m)
        result = engine.arbitrate_view(m.view("v"), 1, scalar_set())
        chain = engine.build_chains(scalar_set(), ("e1",), "k")[0]
        self.assertEqual(chain.seq_low, 1)
        self.assertEqual(chain.seq_high, 3)
        self.assertEqual(chain.representative()["value"], 12)
        self.assertIsNotNone(result.content_digest)


if __name__ == "__main__":
    unittest.main()
