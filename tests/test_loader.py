import json
import tempfile
import unittest
from pathlib import Path

from lease_diag.errors import E_POOL, E_ROLLBACK, LeaseError
from lease_diag.loader import load_workspace
from lease_diag.samples import write_samples


def _write(workdir: Path, pools_doc: dict, rows: list[dict]):
    pools = workdir / "pools.json"
    requests = workdir / "requests.ndjson"
    pools.write_text(json.dumps(pools_doc))
    requests.write_text("\n".join(json.dumps(row) for row in rows))
    return pools, requests


GOOD_DOC = {
    "pools": [
        {"pool_id": "P1", "capacity": 2, "lease_ms": 100, "renew_ms": 30,
         "priority": 1, "quota": 2},
    ],
    "rollback": [{"key": "k", "from_rev": 2, "target_rev": 1}],
}
GOOD_ROWS = [
    {"request_id": "r1", "key": "k", "pool_id": "P1", "revision": 1,
     "priority": 3, "arrive_ms": 0},
    {"request_id": "r2", "key": "k", "pool_id": "P1", "revision": 2,
     "priority": 3, "arrive_ms": 10},
]


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_workspace(self):
        pools, requests = _write(self.workdir, GOOD_DOC, GOOD_ROWS)
        workspace = load_workspace(pools, requests)
        self.assertIn("P1", workspace["pools"])
        self.assertEqual(workspace["rollback_targets"]["k"], {"from_rev": 2, "target_rev": 1})

    def test_duplicate_pool_rejected_with_location(self):
        doc = json.loads(json.dumps(GOOD_DOC))
        doc["pools"].append(dict(doc["pools"][0]))
        pools, requests = _write(self.workdir, doc, GOOD_ROWS)
        with self.assertRaises(LeaseError) as ctx:
            load_workspace(pools, requests)
        self.assertEqual(ctx.exception.code, E_POOL)
        self.assertTrue(any("duplicate pool_id" in loc
                            for loc in ctx.exception.locations))

    def test_renew_interval_must_be_less_than_lease(self):
        doc = json.loads(json.dumps(GOOD_DOC))
        doc["pools"][0]["renew_ms"] = 100
        pools, requests = _write(self.workdir, doc, GOOD_ROWS)
        with self.assertRaises(LeaseError) as ctx:
            load_workspace(pools, requests)
        self.assertEqual(ctx.exception.code, E_POOL)

    def test_non_positive_capacity_rejected(self):
        doc = json.loads(json.dumps(GOOD_DOC))
        doc["pools"][0]["capacity"] = 0
        pools, requests = _write(self.workdir, doc, GOOD_ROWS)
        with self.assertRaises(LeaseError) as ctx:
            load_workspace(pools, requests)
        self.assertEqual(ctx.exception.code, E_POOL)

    def test_unknown_pool_and_non_monotonic_arrival(self):
        rows = [
            {"request_id": "r1", "key": "k", "pool_id": "NOPE",
             "revision": 1, "priority": 3, "arrive_ms": 10},
            {"request_id": "r2", "key": "k", "pool_id": "P1",
             "revision": 2, "priority": 3, "arrive_ms": 5},
        ]
        pools, requests = _write(self.workdir, GOOD_DOC, rows)
        with self.assertRaises(LeaseError) as ctx:
            load_workspace(pools, requests)
        self.assertEqual(ctx.exception.code, E_POOL)
        joined = " ".join(ctx.exception.locations)
        self.assertIn("unknown target pool_id", joined)
        self.assertIn("decreases", joined)

    def test_revision_must_be_non_decreasing_per_key(self):
        rows = [
            {"request_id": "r1", "key": "k", "pool_id": "P1",
             "revision": 2, "priority": 3, "arrive_ms": 0},
            {"request_id": "r2", "key": "k", "pool_id": "P1",
             "revision": 1, "priority": 3, "arrive_ms": 5},
        ]
        pools, requests = _write(self.workdir, GOOD_DOC, rows)
        with self.assertRaises(LeaseError) as ctx:
            load_workspace(pools, requests)
        self.assertEqual(ctx.exception.code, E_POOL)
        self.assertIn("decreases", " ".join(ctx.exception.locations))

    def test_rollback_target_must_exist(self):
        doc = json.loads(json.dumps(GOOD_DOC))
        doc["rollback"][0]["target_rev"] = 99
        pools, requests = _write(self.workdir, doc, GOOD_ROWS)
        with self.assertRaises(LeaseError) as ctx:
            load_workspace(pools, requests)
        self.assertEqual(ctx.exception.code, E_ROLLBACK)

    def test_rollback_cycle_rejected(self):
        info = write_samples(self.workdir, "rollback_cycle")
        with self.assertRaises(LeaseError) as ctx:
            load_workspace(info["pools_path"], info["requests_path"])
        self.assertEqual(ctx.exception.code, E_ROLLBACK)
        self.assertIn("cyclic rollback chain",
                      " ".join(ctx.exception.locations))


if __name__ == "__main__":
    unittest.main()
