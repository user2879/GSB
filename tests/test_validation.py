import json
import tempfile
import unittest
from pathlib import Path

from diagconsole.errors import DiagError
from diagconsole.model import load_spec

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.clusters = json.loads((SAMPLES / "clusters.json").read_text())
        self.tasks = json.loads((SAMPLES / "tasks.json").read_text())
        self.revisions = json.loads((SAMPLES / "revisions.json").read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self):
        c = self.dir / "clusters.json"
        t = self.dir / "tasks.json"
        r = self.dir / "revisions.json"
        c.write_text(json.dumps(self.clusters))
        t.write_text(json.dumps(self.tasks))
        r.write_text(json.dumps(self.revisions))
        return c, t, r

    def test_valid_spec_loads(self):
        c, t, r = self._write()
        spec = load_spec(c, t, r)
        self.assertEqual(len(spec.clusters), 3)
        self.assertEqual(spec.revisions[-1].version, 6)

    def test_duplicate_cluster_id(self):
        self.clusters["clusters"][1]["cluster_id"] = \
            self.clusters["clusters"][0]["cluster_id"]
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("clusters[1]", ctx.exception.location["field"])

    def test_non_loopback_host_and_duplicate_port(self):
        self.clusters["clusters"][0]["host"] = "10.0.0.1"
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.clusters = json.loads((SAMPLES / "clusters.json").read_text())
        self.clusters["clusters"][1]["port"] = self.clusters["clusters"][0]["port"]
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")

    def test_non_monotonic_versions(self):
        self.revisions["revisions"][2]["version"] = 2
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.location["revision"], 2)

    def test_missing_parent(self):
        self.revisions["revisions"][1]["parent"] = 99
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertIn("parent", ctx.exception.location["field"])

    def test_derives_to_later_revision(self):
        self.revisions["revisions"][1]["derives"] = [3]
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.location["revision"], 2)

    def test_task_key_never_in_chain(self):
        self.tasks["tasks"][0]["keys"] = ["rate_limit", "ghost_key"]
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")
        self.assertEqual(ctx.exception.location["key"], "ghost_key")

    def test_duplicate_task_id(self):
        self.tasks["tasks"][1]["task_id"] = self.tasks["tasks"][0]["task_id"]
        c, t, r = self._write()
        with self.assertRaises(DiagError) as ctx:
            load_spec(c, t, r)
        self.assertEqual(ctx.exception.code, "E_SPEC")


if __name__ == "__main__":
    unittest.main()
