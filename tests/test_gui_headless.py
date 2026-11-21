import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hotgray.gateway import Gateway
from hotgray.gui import build_view_data, run_authoritative
from hotgray.samples import build_sample_declaration


class GuiHeadlessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hotgray-gui-")
        cls.decl_path = os.path.join(cls.tmp, "decl.json")
        cls.db_path = os.path.join(cls.tmp, "ledger.sqlite3")
        with open(cls.decl_path, "w", encoding="utf-8") as handle:
            json.dump(build_sample_declaration(18090), handle)
        cls.gateway = Gateway(cls.decl_path, cls.db_path)
        cls.gateway.start()
        for request in cls.gateway.declaration["request_stream"]:
            cls.gateway.admit_request(request)
        cls.gateway.flush_all()
        cls.gateway.hotswap_gray_backends()
        for request in cls.gateway.declaration["request_stream"]:
            cls.gateway.admit_request({
                "sample_id": request["sample_id"] + "-b",
                "rule_id": request["rule_id"],
                "client_key": request["client_key"],
            })
        cls.gateway.flush_all()
        cls.gateway.rebuild_snapshot()

    @classmethod
    def tearDownClass(cls):
        cls.gateway.stop()

    def test_fast_view_rows(self):
        data = build_view_data(self.db_path)
        self.assertIsNotNone(data["snapshot"])
        self.assertTrue(data["rows"])
        for row in data["rows"]:
            self.assertIn(row["destination"], ("gray", "stable"))
            self.assertGreater(row["hit_count"], 0)
            self.assertTrue(0 <= row["gray_ratio_bp"] <= 10000)

    def test_authoritative_view_recompute(self):
        verdict = run_authoritative(self.decl_path, self.db_path)
        self.assertEqual(verdict["verdict"], "byte_identical")


if __name__ == "__main__":
    unittest.main()
