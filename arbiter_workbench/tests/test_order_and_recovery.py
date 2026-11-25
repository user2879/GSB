import shutil
import tempfile
import unittest
from pathlib import Path

from arbiter.engine import ArbitrationEngine
from arbiter.errors import OrderError
from arbiter.ledger import KeyRecord, Ledger
from arbiter.protocol import pick_free_port
from arbiter.recovery import inspect_db
from arbiter.sample import DEFAULT_KEYS
from arbiter.service_manager import ServiceManager
from arbiter.spec import EnvSpec, load_declarations

DECLS = Path(__file__).resolve().parents[1] / "decls"


class OrderGuardTest(unittest.TestCase):
    def test_byte_identical_runs_pass(self):
        runs = [
            {"v1": {"records": [{"key": "k", "authority_q": 1, "adopted_envs": ["a"]}]}},
            {"v1": {"records": [{"key": "k", "authority_q": 1, "adopted_envs": ["a"]}]}},
        ]
        ArbitrationEngine.assert_byte_identical(runs)

    def test_diverging_run_raises_e_order_with_location(self):
        runs = [
            {"v1": {"records": [{"key": "k1", "authority_q": 1}]}},
            {"v1": {"records": [{"key": "k1", "authority_q": 2}]}},
        ]
        with self.assertRaises(OrderError) as ctx:
            ArbitrationEngine.assert_byte_identical(runs)
        self.assertEqual(ctx.exception.code, "E_ORDER")
        self.assertEqual(ctx.exception.context.object_id, "v1")
        self.assertEqual(ctx.exception.context.field, "k1")


class InspectDbTest(unittest.TestCase):
    def test_inspect_reports_unknown_view(self):
        tmp = Path(tempfile.mkdtemp(prefix="wb_inspect_"))
        try:
            ledger = Ledger(tmp / "ledger.sqlite", session_scale=1)
            ledger.begin_unknown("view-x", 7)
            ledger.close()
            latest = inspect_db(tmp / "ledger.sqlite")
            self.assertEqual(latest["view-x"]["status"], "UNKNOWN")
            self.assertEqual(latest["view-x"]["version"], 7)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
