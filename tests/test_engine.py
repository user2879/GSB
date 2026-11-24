from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path

from config_arbiter.engine import ArbiterEngine
from config_arbiter.ledger import Ledger
from config_arbiter.spec import load_spec

from .util import ported_sample


class EngineServiceFailureTests(unittest.TestCase):
    def test_partial_port_failure_is_recorded_without_losing_other_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            directory = Path(tempdir)
            spec_path, payload = ported_sample(directory)
            occupied_port = payload["environments"][0]["port"]
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", occupied_port))
            listener.listen(1)
            ledger = Ledger(directory / "ledger.sqlite3")
            engine: ArbiterEngine | None = None
            try:
                spec = load_spec(spec_path)
                engine = ArbiterEngine(spec, ledger)
                start_errors = engine.start_services()
                self.assertEqual(start_errors["cn-prod"]["code"], "E_PORT")
                self.assertNotIn("us-prod", start_errors)
                report = engine.run_all()
                self.assertEqual(report.start_errors["cn-prod"]["code"], "E_PORT")
                global_run = next(
                    item for item in report.views if item.summary.view_id == "global-release"
                )
                cn_dispatch = next(
                    item
                    for item in global_run.dispatches
                    if item["environment"] == "cn-prod"
                )
                self.assertEqual(cn_dispatch["error"]["code"], "E_PORT")
                self.assertTrue(all(result.status == "OK" for result in global_run.summary.key_results))
                self.assertTrue(
                    all(result.rejected for result in global_run.summary.key_results)
                )
            finally:
                listener.close()
                if engine is not None:
                    engine.close()
                else:
                    ledger.close()


if __name__ == "__main__":
    unittest.main()
