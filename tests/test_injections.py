from __future__ import annotations

import json
import subprocess
import sys
import unittest

from mlpipe_workbench.errors import E_DATA, E_OUTLIER, E_PARAM, E_SPEC
from mlpipe_workbench.ledger import Ledger

from tests.helpers import DATA, PIPELINE, ROOT, TempDirectoryMixin, base_objects, write_json


class InjectionTests(TempDirectoryMixin, unittest.TestCase):
    def _run(self, *extra: str, db_name: str) -> tuple[int, dict, str]:
        db = self.tempdir / db_name
        log = self.tempdir / (db_name + ".jsonl")
        command = [
            sys.executable,
            "-m",
            "mlpipe_workbench",
            "run",
            "--data",
            str(DATA),
            "--pipeline",
            str(PIPELINE),
            "--db",
            str(db),
            "--log",
            str(log),
            "--run-id",
            "inject-test",
            "--injection-dir",
            str(self.tempdir / "injected"),
            *extra,
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        payload = json.loads(result.stdout)
        log_text = log.read_text(encoding="utf-8") if log.exists() else ""
        return result.returncode, payload, log_text

    def test_three_anomaly_injections_have_distinct_codes_and_reports(self) -> None:
        data_code, data_payload, data_log = self._run(
            "--inject-data-corrupt", db_name="data.db"
        )
        param_code, param_payload, param_log = self._run(
            "--inject-param-bounds", db_name="param.db"
        )
        outlier_code, outlier_payload, outlier_log = self._run(
            "--inject-outlier-ratio", "300", db_name="outlier.db"
        )

        self.assertEqual((data_code, data_payload["code"]), (2, E_DATA))
        self.assertEqual(data_payload["stage_id"], "<dataset>")
        self.assertEqual(data_payload["location"], "data.sources[0].columns[0].nulls")
        self.assertEqual(data_payload["affected_rows"], 83)

        self.assertEqual((param_code, param_payload["code"]), (2, E_PARAM))
        self.assertEqual(param_payload["stage_id"], "clip_amount")
        self.assertEqual(param_payload["param"], "max_outlier_permille")
        self.assertEqual(param_payload["param_value"], 1001)

        self.assertEqual((outlier_code, outlier_payload["code"]), (3, E_OUTLIER))
        self.assertEqual(outlier_payload["stage_id"], "clip_amount")
        self.assertEqual(outlier_payload["param"], "max_outlier_permille")
        self.assertGreater(outlier_payload["affected_rows"], 0)
        self.assertGreater(outlier_payload["details"]["outlier_ratio_permille"], 100)

        self.assertEqual(len({E_DATA, E_PARAM, E_OUTLIER}), 3)
        self.assertIn('"code":"E_DATA"', data_log)
        self.assertIn('"code":"E_PARAM"', param_log)
        self.assertIn('"code":"E_OUTLIER"', outlier_log)

    def test_plain_invalid_spec_rejects_wholesale_without_spec_or_run(self) -> None:
        data, pipeline = base_objects()
        pipeline["stages"][2]["id"] = "fill_missing"
        broken_pipeline = write_json(self.tempdir / "bad_pipeline.json", pipeline)
        db = self.tempdir / "spec.db"
        command = [
            sys.executable,
            "-m",
            "mlpipe_workbench",
            "run",
            "--data",
            str(DATA),
            "--pipeline",
            str(broken_pipeline),
            "--db",
            str(db),
            "--log",
            str(self.tempdir / "spec.jsonl"),
            "--run-id",
            "bad",
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(payload["code"], E_SPEC)
        with Ledger(db) as ledger:
            counts = ledger.table_counts()
            errors = ledger.list_errors()
        self.assertEqual(counts["specs"], 0)
        self.assertEqual(counts["runs"], 0)
        self.assertEqual(errors[0]["code"], E_SPEC)


if __name__ == "__main__":
    unittest.main()
