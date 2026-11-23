from __future__ import annotations

import json
import os
import py_compile
import subprocess
import sys
import unittest

from config_diag.injections import run_injection

from .support import ROOT, SAMPLES, local_temp


class InjectionAndCliTests(unittest.TestCase):
    def test_three_exception_injections_report_codes_and_append_jsonl_logs(self) -> None:
        with local_temp("injections-") as tmp:
            results = {
                kind: run_injection(kind, SAMPLES, tmp)
                for kind in ("spec", "cycle", "snapshot")
            }
            self.assertEqual(results["spec"]["diagnostic"]["code"], "E_SPEC")
            self.assertEqual(results["spec"]["diagnostic"]["key"], "pricing.base_fee")
            self.assertEqual(results["spec"]["diagnostic"]["field"], "ttl_ms")
            self.assertEqual(results["cycle"]["diagnostic"]["code"], "E_CYCLE")
            self.assertEqual(results["cycle"]["diagnostic"]["key"], "pricing.net_fee")
            self.assertEqual(results["snapshot"]["diagnostic"]["code"], "E_SNAPSHOT")
            self.assertTrue(results["snapshot"]["backup_readonly"])
            log = tmp / "logs" / "diagnostics.jsonl"
            lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            codes = {line["code"] for line in lines}
            self.assertTrue({"E_SPEC", "E_CYCLE", "E_SNAPSHOT"}.issubset(codes))
            self.assertTrue(all("context" in line for line in lines))

    def test_cli_compare_injection_reports_e_diverge_and_logs_it(self) -> None:
        with local_temp("cli-diverge-") as tmp:
            env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "config_diag",
                    "--workspace",
                    str(tmp),
                    "--samples",
                    str(SAMPLES),
                    "compare",
                    "--inject-diverge",
                    "--now-ns",
                    "1000000000",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(run.returncode, 1, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(payload["code"], "E_DIVERGE")
            self.assertEqual(payload["field"], "value")
            log = tmp / "logs" / "diagnostics.jsonl"
            codes = [json.loads(line)["code"] for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertIn("E_DIVERGE", codes)

    def test_gui_module_compiles_without_importing_tk_at_module_load(self) -> None:
        target = ROOT / "config_diag" / "gui.py"
        py_compile.compile(str(target), doraise=True)
        source = target.read_text(encoding="utf-8")
        before_function = source.split("def run_gui", 1)[0]
        self.assertNotIn("import tkinter", before_function)


if __name__ == "__main__":
    unittest.main()
