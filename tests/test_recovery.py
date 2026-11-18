from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests.helpers import MANIFEST_PATH, build_base, working_copy

from dbml.manifest import load_manifest
from dbml.migrations import MigrationRunner


class RecoveryTests(unittest.TestCase):
    def test_hard_kill_resumes_only_unknown_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            base = build_base(directory)
            db = working_copy(base, directory)
            report_path = directory / "should-not-exist.json"
            worker = textwrap.dedent(
                f"""
                import os
                from pathlib import Path
                from dbml.manifest import load_manifest
                from dbml.migrations import MigrationRunner

                manifest = load_manifest({str(MANIFEST_PATH)!r}, {str(db)!r})
                runner = MigrationRunner({str(db)!r}, manifest, batch_size=9)

                def crash(step_id):
                    if step_id == "backfill-total":
                        os._exit(77)

                runner.run(crash_after_step=crash)
                Path({str(report_path)!r}).write_text("complete", encoding="utf-8")
                """
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            process = subprocess.run(
                [sys.executable, "-c", worker],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
            )
            self.assertEqual(process.returncode, 77)
            self.assertFalse(report_path.exists())
            manifest = load_manifest(MANIFEST_PATH, db)
            runner = MigrationRunner(db, manifest)
            status = runner.status()
            self.assertFalse(status["complete"])
            self.assertTrue(status["requires_rerun"])
            self.assertEqual(
                status["committed_steps"], ["widen-unit-price", "backfill-total"]
            )
            self.assertEqual(
                status["unknown_steps"],
                ["backfill-net", "index-customer", "constraint-quantity"],
            )
            resumed = runner.run()
            self.assertTrue(resumed.complete)
            self.assertEqual(len(resumed.committed_steps), 5)


if __name__ == "__main__":
    unittest.main()
