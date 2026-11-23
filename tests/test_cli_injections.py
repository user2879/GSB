from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.support import ROOT, copy_samples


class CliInjectionTests(unittest.TestCase):
    def _command(self, root: Path, *extra: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "config_arbiter",
            *extra,
            "--env",
            str(root / "environments.json"),
            "--views",
            str(root / "views.json"),
            "--db",
            str(root / "ledger.sqlite"),
            "--snapshot",
            str(root / "snapshot.json"),
            "--log",
            str(root / "log.jsonl"),
        ]

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_run_status_replay_restore_and_three_injections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copy_samples(root)
            env_hash = self._hash(root / "environments.json")
            views_hash = self._hash(root / "views.json")

            run = subprocess.run(self._command(root, "run"), cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            run_payload = json.loads(run.stdout)
            self.assertEqual(len(run_payload["run"]["completed_views"]), 3)

            status = subprocess.run(self._command(root, "status"), cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(status.returncode, 0)
            self.assertEqual(json.loads(status.stdout)["report"]["unknown_views"], [])

            replay = subprocess.run(self._command(root, "replay"), cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(replay.returncode, 0)
            self.assertEqual(json.loads(replay.stdout)["replay"]["compared_count"], 3)

            restore = subprocess.run(self._command(root, "restore"), cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(restore.returncode, 0)
            self.assertEqual(json.loads(restore.stdout)["restore"]["mode"], "replayed")

            spec = subprocess.run(
                self._command(root, "run", "--inject-spec-corrupt"),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(spec.returncode, 2)
            self.assertEqual(json.loads(spec.stdout)["code"], "E_SPEC")

            ref = subprocess.run(
                self._command(root, "run", "--inject-priority-ref-missing-env"),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(ref.returncode, 3)
            self.assertEqual(json.loads(ref.stdout)["code"], "E_REF")

            snapshot = subprocess.run(
                self._command(root, "status", "--inject-ledger-digest-mismatch"),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(snapshot.returncode, 4)
            self.assertEqual(json.loads(snapshot.stdout)["code"], "E_SNAPSHOT")

            self.assertEqual(self._hash(root / "environments.json"), env_hash)
            self.assertEqual(self._hash(root / "views.json"), views_hash)
            log_records = [
                json.loads(line) for line in (root / "log.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            failed_codes = [record.get("code") for record in log_records if record["event"] == "command_failed"]
            self.assertEqual(failed_codes, ["E_SPEC", "E_REF", "E_SNAPSHOT"])
            self.assertTrue(all("timestamp" in record for record in log_records))


if __name__ == "__main__":
    unittest.main()
