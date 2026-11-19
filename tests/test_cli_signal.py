from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

from dp_grad_sim.samplegen import generate_sample


@unittest.skipIf(os.name == "nt", "POSIX signal delivery is used to validate graceful shutdown")
class CliSignalTests(unittest.TestCase):
    def test_sigint_finishes_current_window_and_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = root / "sample"
            generate_sample(
                sample,
                workers=3,
                steps=4,
                layers=2,
                hidden_dim=8,
                topology_type="ring",
                link_bandwidth_bytes=4096,
                window_budget_bytes=65536,
            )
            injection = root / "slow-injection.yaml"
            injection.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "send_delay_ms": 25,
                        "nonfinite": [],
                        "communication_failures": [],
                        "link_budget_overrides": {},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            snapshot_dir = root / "snapshots"
            command = [
                sys.executable,
                "-m",
                "dp_grad_sim",
                "run",
                "--config",
                str(sample / "config.yaml"),
                "--shards",
                str(sample / "shards.jsonl"),
                "--topology",
                str(sample / "topology.json"),
                "--snapshot-dir",
                str(snapshot_dir),
                "--inject",
                str(injection),
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            time.sleep(0.3)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, stderr)
            summary = json.loads(stdout)
            self.assertEqual(summary["status"], "stopped")
            self.assertEqual(summary["stopped_reason"], "signal")
            self.assertLess(summary["completed_steps"], 4)
            self.assertEqual(summary["snapshot_phase"], "tracked")

            resume = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dp_grad_sim",
                    "run",
                    "--config",
                    str(sample / "config.yaml"),
                    "--shards",
                    str(sample / "shards.jsonl"),
                    "--topology",
                    str(sample / "topology.json"),
                    "--snapshot-dir",
                    str(snapshot_dir),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(resume.returncode, 0, resume.stderr)
            resumed_summary = json.loads(resume.stdout)
            self.assertEqual(resumed_summary["status"], "completed")
            self.assertEqual(resumed_summary["completed_steps"], 4)


if __name__ == "__main__":
    unittest.main()

