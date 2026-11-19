from __future__ import annotations

import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

from tests.helpers import ROOT


class SampleTests(unittest.TestCase):
    def test_generator_is_deterministic(self) -> None:
        script = ROOT / "samples" / "generate_samples.py"
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True, timeout=30)
        first = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "samples" / "generated").glob("*.json"))
        }
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True, timeout=30)
        second = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "samples" / "generated").glob("*.json"))
        }
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
