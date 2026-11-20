from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from edgecache.cli import main

from tests.helpers import free_port_base


class CliInjectionTests(unittest.TestCase):
    def test_all_required_anomaly_codes(self) -> None:
        base = free_port_base(4, 45600)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                "cycle": "E_CYCLE",
                "content_truncation": "E_CONTENT",
                "snapshot_digest_mismatch": "E_SNAPSHOT",
            }
            for injection, code in expected.items():
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = main(
                        [
                            "demo",
                            "--workspace",
                            str(root / injection),
                            "--port-base",
                            str(base),
                            "--inject",
                            injection,
                        ]
                    )
                value = json.loads(output.getvalue())
                self.assertEqual(2, status)
                self.assertFalse(value["ok"])
                self.assertEqual(code, value["code"])
                self.assertEqual(injection, value["injection"])


if __name__ == "__main__":
    unittest.main()
