from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from level_workbench.cli import main
from level_workbench.samples import normal_spec


class CliTests(unittest.TestCase):
    def test_generate_and_verify_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec_path = root / "spec.yaml"
            out_path = root / "generated"
            spec_path.write_text(yaml.safe_dump(normal_spec(), sort_keys=True), encoding="utf-8")

            generate_stdout = io.StringIO()
            with redirect_stdout(generate_stdout):
                generate_code = main(
                    ["generate", "--spec", str(spec_path), "--out", str(out_path)]
                )
            self.assertEqual(generate_code, 0)
            payload = json.loads(generate_stdout.getvalue())
            self.assertEqual(payload["status"], "pass")
            self.assertTrue((out_path / "level.json").is_file())
            self.assertTrue((out_path / "grid.txt").is_file())

            verify_stdout = io.StringIO()
            with redirect_stdout(verify_stdout):
                verify_code = main(
                    [
                        "verify",
                        "--spec",
                        str(out_path / "effective_spec.yaml"),
                        "--level",
                        str(out_path / "level.json"),
                    ]
                )
            self.assertEqual(verify_code, 0)
            verification = json.loads(verify_stdout.getvalue())
            self.assertEqual(verification["status"], "pass")


if __name__ == "__main__":
    unittest.main()
