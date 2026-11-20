import json
import shutil
import tempfile
import unittest
from pathlib import Path

from gradcomp.cli import main
from gradcomp.errors import EXIT_CODES


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gc-cli-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, payload: dict) -> Path:
        path = self.tmp / "cfg.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_e_config_exit_code(self):
        path = self._write({"workers": 3, "dimension": 10})  # 不可整除
        rc = main(["summary", str(path)])
        self.assertEqual(rc, EXIT_CODES["E_CONFIG"])

    def test_unknown_field_exit_code(self):
        path = self._write({"not_a_real_field": 1})
        rc = main(["summary", str(path)])
        self.assertEqual(rc, EXIT_CODES["E_CONFIG"])

    def test_summary_success(self):
        path = self._write({"workers": 4, "dimension": 32, "samples": 16,
                            "transport": "loopback"})
        rc = main(["summary", str(path)])
        self.assertEqual(rc, 0)

    def test_train_and_snapshots_command(self):
        cfg_path = self._write({"workers": 4, "dimension": 32, "samples": 16,
                                "transport": "loopback", "steps": 1})
        workdir = self.tmp / "wd"
        self.assertEqual(main(["train", str(cfg_path),
                               "--workdir", str(workdir)]), 0)
        self.assertEqual(main(["snapshots", str(workdir)]), 0)


if __name__ == "__main__":
    unittest.main()
