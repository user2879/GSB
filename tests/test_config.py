import tempfile
import unittest
from pathlib import Path

import yaml

from quantcalib.data.config import load_config
from quantcalib.errors import E_CONFIG_INVALID


class ConfigTests(unittest.TestCase):
    def test_invalid_permille_and_rounding(self):
        cases = [
            {"q_lo_permille": 800, "q_hi_permille": 200},
            {"rounding_mode": "toward_sky"},
            {"saturation_policy": "explode"},
            {"max_workers": 0},
            {"unknown_key": 1},
        ]
        for payload in cases:
            with tempfile.NamedTemporaryFile(
                    "w", suffix=".yaml", delete=False,
                    encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle)
                path = handle.name
            cfg, issues = load_config(path)
            self.assertIsNone(cfg)
            self.assertTrue(issues)
            self.assertEqual(issues[0].code, E_CONFIG_INVALID)


if __name__ == "__main__":
    unittest.main()
