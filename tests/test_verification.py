from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from colmv.changes import load_changes
from colmv.engine import MaterializationEngine
from colmv.errors import ColmvError
from colmv.replay import verify_paths
from colmv.schema import load_config

from tests.helpers import SAMPLES, sample_paths


class VerificationErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.paths = sample_paths(SAMPLES)
        cls.config = load_config(cls.paths["catalog"], cls.paths["views"])
        cls.changes = load_changes(cls.paths["changes"], cls.config.table)

    def test_e_diverge_locates_first_group_field(self) -> None:
        original = MaterializationEngine.materialized_result

        def altered(self):
            result = original(self)
            if self.path_mode == "rescan":
                result = copy.deepcopy(result)
                result["views"][0]["groups"][0]["row_count"] += 1
            return result

        with patch.object(MaterializationEngine, "materialized_result", altered):
            with self.assertRaises(ColmvError) as context:
                verify_paths(
                    self.config,
                    self.changes,
                    budget=1,
                    batch_sizes=[3, 6, 3, 6, 2],
                )
        error = context.exception
        self.assertEqual(error.code, "E_DIVERGE")
        self.assertEqual(error.view_id, "sales_by_region")
        self.assertIn("row_count", error.field or "")
        self.assertIsNotNone(error.details["first_divergence"]["group_key"])


if __name__ == "__main__":
    unittest.main()
