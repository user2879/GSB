from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bucketpack.view import collect_view_model

from .common import ROOT


class ViewModelTests(unittest.TestCase):
    def test_view_model_is_stably_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model = collect_view_model(
                spec_path=ROOT / "samples" / "spec.json",
                changes_path=ROOT / "samples" / "changes_value.json",
                ledger_path=Path(td) / "missing.sqlite",
                log_path=Path(td) / "missing.log",
            )
        self.assertGreater(model["summary"]["pack_count"], 0)
        self.assertEqual(
            model["dependencies"],
            sorted(model["dependencies"], key=lambda item: (item["from"], item["to"])),
        )
        self.assertEqual(
            [pack["pack_id"] for pack in model["packs"]],
            sorted(pack["pack_id"] for pack in model["packs"]),
        )
        self.assertTrue(model["invalidation"]["bucket_columns"])
        self.assertEqual(model["ledger"]["batches"], [])


if __name__ == "__main__":
    unittest.main()
