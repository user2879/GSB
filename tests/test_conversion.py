import json
import tempfile
import unittest
from pathlib import Path

from feature_pipeline.config import load_feature_metadata
from feature_pipeline.conversion import (
    to_v3,
    v1_to_v2,
    v2_to_v1,
    v3_to_v1,
    v3_to_v2,
)
from feature_pipeline.models import V1, V2, V3, Cell, EntityTable


METADATA_CSV = (
    "feature,semantic_type,allowed_precisions\n"
    "f_a,continuous,per_mille|float32|float64\n"
    "f_b,ratio,float32|float64\n"
)


class ConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        meta_path = self.dir / "meta.csv"
        meta_path.write_text(METADATA_CSV, encoding="utf-8")
        self.metadata = load_feature_metadata(meta_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_v1_to_v2_exact_decimal_scaling_and_label(self) -> None:
        table = EntityTable(source_version=V1, feature_order=["f_a"])
        table.cells.append(Cell("e1", "f_a", V1, 1.234, False, "per_mille"))
        v2 = v1_to_v2(table, self.metadata)
        self.assertEqual(v2.source_version, V2)
        self.assertEqual(v2.cells[0].value, 1.234)
        self.assertIn(v2.cells[0].precision_label, ("float32", "float64"))

    def test_v2_to_v1_records_residuals_and_source_marker(self) -> None:
        table = EntityTable(source_version=V2, feature_order=["f_a"])
        table.cells.append(Cell("e1", "f_a", V2, 1.2349, False, "float64"))
        sidecar = self.dir / "residual.json"
        v1 = v2_to_v1(table, sidecar)
        self.assertEqual(v1.cells[0].value, 1.235)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        record = payload["records"]["e1|f_a"]
        self.assertAlmostEqual(record["truncation_residual"], 1.2349 - 1.235)
        self.assertEqual(record["source_marker"], "v2_to_v1_downgrade")
        self.assertEqual(record["precision_label"], "float64")

    def test_v3_conversion_keeps_boundaries_and_reports_losses(self) -> None:
        table = EntityTable(source_version=V2, feature_order=["f_a", "f_b"])
        table.partition_boundaries = {0: ["e1"], 1: ["e2"]}
        table.cells = [
            Cell("e1", "f_a", V2, 1.23456789, False, "float64", partition_id=0),
            Cell("e2", "f_b", V2, 2.0, False, "float64", partition_id=1),
        ]
        v3, losses = to_v3(table)
        self.assertEqual(v3.partition_boundaries, table.partition_boundaries)
        self.assertEqual(v3.feature_order, ["f_a", "f_b"])
        self.assertTrue(any(
            item["lost_attribute"] == "float64_precision" for item in losses
        ))

        v2, v2_losses = v3_to_v2(v3)
        self.assertEqual(v2.partition_boundaries, v3.partition_boundaries)
        self.assertTrue(any(
            item["lost_attribute"] == "partition_offsets" for item in v2_losses
        ))

        sidecar = self.dir / "v3_v1.json"
        v1, v1_losses = v3_to_v1(v3, sidecar)
        self.assertEqual(v1.source_version, V1)
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        self.assertIn("partition_boundaries", sidecar_payload)
        self.assertTrue(any(
            item["lost_attribute"] == "partition_structure"
            for item in v1_losses
        ))


if __name__ == "__main__":
    unittest.main()
