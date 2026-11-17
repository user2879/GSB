import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from feature_pipeline.errors import (
    E_LINE_OVERFLOW,
    E_NONFINITE_INPUT,
    E_OFFSET_OUT_OF_RANGE,
    LineOverflowError,
    NonFiniteInputError,
    OffsetOutOfRangeError,
)
from feature_pipeline.models import V1, V2, V3, Cell, EntityTable
from feature_pipeline.parsers import v1_fixedwidth, v2_json, v3_binary


def sample_v1_table() -> EntityTable:
    table = EntityTable(source_version=V1, feature_order=["f_a", "f_b"])
    table.cells = [
        Cell("ent_1", "f_a", V1, 1.234, False, "per_mille"),
        Cell("ent_1", "f_b", V1, float("nan"), True, "per_mille"),
        Cell("ent_2", "f_a", V1, -2.5, False, "per_mille"),
    ]
    return table


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_v1_roundtrip_preserves_permille_and_missing(self) -> None:
        table = sample_v1_table()
        path = self.dir / "v1.txt"
        v1_fixedwidth.write_file(path, table)
        parsed = v1_fixedwidth.parse_file(path)
        self.assertEqual(parsed.feature_order, ["f_a", "f_b"])
        lookup = parsed.cell_map()
        self.assertAlmostEqual(lookup[("ent_1", "f_a")].value, 1.234)
        self.assertTrue(lookup[("ent_1", "f_b")].missing)
        self.assertTrue(math.isnan(lookup[("ent_1", "f_b")].value))
        self.assertEqual(lookup[("ent_2", "f_a")].value, -2.5)
        for line in path.read_text().splitlines()[1:]:
            self.assertEqual(len(line), v1_fixedwidth.LINE_WIDTH)

    def test_v1_line_overflow_business_code(self) -> None:
        path = self.dir / "bad.txt"
        good = self.dir / "good.txt"
        v1_fixedwidth.write_file(good, sample_v1_table())
        text = good.read_text()
        path.write_text(text + "x" * 5, encoding="utf-8")
        with self.assertRaises(LineOverflowError) as ctx:
            v1_fixedwidth.parse_file(path)
        self.assertEqual(ctx.exception.code, E_LINE_OVERFLOW)

    def test_v2_explicit_null_and_absent_key(self) -> None:
        payload = {
            "format_version": "v2",
            "feature_order": ["f_a", "f_b"],
            "entities": {
                "ent_1": {
                    "f_a": {"value": 1.5, "precision": "float64"},
                    "f_b": {"value": None, "precision": "float32",
                            "explicit_null": True},
                },
                "ent_2": {"f_a": {"value": 2.0, "precision": "float32"}},
            },
        }
        path = self.dir / "v2.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        table = v2_json.parse_file(path)
        lookup = table.cell_map()
        self.assertTrue(lookup[("ent_1", "f_b")].missing)
        self.assertTrue(lookup[("ent_2", "f_b")].missing)
        self.assertIn("absent_key", lookup[("ent_2", "f_b")].raw_text)
        self.assertFalse(lookup[("ent_1", "f_a")].missing)

        out = self.dir / "v2_rt.json"
        v2_json.write_file(out, table)
        again = v2_json.parse_file(out)
        self.assertEqual(
            sorted(again.entity_ids()), sorted(table.entity_ids())
        )

    def test_v2_nonfinite_business_code(self) -> None:
        payload = {
            "format_version": "v2",
            "entities": {
                "ent_1": {"f_a": {"value": float("inf"), "precision": "float64"}}
            },
        }
        path = self.dir / "inf.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(NonFiniteInputError) as ctx:
            v2_json.parse_file(path)
        self.assertEqual(ctx.exception.code, E_NONFINITE_INPUT)

    def test_v3_partition_roundtrip_and_offsets(self) -> None:
        table = EntityTable(source_version=V3, feature_order=["f_a", "f_b"])
        entities = [f"ent_{i}" for i in range(5)]
        for pid, group in enumerate((entities[:2], entities[2:])):
            table.partition_boundaries[pid] = group
            for entity_id in group:
                for feature in table.feature_order:
                    missing = entity_id == "ent_4" and feature == "f_b"
                    table.cells.append(
                        Cell(
                            entity_id,
                            feature,
                            V3,
                            float("nan") if missing else 0.75,
                            missing,
                            "float32",
                            partition_id=pid,
                        )
                    )
        path = self.dir / "v3.bin"
        v3_binary.write_file(path, table)
        parsed = v3_binary.parse_file(path)
        self.assertEqual(parsed.partition_boundaries, table.partition_boundaries)
        self.assertEqual(parsed.feature_order, ["f_a", "f_b"])
        lookup = parsed.cell_map()
        self.assertTrue(lookup[("ent_4", "f_b")].missing)
        self.assertEqual(lookup[("ent_0", "f_a")].raw_bytes,
                         struct.pack("<f", 0.75))

    def test_v3_corrupted_offset_business_code(self) -> None:
        table = EntityTable(source_version=V3, feature_order=["f_a"])
        table.partition_boundaries[0] = ["ent_1"]
        table.cells.append(Cell("ent_1", "f_a", V3, 1.0, False, "float32",
                                 partition_id=0))
        path = self.dir / "v3.bin"
        v3_binary.write_file(path, table)
        data = bytearray(path.read_bytes())
        # Locate the first offset slot (after magic/header/name tables) and
        # force it beyond the float32 block.
        for index, byte in enumerate(data):
            if byte == 0 and index > 20:
                data[index] = 0xFF
                break
        bad = self.dir / "bad.bin"
        bad.write_bytes(bytes(data))
        with self.assertRaises((OffsetOutOfRangeError, struct.error)):
            try:
                v3_binary.parse_file(bad)
            except OffsetOutOfRangeError as exc:
                self.assertEqual(exc.code, E_OFFSET_OUT_OF_RANGE)
                raise


if __name__ == "__main__":
    unittest.main()
