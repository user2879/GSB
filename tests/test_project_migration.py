import unittest

from voice_repair_workbench import config
from voice_repair_workbench.errors import (
    E_FIELD_MISSING,
    E_PARAM_RANGE,
    E_SCHEMA_VERSION,
    RepairError,
)
from voice_repair_workbench.project import (
    default_v2_project,
    legacy_v1_project,
    load_and_normalize,
    migrate_v1_to_v2,
)


class TestProjectMigration(unittest.TestCase):
    def test_v1_to_v2_idempotent(self):
        v1 = legacy_v1_project()
        once = migrate_v1_to_v2(v1)
        twice = migrate_v1_to_v2(once)
        self.assertEqual(once, twice)
        self.assertEqual(once["schema_version"], config.SCHEMA_V2)
        self.assertEqual(once["gate"], {"threshold_db": -120.0, "ratio": 1.0})
        self.assertEqual(once["normalize"], {"target_peak": 1.0, "max_gain_db": 0.0})
        self.assertEqual(once["limit"], {"ceiling": 1.0})

    def test_unknown_schema_version(self):
        with self.assertRaises(RepairError) as ctx:
            load_and_normalize({"schema_version": 7})
        self.assertEqual(ctx.exception.code, E_SCHEMA_VERSION)
        self.assertEqual(ctx.exception.field, "schema_version")

    def test_immigratable_field_reports_name(self):
        v1 = legacy_v1_project()
        v1["denoise"]["unknown_legacy_knob"] = 3
        with self.assertRaises(RepairError) as ctx:
            migrate_v1_to_v2(v1)
        self.assertEqual(ctx.exception.code, E_SCHEMA_VERSION)
        self.assertEqual(ctx.exception.field, "denoise.unknown_legacy_knob")

    def test_missing_field(self):
        v2 = default_v2_project()
        del v2["normalize"]
        with self.assertRaises(RepairError) as ctx:
            load_and_normalize(v2)
        self.assertEqual(ctx.exception.code, E_FIELD_MISSING)
        self.assertEqual(ctx.exception.field, "normalize")

    def test_param_out_of_range(self):
        v2 = default_v2_project()
        v2["denoise"]["over_subtraction"] = 99.0
        with self.assertRaises(RepairError) as ctx:
            load_and_normalize(v2)
        self.assertEqual(ctx.exception.code, E_PARAM_RANGE)
        self.assertEqual(ctx.exception.field, "denoise.over_subtraction")

    def test_illegal_bit_depth_policy(self):
        v2 = default_v2_project()
        v2["bit_depth_policy"]["output_bit_depth"] = 8
        with self.assertRaises(RepairError) as ctx:
            load_and_normalize(v2)
        self.assertEqual(ctx.exception.code, E_PARAM_RANGE)


if __name__ == "__main__":
    unittest.main()
