"""Requirement 5: parameter/structural validation codes and write isolation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from dbench.config import load_config
from dbench.errors import (
    E_CONFIG_INVALID,
    E_GUIDE_NOT_FINITE,
    E_NOISE_TABLE_MISMATCH,
    E_PATCH_OUT_OF_RANGE,
    E_PATCH_SHAPE_MISMATCH,
    E_SEED_OUT_OF_RANGE,
    E_STEP_ZERO,
    WorkbenchError,
)
from dbench.store import SnapshotStore

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG_DIR = os.path.join(ROOT, "configs")

CASES = {
    "invalid_missing_field.json": E_CONFIG_INVALID,
    "invalid_type.json": E_CONFIG_INVALID,
    "invalid_seed_range.json": E_SEED_OUT_OF_RANGE,
    "invalid_guide.json": E_GUIDE_NOT_FINITE,
    "invalid_step_zero.json": E_STEP_ZERO,
    "invalid_noise_length.json": E_NOISE_TABLE_MISMATCH,
    "invalid_patch_range.json": E_PATCH_OUT_OF_RANGE,
    "invalid_patch_shape.json": E_PATCH_SHAPE_MISMATCH,
}


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dbench-val-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_each_invalid_config_maps_to_expected_code(self):
        for fname, code in CASES.items():
            with self.subTest(file=fname):
                with self.assertRaises(WorkbenchError) as ctx:
                    load_config(os.path.join(CFG_DIR, fname))
                self.assertEqual(ctx.exception.code, code)

    def test_failed_validation_writes_nothing(self):
        store = SnapshotStore(os.path.join(self.tmp, "store"))
        before = sorted(os.listdir(store.root))
        with self.assertRaises(WorkbenchError):
            load_config(os.path.join(CFG_DIR, "invalid_seed_range.json"))
        self.assertEqual(before, sorted(os.listdir(store.root)))


if __name__ == "__main__":
    unittest.main()
