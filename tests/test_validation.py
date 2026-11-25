from __future__ import annotations

import copy
import unittest
from pathlib import Path

from mgirecon.commands import apply_commands
from mgirecon.config import load_config, save_config, validate_config
from mgirecon.errors import E_PARAM_INVALID, E_PNG_INVALID, E_VIEW_UNKNOWN, E_VOXEL_ALIGN
from mgirecon.pngio import PngWriter
from mgirecon.synth import initialize_project

from tests.helpers import copy_project, make_project


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary, cls.root, cls.config = make_project("mgirecon-validation-")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_default_project_is_valid(self) -> None:
        validation = validate_config(self.config, self.root)
        self.assertEqual(validation["errors"], [])
        self.assertEqual([view["id"] for view in validation["valid_views"]], ["v01", "v02", "v03", "v04"])
        self.assertTrue(validation["voxel_aligned"])

    def test_bad_png_is_isolated_and_other_views_continue(self) -> None:
        temporary, root = copy_project(self.root)
        try:
            image_path = root / "views" / "v02.png"
            data = bytearray(image_path.read_bytes())
            data[0:8] = b"NOTAPNG!"
            image_path.write_bytes(bytes(data))
            validation = validate_config(load_config(root), root)
            self.assertEqual([error["code"] for error in validation["errors"]], [E_PNG_INVALID])
            self.assertEqual([view["id"] for view in validation["valid_views"]], ["v01", "v03", "v04"])
        finally:
            temporary.cleanup()

    def test_fixed_resolution_mismatch_is_rejected(self) -> None:
        temporary, root = copy_project(self.root)
        try:
            wrong = PngWriter(64, 48)
            import numpy as np

            wrong.write(root / "views" / "v03.png", np.zeros((48, 64, 3), dtype=np.uint8))
            validation = validate_config(load_config(root), root)
            self.assertIn(E_PARAM_INVALID, [error["code"] for error in validation["errors"]])
            self.assertNotIn("v03", [view["id"] for view in validation["valid_views"]])
            self.assertEqual([view["id"] for view in validation["valid_views"]], ["v01", "v02", "v04"])
        finally:
            temporary.cleanup()

    def test_duplicate_view_and_bad_camera_are_rejected(self) -> None:
        temporary, root = copy_project(self.root)
        try:
            duplicate = copy.deepcopy(self.config)
            duplicate["views"][1]["id"] = "v01"
            save_config(root, duplicate)
            validation = validate_config(duplicate, root)
            self.assertIn(E_PARAM_INVALID, [error["code"] for error in validation["errors"]])

            singular = copy.deepcopy(self.config)
            singular["views"][0]["intrinsics"] = [[1, 2, 3], [2, 4, 6], [0, 0, 1]]
            save_config(root, singular)
            validation = validate_config(singular, root)
            self.assertIn(E_PARAM_INVALID, [error["code"] for error in validation["errors"]])
            self.assertNotIn("v01", [view["id"] for view in validation["valid_views"]])

            nonorthogonal = copy.deepcopy(self.config)
            nonorthogonal["views"][0]["extrinsics"] = [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
            save_config(root, nonorthogonal)
            validation = validate_config(nonorthogonal, root)
            self.assertIn(E_PARAM_INVALID, [error["code"] for error in validation["errors"]])
        finally:
            temporary.cleanup()

    def test_positive_parameter_and_alignment_errors(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["reconstruction"]["min_disparity"] = 0
        invalid["reconstruction"]["residual_threshold"] = -2
        validation = validate_config(invalid, self.root)
        self.assertGreaterEqual(
            sum(error["code"] == E_PARAM_INVALID for error in validation["errors"]),
            2,
        )

        misaligned = copy.deepcopy(self.config)
        misaligned["reconstruction"]["voxel_size"] = 64
        validation = validate_config(misaligned, self.root)
        self.assertIn(E_VOXEL_ALIGN, [error["code"] for error in validation["errors"]])
        self.assertFalse(validation["voxel_aligned"])

    def test_commands_report_unknown_view_and_continue(self) -> None:
        temporary, root, config = make_project("mgirecon-commands-")
        try:
            updated, errors = apply_commands(
                root,
                config,
                [
                    {"op": "replace_view", "view": "v99", "render_seed": 1},
                    {"op": "set_param", "name": "residual_threshold", "value": 17},
                    {"op": "set_param", "name": "min_disparity", "value": -1},
                ],
            )
            self.assertEqual([error["code"] for error in errors], [E_VIEW_UNKNOWN, E_PARAM_INVALID])
            self.assertEqual(updated["reconstruction"]["residual_threshold"], 17)
            self.assertEqual(load_config(root)["reconstruction"]["residual_threshold"], 17)
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
