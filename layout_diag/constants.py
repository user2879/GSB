"""Shared constants and the fixed business-code vocabulary."""

from __future__ import annotations

SCHEMA_VERSION = 1

FIXED_RESOLUTION = (640, 960)
FIXED_PIXEL_MODE = "L"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

FIXED_BITS_ALLOWED = (4, 6, 8, 10, 12)
QUANT_STEPS_ALLOWED = (1, 2, 4, 8)

DEFAULT_INK_THRESHOLD = 192
DEFAULT_REGRESSION_ALPHA = 0.65
DEFAULT_WINDOW_CUTOFF = 135
DEFAULT_MIN_WINDOW_DENSITY_Q = 4_500
SCORE_SCALE = 1_000_000

ERROR_CODES = (
    "E_DATASET",
    "E_MANIFEST",
    "E_IMAGE_MISSING",
    "E_IMAGE_HEADER",
    "E_IMAGE_SIZE",
    "E_IMAGE_MODE",
    "E_IMAGE_UNLISTED",
    "E_TRUTH",
    "E_KEYPOINT_NOT_INTEGER",
    "E_KEYPOINT_OOB",
    "E_COLUMN_OVERLAP",
    "E_COLUMN_INTERVAL",
    "E_READING_SEQUENCE",
    "E_WINDOW_ALIGN",
    "E_CONFIG_VALUE",
    "E_QUANT_ENUM",
    "E_TIER",
    "E_KEYPOINT_VERSION",
    "E_EDIT_COMMAND",
    "E_POINT_BUDGET_PENDING",
    "E_SNAPSHOT",
)

CONFLICT_TYPES = ("cross_column", "wrong_order", "missing_number")

DEFAULT_STRATEGIES = {
    "small": {
        "window": {"width": 32, "height": 32, "step_x": 16, "step_y": 16},
        "cc": {"merge_gap": 4},
        "precision": {"fixed_bits": 10, "quant_step": 1},
        "capacity_points": 48,
        "capacity_columns": 3,
        "capacity_chars_per_column": 5,
        "single_image_cost": 0.12,
    },
    "medium": {
        "window": {"width": 64, "height": 64, "step_x": 32, "step_y": 32},
        "cc": {"merge_gap": 3},
        "precision": {"fixed_bits": 8, "quant_step": 2},
        "capacity_points": 180,
        "capacity_columns": 4,
        "capacity_chars_per_column": 8,
        "single_image_cost": 0.34,
    },
    "large": {
        "window": {"width": 80, "height": 80, "step_x": 40, "step_y": 40},
        "cc": {"merge_gap": 5},
        "precision": {"fixed_bits": 6, "quant_step": 4},
        "capacity_points": 10_000_000,
        "capacity_columns": 32,
        "capacity_chars_per_column": 128,
        "single_image_cost": 0.91,
    },
}

STAGE_ORDER = {"sliding": 0, "components": 1, "keypoints": 2, "reading_order": 3}
SCOPE_ORDER = {"image": 0, "column": 1, "keypoint": 2, "reading": 3}
