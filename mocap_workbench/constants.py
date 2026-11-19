"""Stable constants shared by the CLI, GUI and verification suite."""

from __future__ import annotations

APP_NAME = "MocapClean Workbench"
APP_VERSION = "1.0.0"
ALGORITHM_VERSION = "mocap-algo-v1"
SCHEMA_RECORD = "mocap-record/v1"
SCHEMA_TRUTH = "mocap-truth/v1"
SCHEMA_DEFECTS = "mocap-defects/v1"
SCHEMA_DETECTION = "mocap-detection/v1"
SCHEMA_REPAIR = "mocap-repair/v1"
SCHEMA_EVALUATION = "mocap-evaluation/v1"
SCHEMA_LEDGER = "mocap-ledger/v1"
SCHEMA_CLOSURE = "mocap-closure/v1"
SCHEMA_CONSISTENCY = "mocap-consistency/v1"

FPS = 60
FRAME_COUNTS = (1200, 6000, 12000)
DEFAULT_SEED = 20261120
FLOAT_DIGITS = 9

# All joint rotations are local-space unit quaternions in (w, x, y, z) order.
SKELETON = (
    "Hips",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftUpperArm",
    "LeftLowerArm",
    "LeftHand",
    "RightShoulder",
    "RightUpperArm",
    "RightLowerArm",
    "RightHand",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
)
JOINT_INDEX = {name: index for index, name in enumerate(SKELETON)}
ROOT_NAME = "Root"
ROOT_INDEX = -1
INVALID_QUAT = (0.0, 0.0, 0.0, 0.0)

DEFECT_JUMP = "joint_jump"
DEFECT_JITTER = "jitter"
DEFECT_MISSING = "missing_marker"
DEFECT_ROOT_SLIP = "root_slip"
DEFECT_TYPES = (DEFECT_JUMP, DEFECT_JITTER, DEFECT_MISSING, DEFECT_ROOT_SLIP)
JOINT_DEFECT_TYPES = (DEFECT_JUMP, DEFECT_JITTER, DEFECT_MISSING)

# Default detector thresholds are part of the public, versioned parameters.
DEFAULT_THRESHOLDS = {
    "joint_jump_deg": 25.0,
    "joint_jitter_deg": 1.25,
    "invalid_norm_min": 0.5,
    "root_slip_m": 0.045,
}
THRESHOLD_RANGES = {
    "joint_jump_deg": (1.0, 179.0),
    "joint_jitter_deg": (0.01, 90.0),
    "invalid_norm_min": (0.001, 0.999),
    "root_slip_m": (0.0001, 10.0),
}

# Declared interaction budget for candidate previews. These are deliberately
# versioned and included in verification output.
PREVIEW_BUDGET_MS = {1200: 900, 6000: 1800, 12000: 3200}
DEFAULT_PREVIEW_BUDGET_MS = 3200

# Evaluation / consistency tolerances. All are explicit in every report.
EVALUATION_TOLERANCES = {
    "joint_angle_error_deg": 2.0,
    "root_error_m": 0.02,
    "quaternion_component_abs": 1e-9,
    "root_component_abs_m": 1e-9,
}

POLICY_TEXT = {
    "endpoint_rule": (
        "Clean frames are byte-preserved. A defective coordinate is replaced only "
        "between the nearest clean/auto/user anchor on the left and right; anchor "
        "frames themselves are unchanged. At a clip boundary with no anchor on one "
        "side, the sole available endpoint pose is held (constant extension)."
    ),
    "sign_rule": (
        "For every interpolation pair, the right quaternion is multiplied by -1 when "
        "dot(q_left,q_right) < 0 before shortest-path SLERP. Stored repair quaternions "
        "are normalized and rounded to 9 decimals; signs are never independently "
        "canonicalized per output frame because that would create needless jumps."
    ),
    "root_rule": (
        "Root position is interpolated componentwise with piecewise linear position "
        "between the same inclusive-clean anchor endpoints used by rotations."
    ),
    "acceleration_rule": (
        "Angular acceleration at frame t is ||r(t+1)-r(t)|| * FPS^2, where r is the "
        "shortest axis-angle vector of q(t)^-1 q(t+1), reported in deg/s^2."
    ),
    "error_rule": (
        "Quaternion error is 2*acos(min(1,abs(dot(q_repair,q_truth)))) in degrees. "
        "Root error is Euclidean distance in metres. RMS is over detailed repaired "
        "coordinates, peak acceleration is over the affected frame interval plus its "
        "two neighbouring frames."
    ),
    "trust_rule": (
        "A repaired frame is trusted only when every repaired joint angle error is <= "
        "2.0 deg and every repaired root error is <= 0.02 m. Overall status becomes "
        "WITHHELD_UNTRUSTED if any detail crosses either tolerance; details are never "
        "silently discarded. These are evaluation tolerances, not consistency tolerances."
    ),
}

COLOR_JUMP = "#e45756"
COLOR_JITTER = "#f2b134"
COLOR_MISSING = "#8e6cef"
COLOR_ROOT = "#2f9e9b"
COLOR_KEYFRAME = "#2d6cdf"
