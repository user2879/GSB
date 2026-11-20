"""Shared constants and stage ordering."""

SCHEMA_VERSION = "aed-workbench/1.0"
PACKET_SCHEMA_VERSION = "aed-packet/1"
CONFIG_SCHEMA_VERSION = "aed-global-config/1"

CLASSES = ("speech", "music", "noise", "alarm")

STAGES = ("parse", "map", "align", "calibrate", "decide", "drift")
STAGE_ORDER = {name: index for index, name in enumerate(STAGES)}

ERROR_CODES = {
    "packet_invalid": "E_PACKET_INVALID",
    "class_unmapped": "E_CLASS_UNMAPPED",
    "calib_residual": "E_CALIB_RESIDUAL",
    "calib_baseline": "E_CALIB_BASELINE",
    "snapshot_stale": "E_SNAPSHOT_STALE",
    "config_invalid": "E_CONFIG_INVALID",
}

SCORE_TYPES = ("logit", "probability")
