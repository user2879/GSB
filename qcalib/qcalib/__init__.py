"""逐组量化标定与饱和截断领域组件。"""
from .config import CalibrationConfig, FeatureTable, GroupConfig, load_calibration_config, load_feature_table
from .exchange import (
    CalibEntry,
    downgrade_v2_to_v1,
    read_sidecar,
    read_v1,
    read_v2,
    upgrade_v1_to_v2,
    write_sidecar,
    write_v1,
    write_v2,
)
from .pipeline import Pipeline
from .runner import CalibrationRunner

__all__ = [
    "CalibrationConfig",
    "FeatureTable",
    "GroupConfig",
    "load_feature_table",
    "load_calibration_config",
    "CalibEntry",
    "read_v1",
    "write_v1",
    "read_v2",
    "write_v2",
    "upgrade_v1_to_v2",
    "downgrade_v2_to_v1",
    "read_sidecar",
    "write_sidecar",
    "Pipeline",
    "CalibrationRunner",
]
