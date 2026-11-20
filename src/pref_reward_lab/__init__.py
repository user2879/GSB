"""Deterministic preference-reward interoperability lab."""

__version__ = "0.1.0"

from .config import LabConfig, StrategyTier
from .engine import ExperimentLab
from .conversion import convert_v1_to_v2, convert_v2_to_v1
from .formats import (
    V1_PAIR_ID_WIDTH,
    V1_INT_WIDTH,
    V1_LINE_WIDTH,
    encode_v1_record,
    parse_raw_pairs,
    parse_v1_file,
    parse_v1_text,
    parse_v2_file,
    parse_v2_text,
    write_v1_file,
    write_v2_file,
)
from .models import FeatureResult, RawPair, ScoreResult, V1Record, V2Record
from .sample import generate_dataset
from .scoring import score_features
from .strategy import select_strategy

__all__ = [
    "ExperimentLab",
    "LabConfig",
    "StrategyTier",
    "V1_PAIR_ID_WIDTH",
    "V1_INT_WIDTH",
    "V1_LINE_WIDTH",
    "FeatureResult",
    "RawPair",
    "ScoreResult",
    "V1Record",
    "V2Record",
    "convert_v1_to_v2",
    "convert_v2_to_v1",
    "encode_v1_record",
    "generate_dataset",
    "parse_raw_pairs",
    "parse_v1_file",
    "parse_v1_text",
    "parse_v2_file",
    "parse_v2_text",
    "score_features",
    "select_strategy",
    "write_v1_file",
    "write_v2_file",
]
