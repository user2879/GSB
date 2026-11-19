"""Deterministic, scale-adaptive voice activity detection toolkit."""

from .audio import AudioFormatError, AudioData, read_wav
from .config import ConfigError, load_config, load_manifest
from .endpoints import build_endpoints
from .features import extract_features
from .pipeline import run_vad
from .policy import estimate_scale_metrics, select_policy
from .state_machine import estimate_noise_floor, run_state_machine

__all__ = [
    "ConfigError",
    "AudioData",
    "AudioFormatError",
    "build_endpoints",
    "estimate_noise_floor",
    "estimate_scale_metrics",
    "extract_features",
    "load_config",
    "load_manifest",
    "read_wav",
    "run_vad",
    "run_state_machine",
    "select_policy",
]

__version__ = "1.0.0"
