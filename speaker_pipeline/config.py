"""Pipeline configuration loading and validation."""

import json

from .errors import InvalidThresholdError

REQUIRED_KEYS = (
    "frame_length_ms",
    "frame_shift_ms",
    "feature_dim",
    "energy_threshold",
    "spectral_flatness_threshold",
    "overlap_similarity_threshold",
    "cluster_distance_threshold",
    "min_speech_ms",
    "min_silence_ms",
    "worker_threads",
    "snapshot_interval",
)


class Config:
    def __init__(self, mapping):
        missing = [key for key in REQUIRED_KEYS if key not in mapping]
        if missing:
            raise InvalidThresholdError(
                "missing config keys: " + ",".join(sorted(missing)),
                stage="config",
            )
        self.data = dict(mapping)

    @staticmethod
    def validate(mapping):
        cfg = dict(mapping)
        if cfg.get("cluster_distance_threshold", 0) <= 0:
            raise InvalidThresholdError(
                "cluster_distance_threshold must be > 0",
                stage="clustering",
            )
        if cfg.get("worker_threads", 0) < 1:
            raise InvalidThresholdError(
                "worker_threads must be >= 1", stage="config"
            )
        if cfg.get("frame_shift_ms", 0) <= 0 or cfg.get(
            "frame_length_ms", 0
        ) <= 0:
            raise InvalidThresholdError(
                "frame length/shift must be positive", stage="config"
            )
        if cfg.get("feature_dim", 0) < 1:
            raise InvalidThresholdError(
                "feature_dim must be >= 1", stage="config"
            )
        return cfg

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def param_subset(self, keys):
        return {key: self.data[key] for key in keys if key in self.data}

    def hash_key(self):
        from .fingerprint import param_fingerprint

        return param_fingerprint(self.data)
