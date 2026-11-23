"""分段与归属配置：加载、默认值、校验与配置指纹。"""

import copy
import os

from .errors import PipelineError, E_STAGE_FAILED
from .utils import canonical_json, sha256_text

DEFAULT_DIMENSIONS = [
    {"name": "f0_median", "unit": "hz", "levels": [4, 6, 8, 10]},
    {"name": "f0_iqr", "unit": "hz", "levels": [3, 4, 5, 6]},
    {"name": "f1_median", "unit": "hz", "levels": [5, 7, 9, 11]},
    {"name": "f2_median", "unit": "hz", "levels": [5, 7, 9, 11]},
    {"name": "f3_median", "unit": "hz", "levels": [5, 7, 9, 11]},
    {"name": "syllable_rate", "unit": "hz", "levels": [3, 4, 5, 6]},
    {"name": "voiced_ratio", "unit": "ratio", "levels": [3, 4, 5, 6]},
    {"name": "energy_cv", "unit": "ratio", "levels": [3, 4, 5, 6]},
]

DEFAULTS = {
    "sample_rate": 16000,
    "frame_ms": 25.0,
    "hop_ms": 10.0,
    "energy_gate_high": 0.35,
    "energy_gate_low": 0.15,
    "min_segment_ms": 120.0,
    "min_gap_ms": 90.0,
    "quantile_method": "linear",
    "dimensions": DEFAULT_DIMENSIONS,
    "similarity_threshold": 0.78,
    "threshold_tie_rule": "split",
    "max_consecutive_segments": 128,
    "first_response_budget_ms": 3000.0,
    "max_workers": 2,
    "snapshot_commit_every": 3,
    "adaptive_level_bins": 4,
}


class Config:
    """不可变风格的配置容器，配置哈希只由业务参数决定。"""

    def __init__(self, data):
        merged = copy.deepcopy(DEFAULTS)
        merged.update(copy.deepcopy(data))
        self._data = merged
        self.validate()
        self.hash = sha256_text(canonical_json(self.fingerprint_payload()))

    @classmethod
    def load(cls, path):
        import json

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data)

    def get(self, key):
        return copy.deepcopy(self._data[key])

    @property
    def data(self):
        return copy.deepcopy(self._data)

    def with_overrides(self, **overrides):
        """返回带覆盖项的新配置（用于编辑命令切换档位/阈值）。"""
        merged = self.data
        merged.update(copy.deepcopy(overrides))
        return Config(merged)

    def dimensions(self):
        return self.get("dimensions")

    def dimension_names(self):
        return [dim["name"] for dim in self._data["dimensions"]]

    def frame_length(self):
        return int(round(self._data["sample_rate"] * self._data["frame_ms"] / 1000.0))

    def hop_length(self):
        return int(round(self._data["sample_rate"] * self._data["hop_ms"] / 1000.0))

    def fingerprint_payload(self):
        """参与配置指纹的完整业务参数。"""
        return {key: self._data[key] for key in sorted(self._data)}

    def stage_fingerprint(self, stage):
        """阶段级配置指纹：阈值变更不影响 feature，位宽变更不影响 segment。"""
        payload = self.fingerprint_payload()
        if stage == "segment":
            keys = (
                "sample_rate", "frame_ms", "hop_ms", "energy_gate_high",
                "energy_gate_low", "min_segment_ms", "min_gap_ms",
            )
        elif stage == "feature":
            keys = (
                "sample_rate", "frame_ms", "hop_ms", "quantile_method",
                "dimensions",
            )
        elif stage == "quantize":
            keys = ("dimensions", "adaptive_level_bins", "quantile_method")
        elif stage == "attach":
            keys = ("similarity_threshold", "threshold_tie_rule", "max_consecutive_segments")
        elif stage == "aggregate":
            keys = ("dimensions",)
        else:
            keys = tuple(sorted(payload))
        return sha256_text(canonical_json({key: payload[key] for key in keys}))

    def validate(self):
        data = self._data
        if data["sample_rate"] != 16000:
            raise PipelineError(E_STAGE_FAILED, "sample_rate 必须为 16000", "config")
        if not 0 < data["frame_ms"]:
            raise PipelineError(E_STAGE_FAILED, "frame_ms 非法", "config")
        if not 0 < data["hop_ms"] <= data["frame_ms"]:
            raise PipelineError(E_STAGE_FAILED, "hop_ms 非法", "config")
        if not 0.0 <= data["energy_gate_low"] <= data["energy_gate_high"] <= 1.0:
            raise PipelineError(E_STAGE_FAILED, "能量双门限需满足 0<=low<=high<=1", "config")
        if data["min_segment_ms"] <= 0 or data["min_gap_ms"] < 0:
            raise PipelineError(E_STAGE_FAILED, "最短段/最短间隙非法", "config")
        if not data["dimensions"]:
            raise PipelineError(E_STAGE_FAILED, "至少需要一个音色维度", "config")
        names = [dim["name"] for dim in data["dimensions"]]
        if len(names) != len(set(names)):
            raise PipelineError(E_STAGE_FAILED, "音色维度名称重复", "config")
        for dim in data["dimensions"]:
            levels = dim.get("levels")
            if not levels or any(int(level) < 1 for level in levels):
                raise PipelineError(E_STAGE_FAILED, "维度量化档位非法: {}".format(dim["name"]), "config")
        if not 0.0 <= data["similarity_threshold"] <= 1.0:
            raise PipelineError(E_STAGE_FAILED, "相似度阈值超出 [0,1]", "config")
        if data["threshold_tie_rule"] not in ("merge", "split"):
            raise PipelineError(E_STAGE_FAILED, "threshold_tie_rule 仅支持 merge/split", "config")
        if int(data["max_workers"]) < 1:
            raise PipelineError(E_STAGE_FAILED, "max_workers 必须 >=1", "config")
        if int(data["max_consecutive_segments"]) < 1:
            raise PipelineError(E_STAGE_FAILED, "max_consecutive_segments 必须 >=1", "config")

    def validate_threshold(self):
        """供阈值越界注入调用：越界返回业务码，正常返回 None。"""
        threshold = self._data["similarity_threshold"]
        if not 0.0 <= threshold <= 1.0:
            return "E_INJECT_THRESHOLD_RANGE"
        return None
