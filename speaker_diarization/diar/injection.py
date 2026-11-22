"""异常注入注册表：在真实输入边界施加故障并映射业务码。"""

import dataclasses

import numpy as np

from .codes import E_RANGE_VIOLATION, E_THRESHOLD_ZERO, PipelineError


class InjectionRegistry:
    def __init__(self, injections):
        self._items = list(injections)
        self.by_audio_stage = {}
        for item in self._items:
            self.by_audio_stage[(item["audio"], item["stage"])] = item

    @property
    def items(self):
        return list(self._items)

    def get(self, audio, stage):
        return self.by_audio_stage.get((audio, stage))

    def prepare_audio_bytes(self, audio, raw):
        item = self.get(audio, "audio")
        if item and item["fault_type"] == "wav_truncated":
            keep = max(12, len(raw) - int(len(raw) * 0.25))
            return raw[:keep], item
        return raw, None

    def prepare_samples(self, audio, samples):
        item = self.get(audio, "features")
        if item and item["fault_type"] == "nonfinite_samples":
            damaged = np.array(samples, dtype=np.float64)
            damaged[:: max(1, damaged.shape[0] // 16)] = np.nan
            damaged[1:: max(1, damaged.shape[0] // 16)] = np.inf
            return damaged, item
        return samples, None

    def effective_config(self, audio, config):
        item = self.get(audio, "clusters")
        if item and item["fault_type"] == "threshold_zero":
            cfg = dataclasses.replace(config, cluster_distance_threshold=0.0)
            return cfg, item
        return config, None

    def check_segment_range(self, audio, n_frames):
        item = self.get(audio, "segments")
        if item and item["fault_type"] == "range_violation":
            start, end = item["replacement_interval_frames"]
            if start < 0 or end > n_frames or start >= end:
                raise PipelineError(
                    E_RANGE_VIOLATION, "segments",
                    f"替换区间帧 [{start},{end}) 超出 [0,{n_frames})",
                    injection_id=item["injection_id"],
                )

    def effective_truth(self, audio, truth_entry):
        item = self.get(audio, "verification")
        if item and item["fault_type"] == "truth_mismatch":
            patched = dict(truth_entry)
            patched["expected_speakers"] = item["forced_expected_speakers"]
            return patched, item
        return truth_entry, None

    @staticmethod
    def validate_threshold(cfg, audio, injection_id=None):
        if cfg.cluster_distance_threshold <= 0.0:
            raise PipelineError(
                E_THRESHOLD_ZERO, "clusters",
                "聚类距离阈值必须为正数，当前为 0",
                injection_id=injection_id,
            )
