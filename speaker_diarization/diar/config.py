"""分离配置：加载、保存与分阶段配置指纹。"""

from dataclasses import asdict, dataclass

from .util import content_hash


@dataclass
class PipelineConfig:
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    feature_dim: int = 12
    energy_threshold: float = 0.02
    spectral_flatness_threshold: float = 0.6
    cluster_distance_threshold: float = 0.5
    overlap_rule: str = "nearest"          # nearest | max_margin
    min_speech_ms: float = 60.0
    min_silence_ms: float = 120.0
    frac_bits: int = 10                    # 定点小数位宽（量化档位）
    max_workers: int = 4
    snapshot_interval: int = 2             # 快照提交间隔（音频数）
    boundary_tolerance_ms: float = 30.0    # 边界容差
    confidence_tolerance: float = 0.12     # 置信度校准容差

    @staticmethod
    def from_dict(data):
        known = {f for f in PipelineConfig.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"未知配置项: {sorted(unknown)}")
        return PipelineConfig(**{k: data[k] for k in data if k in known})

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def load(path):
        import json
        with open(path, "r", encoding="utf-8") as fh:
            return PipelineConfig.from_dict(json.load(fh))

    def save(self, path):
        import json
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2, sort_keys=True)

    def fingerprint(self):
        return content_hash(self.to_dict())

    def stage_fingerprint(self, stage):
        """分阶段配置指纹：每个阶段只依赖真正影响它的配置项。

        这样修改能量门限只会改变切分及其下游指纹，特征键保持不变。
        """
        groups = {
            "features": [
                "frame_length_ms", "frame_shift_ms", "feature_dim", "frac_bits",
            ],
            "segments": [
                "frame_length_ms", "frame_shift_ms", "feature_dim", "frac_bits",
                "energy_threshold", "spectral_flatness_threshold",
                "min_speech_ms", "min_silence_ms", "cluster_distance_threshold",
            ],
            "clusters": [
                "feature_dim", "frac_bits", "cluster_distance_threshold",
            ],
            "attribution": [
                "feature_dim", "frac_bits", "cluster_distance_threshold",
                "overlap_rule",
            ],
            "verification": [
                "boundary_tolerance_ms", "confidence_tolerance", "frac_bits",
            ],
        }
        full = self.to_dict()
        subset = {key: full[key] for key in groups[stage]}
        return content_hash(subset)
