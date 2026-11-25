"""全链路配置模型（pydantic 严格校验）与配置/词表指纹。"""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .fingerprint import content_fingerprint

ALLOWED_QUANT_LEVELS = (8, 16, 24, 32)
# 特征维顺序固定：n_mels 个梅尔能量 + 过零率 + 谱质心
ZCR_DIM = 1
CENTROID_DIM = 1


class PathsConfig(BaseModel):
    data_dir: str = "data/wav"
    manifest_file: str = "data/fragment_manifest.json"
    command_stream_file: str = "data/command_stream.json"
    injection_file: str = "configs/injection.json"
    snapshot_dir: str = "data/snapshot"
    report_dir: str = "artifacts"


class SampleConfig(BaseModel):
    fragment_count: int = Field(gt=0)
    samples_per_fragment: int = Field(gt=0)
    speech_ratio: float = Field(gt=0.0, lt=1.0)
    noise_seed: int
    sample_rate: int = 16000
    tone_base_hz: float = 220.0
    tone_step_hz: float = 40.0
    tone_amplitude: float = 4000.0
    noise_std: float = 300.0


class VadConfig(BaseModel):
    frame_length: int = Field(gt=1)
    frame_shift: int = Field(gt=0)
    energy_high: float = Field(gt=0.0)
    energy_low: float = Field(gt=0.0)
    flatness_high: float = Field(gt=0.0, le=1.0)
    flatness_low: float = Field(gt=0.0, le=1.0)
    min_speech_frames: int = Field(ge=1)
    min_silence_frames: int = Field(ge=1)

    @field_validator("energy_low")
    @classmethod
    def _threshold_order(cls, value, info):
        high = info.data.get("energy_high")
        if high is not None and value >= high:
            raise ValueError("energy_low 必须小于 energy_high")
        return value


class WindowConfig(BaseModel):
    window_length: int = Field(gt=1)
    window_shift: int = Field(gt=0)
    drop_partial: bool = True


class MelConfig(BaseModel):
    n_mels: int = Field(gt=0)
    n_fft: int = Field(ge=4)
    low_hz: float = Field(ge=0.0)
    high_hz: float = Field(gt=0.0)


class QuantConfig(BaseModel):
    quant_level: int = Field(description="量化档位（位宽），必须落在允许枚举内")
    allowed_levels: tuple[int, ...] = ALLOWED_QUANT_LEVELS
    float_scale: int = Field(gt=0)
    mel_full_scale: float = Field(gt=0.0)
    zcr_full_scale: float = Field(gt=0.0)
    centroid_full_scale: float = Field(gt=0.0)

    @field_validator("quant_level")
    @classmethod
    def _level_allowed(cls, value, info):
        allowed = info.data.get("allowed_levels", ALLOWED_QUANT_LEVELS)
        if value not in allowed:
            raise ValueError(f"quant_level={value} 不在允许枚举 {allowed} 内")
        return value


class ClassifierConfig(BaseModel):
    weight_seed: int
    boundary_margin: float = Field(ge=0.0)
    softmax_temperature: float = Field(gt=0.0)


class RuntimeConfig(BaseModel):
    max_workers: int = Field(ge=1)
    default_budget: int = Field(ge=0)


class CommandDef(BaseModel):
    code: int = Field(ge=0)
    word: str


class PipelineConfig(BaseModel):
    version: int = 1
    paths: PathsConfig = PathsConfig()
    sample: SampleConfig
    vad: VadConfig
    window: WindowConfig
    mel: MelConfig
    quant: QuantConfig
    classifier: ClassifierConfig
    runtime: RuntimeConfig
    commands: list[CommandDef]

    @field_validator("commands")
    @classmethod
    def _unique_commands(cls, value):
        codes = [item.code for item in value]
        words = [item.word for item in value]
        if len(set(codes)) != len(codes):
            raise ValueError("命令编码必须唯一")
        if len(set(words)) != len(words):
            raise ValueError("命令词必须唯一")
        return value

    @property
    def feature_dim(self) -> int:
        return self.mel.n_mels + ZCR_DIM + CENTROID_DIM

    @property
    def vocab(self) -> dict[int, str]:
        return {item.code: item.word for item in sorted(self.commands, key=lambda c: c.code)}

    def config_fingerprint(self) -> str:
        data = self.model_dump(exclude={"paths"})
        return content_fingerprint(data)

    def vocab_fingerprint(self) -> str:
        return content_fingerprint(
            sorted((item.code, item.word) for item in self.commands)
        )


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return PipelineConfig.model_validate(payload)
