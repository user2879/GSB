"""表征配置模型与校验。

校验规则对应业务要求 1：
- 分块尺寸能整除图像边长
- 颜色/梯度特征维度与嵌入维度为正
- 量化档位与定点小数位宽必须落在允许集合内
- 单图分块上限、单批特征点数预算、线程数、提交间隔为正
非法项统一抛出 ``ConfigError``，由引擎映射为 E_CONFIG_INVALID。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

ALLOWED_QUANT_LEVELS = (8, 16, 32)
ALLOWED_FIXED_BITS = (8, 12, 16)
MAX_WORKERS_LIMIT = 32

_REQUIRED_KEYS = (
    "tile_size",
    "color_dim",
    "gradient_dim",
    "embed_dim",
    "quant_level",
    "fixed_bits",
    "similarity_threshold",
    "max_tiles_per_image",
    "worker_count",
    "point_budget_per_batch",
    "snapshot_commit_interval",
)


class ConfigError(ValueError):
    """配置非法异常，携带可审计的字段名与原因。"""

    def __init__(self, field: str, reason: str):
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


def _is_positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class Config:
    tile_size: int
    color_dim: int
    gradient_dim: int
    embed_dim: int
    quant_level: int
    fixed_bits: int
    similarity_threshold: float
    max_tiles_per_image: int
    worker_count: int
    point_budget_per_batch: int
    snapshot_commit_interval: int

    @property
    def feature_dim(self) -> int:
        return self.color_dim + self.gradient_dim

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict, image_size: int | None = None) -> "Config":
        if not isinstance(data, dict):
            raise ConfigError("<root>", "配置必须是 JSON 对象")
        missing = [key for key in _REQUIRED_KEYS if key not in data]
        if missing:
            raise ConfigError(",".join(missing), "缺少必填字段")

        int_fields = (
            "tile_size",
            "color_dim",
            "gradient_dim",
            "embed_dim",
            "max_tiles_per_image",
            "worker_count",
            "point_budget_per_batch",
            "snapshot_commit_interval",
        )
        for field in int_fields:
            if not _is_positive_int(data[field]):
                raise ConfigError(field, "必须为正整数")

        quant_level = data["quant_level"]
        if quant_level not in ALLOWED_QUANT_LEVELS:
            raise ConfigError(
                "quant_level",
                f"必须属于允许集合 {list(ALLOWED_QUANT_LEVELS)}，实际为 {quant_level!r}",
            )
        fixed_bits = data["fixed_bits"]
        if fixed_bits not in ALLOWED_FIXED_BITS:
            raise ConfigError(
                "fixed_bits",
                f"必须属于允许集合 {list(ALLOWED_FIXED_BITS)}，实际为 {fixed_bits!r}",
            )

        threshold = data["similarity_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ConfigError("similarity_threshold", "必须为数值")
        threshold = float(threshold)
        if not -1.0 <= threshold <= 1.0:
            raise ConfigError("similarity_threshold", "必须落在 [-1, 1] 区间")

        config = Config(
            tile_size=int(data["tile_size"]),
            color_dim=int(data["color_dim"]),
            gradient_dim=int(data["gradient_dim"]),
            embed_dim=int(data["embed_dim"]),
            quant_level=int(quant_level),
            fixed_bits=int(fixed_bits),
            similarity_threshold=threshold,
            max_tiles_per_image=int(data["max_tiles_per_image"]),
            worker_count=int(data["worker_count"]),
            point_budget_per_batch=int(data["point_budget_per_batch"]),
            snapshot_commit_interval=int(data["snapshot_commit_interval"]),
        )

        if image_size is not None:
            if config.tile_size > image_size or image_size % config.tile_size != 0:
                raise ConfigError(
                    "tile_size",
                    f"分块尺寸 {config.tile_size} 必须能整除图像边长 {image_size}",
                )
        if config.worker_count > MAX_WORKERS_LIMIT:
            raise ConfigError(
                "worker_count", f"并发数不得超过上限 {MAX_WORKERS_LIMIT}"
            )
        return config

    @staticmethod
    def load(path: str, image_size: int | None = None) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return Config.from_dict(data, image_size=image_size)
