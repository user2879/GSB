"""增量配置加载与校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError


@dataclass(frozen=True)
class ComponentConfig:
    """增量配置。

    workers: 线程池最大并发数（正整数）。
    dict_change_mode: entry=单条增删/词频调整；batch=批量编辑一次失效。
    snapshot_commit_interval: 每处理多少句提交一次版本化快照（正整数）。
    low_confidence_threshold: 边界置信度低于该值计入低置信度清单。
    smoothing: 缺失发射（及缺失起始/转移）使用的自然对数平滑值。
    """

    workers: int = 1
    dict_change_mode: str = "entry"
    snapshot_commit_interval: int = 5
    low_confidence_threshold: float = 0.4
    smoothing: float = -20.0

    @classmethod
    def load(cls, path: str | Path) -> "ComponentConfig":
        raw_path = Path(path)
        try:
            data = json.loads(raw_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"配置文件不存在: {raw_path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"配置文件不是合法 JSON: {raw_path}") from exc
        if not isinstance(data, dict):
            raise ConfigError("增量配置 JSON 顶层必须是对象")
        return cls.from_dict(data)

    @classmethod
    @classmethod
    def from_dict(cls, data: dict) -> "ComponentConfig":
        workers = data.get("workers", 1)
        if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
            raise ConfigError("并发数 workers 必须为正整数", code="E_CONFIG_INVALID")
        mode = data.get("dict_change_mode", "entry")
        if mode not in ("entry", "batch"):
            raise ConfigError("词典变更模式 dict_change_mode 仅支持 entry/batch")
        interval = data.get("snapshot_commit_interval", 5)
        if (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval <= 0
        ):
            raise ConfigError("快照提交间隔 snapshot_commit_interval 必须为正整数")
        threshold = data.get("low_confidence_threshold", 0.4)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ConfigError("低置信度阈值必须是数值")
        smoothing = data.get("smoothing", -20.0)
        if not isinstance(smoothing, (int, float)) or isinstance(smoothing, bool):
            raise ConfigError("平滑值必须是数值")
        if smoothing >= 0:
            raise ConfigError("平滑值必须为自然对数域的负值")
        return cls(
            workers=int(workers),
            dict_change_mode=mode,
            snapshot_commit_interval=int(interval),
            low_confidence_threshold=float(threshold),
            smoothing=float(smoothing),
        )
