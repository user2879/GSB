"""本地命令行配置：解析、严格校验与确定性摘要。

配置决定了初始参数、各工作进程梯度分片以及全部量化行为。
任何不合法字段都以 E_CONFIG 整体拒绝（不接受部分生效）。
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .digest import token_digest
from .errors import E_CONFIG, GradCompError

TRUNCATIONS = ("floor", "nearest", "toward_zero")
TRANSPORTS = ("tcp", "loopback")
SESSION_VERSION = 1
# 允许在协议/快照格式升级时做整体替换恢复的最低可识别版本。
MIN_SNAPSHOT_VERSION = 1

_CONFIG_FIELDS = {
    "workers", "dimension", "seed", "samples", "scale_exponent", "truncation",
    "learning_rate", "block_size", "base_port", "transport", "compressed",
    "steps",
}


def _fail(message: str, field_name: str, details: dict[str, Any] | None = None):
    raise GradCompError(E_CONFIG, message, field=field_name, details=details)


@dataclass(frozen=True)
class Config:
    workers: int = 4
    dimension: int = 64
    seed: int = 20261120
    samples: int = 128
    scale_exponent: int = 2          # 缩放因子 s = 10 ** scale_exponent
    truncation: str = "floor"        # floor / nearest / toward_zero
    learning_rate: float = 0.1
    block_size: int = 16             # 每次沿环转发的元素数（块大小可配置）
    base_port: int = 39000          # rank r 监听 base_port + r
    transport: str = "tcp"          # tcp（真实回环套接字）/ loopback（进程内）
    compressed: bool = True         # False 时执行未压缩浮点训练步
    steps: int = 1

    @property
    def scale(self) -> float:
        return float(10 ** self.scale_exponent)

    @property
    def chunks(self) -> int:
        return self.workers

    @property
    def chunk_size(self) -> int:
        if self.dimension % self.workers != 0:
            _fail("dimension 必须能被 workers 整除", "dimension",
                  {"dimension": self.dimension, "workers": self.workers})
        return self.dimension // self.workers

    @property
    def ports(self) -> list[int]:
        return [self.base_port + rank for rank in range(self.workers)]

    def digest(self) -> str:
        return token_digest(
            [f"{key}={value}" for key, value in sorted(asdict(self).items())]
        )

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data["scale"] = self.scale
        data["ports"] = self.ports
        data["config_digest"] = self.digest()
        data["session_version"] = SESSION_VERSION
        return data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate(raw: dict[str, Any]) -> Config:
    """严格校验原始 dict 并构造 Config；任何问题都整体拒绝。"""
    if not isinstance(raw, dict):
        _fail("配置必须是 JSON 对象", "<root>")
    unknown = sorted(set(raw) - _CONFIG_FIELDS)
    if unknown:
        _fail(f"存在未知配置字段: {', '.join(unknown)}", unknown[0],
              {"unknown_fields": unknown})

    values = dict(raw)

    for name in ("workers", "dimension", "samples", "block_size", "steps"):
        if name in values and not _is_int(values[name]):
            _fail(f"{name} 必须是整数", name)
    if "seed" in values and (not _is_int(values["seed"]) or values["seed"] < 0):
        _fail("seed 必须是非负整数", "seed")
    if "scale_exponent" in values and not _is_int(values["scale_exponent"]):
        _fail("scale_exponent 必须是整数（缩放因子为 10 的整数幂）",
              "scale_exponent")
    if "base_port" in values and not _is_int(values["base_port"]):
        _fail("base_port 必须是整数", "base_port")

    cfg = Config(**values)

    if cfg.workers < 2 or cfg.workers > 64:
        _fail("workers 取值范围为 2..64（环形拓扑至少需要两个进程）", "workers")
    if cfg.dimension < cfg.workers or cfg.dimension > 1_000_000:
        _fail("dimension 必须不小于 workers 且不超过 1000000", "dimension")
    if cfg.dimension % cfg.workers != 0:
        _fail("dimension 必须能被 workers 整除以形成等长环形分片", "dimension",
              {"dimension": cfg.dimension, "workers": cfg.workers})
    if cfg.samples <= 0:
        _fail("samples 必须为正整数", "samples")
    if cfg.scale_exponent < -6 or cfg.scale_exponent > 9:
        _fail("scale_exponent 取值范围为 -6..9", "scale_exponent")
    if cfg.truncation not in TRUNCATIONS:
        _fail(f"truncation 必须是 {TRUNCATIONS} 之一", "truncation")
    if not isinstance(cfg.learning_rate, (int, float)) or cfg.learning_rate <= 0:
        _fail("learning_rate 必须为正数", "learning_rate")
    if cfg.block_size <= 0:
        _fail("block_size 必须为正整数", "block_size")
    if not (1024 <= cfg.base_port <= 65535):
        _fail("base_port 必须位于 1024..65535（高位端口段）", "base_port")
    if cfg.base_port + cfg.workers - 1 > 65535:
        _fail("base_port + workers 超出端口上限", "base_port")
    if cfg.transport not in TRANSPORTS:
        _fail(f"transport 必须是 {TRANSPORTS} 之一", "transport")
    if cfg.steps <= 0:
        _fail("steps 必须为正整数", "steps")
    return cfg


def load_config(path: str | Path) -> Config:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _fail(f"配置文件不存在: {p}", "config_path")
    except json.JSONDecodeError as exc:
        _fail(f"配置文件不是合法 JSON: {exc.msg}", "config_path",
              {"line": exc.lineno, "column": exc.colno})
    return validate(raw)


def dumps(cfg: Config) -> str:
    return json.dumps(cfg.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
