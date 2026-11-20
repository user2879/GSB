"""交互命令模型与配置变更应用（纯函数，产生新的配置对象）。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .config import RerankConfig
from .errors import ConfigInvalidError
from .fixedpoint import parse_exact

COMMAND_TYPES = (
    "set_weight",
    "set_normalization",
    "set_quant",
    "set_rerank_window",
    "set_penalty",
    "add_candidate",
    "delete_candidate",
    "poll",
)


@dataclass(frozen=True)
class Command:
    seq: int
    type: str
    payload: dict[str, Any]


def parse_commands(raw_list: list[dict]) -> list[Command]:
    commands: list[Command] = []
    for seq, raw in enumerate(raw_list):
        if not isinstance(raw, dict) or raw.get("type") not in COMMAND_TYPES:
            raise ConfigInvalidError(
                f"第 {seq} 条交互命令非法: {raw!r}",
                details={"seq": seq},
            )
        commands.append(Command(seq=seq, type=raw["type"], payload=raw.get("payload", {})))
    return commands


def _replace(config: RerankConfig, **changes) -> RerankConfig:
    import dataclasses

    return dataclasses.replace(config, **changes)


def apply_command_to_config(config: RerankConfig, command: Command) -> RerankConfig:
    payload = command.payload
    if command.type == "set_weight":
        index = payload["feature_index"]
        if not isinstance(index, int) or not (0 <= index < config.feature_dim):
            raise ConfigInvalidError(
                f"set_weight 特征下标越界: {index}", details={"feature_index": index}
            )
        value = parse_exact(payload["weight"], field="weight")
        weights = list(config.weights)
        weights[index] = value
        return _replace(config, weights=tuple(weights))

    if command.type == "set_normalization":
        method = payload["normalization"]
        if method not in ("minmax", "zscore"):
            raise ConfigInvalidError(f"归一化方式非法: {method}")
        return _replace(config, normalization=method)

    if command.type == "set_quant":
        frac_bits = payload.get("frac_bits", config.frac_bits)
        levels = payload.get("quant_levels", config.quant_levels)
        from .config import MAX_FRAC_BITS, MIN_FRAC_BITS

        if not isinstance(frac_bits, int) or not (MIN_FRAC_BITS <= frac_bits <= MAX_FRAC_BITS):
            raise ConfigInvalidError(f"frac_bits 越界: {frac_bits}")
        if not isinstance(levels, int) or levels < 2:
            raise ConfigInvalidError(f"quant_levels 非法: {levels}")
        return _replace(config, frac_bits=frac_bits, quant_levels=levels)

    if command.type == "set_rerank_window":
        window = payload["window"]
        if not isinstance(window, int) or window <= 0:
            raise ConfigInvalidError(f"rerank_window 必须为正整数: {window}")
        return _replace(config, rerank_window=window)

    if command.type == "set_penalty":
        penalty = parse_exact(payload["penalty"], field="penalty")
        if not (Fraction(0) <= penalty <= 1):
            raise ConfigInvalidError("diversity_penalty 必须位于 [0,1]")
        return _replace(config, diversity_penalty=penalty)

    return config
