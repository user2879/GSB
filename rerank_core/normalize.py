"""逐请求特征归一化与定点量化（全程 Fraction / int，无 float 参与运算）。"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .config import RerankConfig
from .fixedpoint import (
    isqrt_frac,
    mean,
    parse_exact,
    population_variance,
    round_half_away,
)
from .models import Candidate


@dataclass
class NormalizationStats:
    request_id: str
    method: str
    feature_index: int
    minimum: str
    maximum: str
    mean: str | None
    std: str | None
    constant: bool

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "method": self.method,
            "feature_index": self.feature_index,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "constant": self.constant,
        }


def feature_vector(candidate: Candidate) -> list[Fraction]:
    """按 召回 -> 稠密 -> 稀疏 的固定顺序构造统一精确特征向量。"""
    vector = [parse_exact(score, field=f"recall:{ch}")
              for ch, score in candidate.recall_scores.items()]
    vector += [parse_exact(value, field="dense") for value in candidate.dense]
    vector += [parse_exact(value, field="sparse") for value in candidate.sparse]
    return vector


def raw_matrix(candidates: list[Candidate]) -> list[list[Fraction]]:
    return [feature_vector(candidate) for candidate in candidates]


def compute_stats(
    request_id: str, matrix: list[list[Fraction]], feature_dim: int, method: str
) -> list[NormalizationStats]:
    stats: list[NormalizationStats] = []
    for index in range(feature_dim):
        column = [row[index] for row in matrix]
        minimum = min(column)
        maximum = max(column)
        constant = maximum == minimum
        if method == "minmax":
            stats.append(
                NormalizationStats(
                    request_id, method, index,
                    f"{minimum.numerator}/{minimum.denominator}",
                    f"{maximum.numerator}/{maximum.denominator}",
                    None, None, constant,
                )
            )
        else:
            mu = mean(column)
            variance = population_variance(column, mu)
            std = isqrt_frac(variance)
            stats.append(
                NormalizationStats(
                    request_id, method, index,
                    f"{minimum.numerator}/{minimum.denominator}",
                    f"{maximum.numerator}/{maximum.denominator}",
                    f"{mu.numerator}/{mu.denominator}",
                    f"{std.numerator}/{std.denominator}",
                    constant or std == 0,
                )
            )
    return stats


def normalize_value(value: Fraction, stat: NormalizationStats, method: str) -> Fraction:
    if stat.constant:
        return Fraction(0)
    if method == "minmax":
        lo = Fraction(stat.minimum)
        hi = Fraction(stat.maximum)
        return (value - lo) / (hi - lo)
    mu = Fraction(stat.mean)
    std = Fraction(stat.std)
    return (value - mu) / std


def quantize_value(normalized: Fraction, frac_bits: int, quant_levels: int) -> int:
    """按小数位宽与量化档位量化：scale=10^k，档位限制归一化值的整数部分上限。"""
    scale = 10**frac_bits
    max_level = quant_levels // 2
    capped = max(Fraction(-max_level), min(Fraction(max_level), normalized))
    return round_half_away(capped * scale)


def quantize_request(
    request_id: str,
    candidates: list[Candidate],
    stats: list[NormalizationStats],
    config: RerankConfig,
) -> list[list[int]]:
    method = config.normalization
    matrix = raw_matrix(candidates)
    quantized: list[list[int]] = []
    for row in matrix:
        fixed_row: list[int] = []
        for index, value in enumerate(row):
            normalized = normalize_value(value, stats[index], method)
            fixed_row.append(quantize_value(normalized, config.frac_bits, config.quant_levels))
        quantized.append(fixed_row)
    return quantized
