"""配置加载与严格校验：整除关系、正整数预算、量化档位枚举、并发上限。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import E_CONFIG_INVALID, PipelineValidationError
from .fixedpoint import QUANT_LEVELS, validate_quant
from .hashing import digest
from .sample_gen import PAGE_H, PAGE_W


def _positive_int(payload: dict, key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PipelineValidationError(
            E_CONFIG_INVALID, "<config>", key, f"{key} 必须为正整数，实际为 {value!r}")
    return value


@dataclass(frozen=True)
class Config:
    window_size: int
    window_stride: int
    scan_row_budget: int
    compare_point_budget: int
    adjudication_budget: int
    frac_bits: int
    quant_level: str
    max_workers: int
    min_component_area: int
    max_aspect_ratio: int
    candidate_score_threshold: int
    template_variants: int

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, payload: dict) -> "Config":
        window_size = _positive_int(payload, "window_size")
        stride = _positive_int(payload, "window_stride")
        frac_bits = _positive_int(payload, "frac_bits")
        quant_level = payload.get("quant_level")
        validate_quant(frac_bits, quant_level)

        if PAGE_W % window_size != 0 or PAGE_H % window_size != 0:
            raise PipelineValidationError(
                E_CONFIG_INVALID, "<config>", "window_size",
                f"滑窗尺寸 {window_size} 必须整除版面边长 {PAGE_W}x{PAGE_H}")
        if stride > window_size:
            raise PipelineValidationError(
                E_CONFIG_INVALID, "<config>", "window_stride",
                f"步长 {stride} 不得大于窗口尺寸 {window_size}")
        if (PAGE_W - window_size) % stride != 0 or (PAGE_H - window_size) % stride != 0:
            raise PipelineValidationError(
                E_CONFIG_INVALID, "<config>", "window_stride",
                f"步长 {stride} 与窗口 {window_size} 在 {PAGE_W}x{PAGE_H} 上不能完整切窗")

        cfg = cls(
            window_size=window_size,
            window_stride=stride,
            scan_row_budget=_positive_int(payload, "scan_row_budget"),
            compare_point_budget=_positive_int(payload, "compare_point_budget"),
            adjudication_budget=_positive_int(payload, "adjudication_budget"),
            frac_bits=frac_bits,
            quant_level=quant_level,
            max_workers=_positive_int(payload, "max_workers"),
            min_component_area=_positive_int(payload, "min_component_area"),
            max_aspect_ratio=_positive_int(payload, "max_aspect_ratio"),
            candidate_score_threshold=_positive_int(payload, "candidate_score_threshold"),
            template_variants=_positive_int(payload, "template_variants"),
        )
        if cfg.max_workers > 8:
            raise PipelineValidationError(
                E_CONFIG_INVALID, "<config>", "max_workers",
                "max_workers 不得超过硬上限 8")
        if cfg.template_variants > 5:
            raise PipelineValidationError(
                E_CONFIG_INVALID, "<config>", "template_variants",
                "template_variants 不得超过内置旋转变体数 5")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return digest(self.to_dict())

    def param_fingerprint(self) -> str:
        return digest({
            "window_size": self.window_size,
            "window_stride": self.window_stride,
            "min_component_area": self.min_component_area,
            "max_aspect_ratio": self.max_aspect_ratio,
            "candidate_score_threshold": self.candidate_score_threshold,
        })

    def budget_fingerprint(self) -> str:
        return digest({
            "scan_row_budget": self.scan_row_budget,
            "compare_point_budget": self.compare_point_budget,
            "adjudication_budget": self.adjudication_budget,
        })


DEFAULT_CONFIG: dict = {
    "window_size": 32,
    "window_stride": 16,
    "scan_row_budget": 60000,
    "compare_point_budget": 3000000,
    "adjudication_budget": 12,
    "frac_bits": 4,
    "quant_level": "standard",
    "max_workers": 4,
    "min_component_area": 12,
    "max_aspect_ratio": 6,
    "candidate_score_threshold": 8,
    "template_variants": 3,
}


def write_default_config(path: str | Path, overrides: dict | None = None) -> Path:
    path = Path(path)
    payload = dict(DEFAULT_CONFIG)
    if overrides:
        payload.update(overrides)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
