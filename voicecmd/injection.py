"""异常注入：独立注入配置，对真实输入路径施加故障并采集命中证据。"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, field_validator

from .errors import (
    BusinessError,
    E_AUDIO_FORMAT,
    E_DIMENSION_MISMATCH,
    E_QUANT_LEVEL_INVALID,
    E_SAMPLE_NON_FINITE,
    E_SPAN_OUT_OF_RANGE,
)

VALID_TYPES = frozenset(
    {
        "truncate_wav",
        "non_finite_sample",
        "span_out_of_range",
        "bad_quant_level",
        "posterior_dimension_mismatch",
    }
)


class InjectionEntry(BaseModel):
    id: str
    type: str
    fragment_seq: int | None = None
    window_index: int | None = None
    stage: str
    enabled: bool = True

    @field_validator("type")
    @classmethod
    def _type_known(cls, value):
        if value not in VALID_TYPES:
            raise ValueError(f"未知注入类型 {value}，允许 {sorted(VALID_TYPES)}")
        return value


class InjectionConfig(BaseModel):
    entries: list[InjectionEntry]


@dataclass
class InjectionReport:
    entries: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def record_hit(self, injection_id: str, evidence: str) -> None:
        self.hits.append({"injection_id": injection_id, "evidence": evidence})

    def record_failure(self, error: BusinessError) -> None:
        self.failures.append(error.to_record())

    def as_dict(self) -> dict:
        return {"entries": self.entries, "hits": self.hits, "failures": self.failures}


def load_injection_config(path: str | Path) -> InjectionConfig:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return InjectionConfig.model_validate(payload)


class InjectionEngine:
    def __init__(self, config: InjectionConfig | None):
        self.config = config
        self.report = InjectionReport(
            entries=[entry.model_dump() for entry in config.entries] if config else []
        )
        self._map = {}
        if config:
            for entry in config.entries:
                if entry.enabled:
                    self._map[entry.id] = entry

    def _find(self, injection_type: str, *, fragment_seq=None, window_index=None):
        for entry in self._map.values():
            if entry.type != injection_type:
                continue
            if fragment_seq is not None and entry.fragment_seq != fragment_seq:
                continue
            if window_index is not None and entry.window_index != window_index:
                continue
            return entry
        return None

    def mutate_wav_bytes(self, fragment_seq: int, raw: bytes) -> bytes:
        entry = self._find("truncate_wav", fragment_seq=fragment_seq)
        if entry is None:
            return raw
        cut = max(0, len(raw) - 256)
        self.report.record_hit(
            entry.id,
            f"片段 {fragment_seq} WAV data 截断 {len(raw)}->{cut} 字节",
        )
        return raw[:cut]

    def wants_non_finite(self, fragment_seq: int) -> str | None:
        entry = self._find("non_finite_sample", fragment_seq=fragment_seq)
        if entry is None:
            return None
        self.report.record_hit(
            entry.id, f"片段 {fragment_seq} 采样缓冲注入 NaN（非有限值）"
        )
        return entry.id

    def mutate_span(self, window_index: int, span, total_samples):
        entry = self._find("span_out_of_range", window_index=window_index)
        if entry is None:
            return span
        mutated = (span[0], total_samples + 1)
        self.report.record_hit(
            entry.id,
            f"窗 {window_index} 区间越界 {span} -> {mutated}，总采样 {total_samples}",
        )
        raise BusinessError(
            E_SPAN_OUT_OF_RANGE,
            f"注入 {entry.id}: 窗引用采样区间越界",
            stage="window_feature",
            window_index=window_index,
            injection_id=entry.id,
            evidence=f"span {span} -> {mutated}, total={total_samples}",
        )

    def mutate_quant_level(self, quant_level: int, allowed_levels, *, fragment_seq=None,
                           window_index=None):
        entry = self._find(
            "bad_quant_level",
            fragment_seq=fragment_seq,
            window_index=window_index,
        )
        if entry is None:
            return quant_level
        mutated = max(allowed_levels) + 1
        self.report.record_hit(
            entry.id, f"量化档位被改写 {quant_level} -> {mutated}（不在允许枚举内）"
        )
        raise BusinessError(
            E_QUANT_LEVEL_INVALID,
            f"注入 {entry.id}: 量化档位 {mutated} 不在允许枚举 {allowed_levels} 内",
            stage="quantize",
            fragment_seq=fragment_seq,
            window_index=window_index,
            injection_id=entry.id,
            evidence=f"quant_level {quant_level} -> {mutated}",
        )

    def mutate_dimension(self, window_index: int, expected_cols: int):
        entry = self._find(
            "posterior_dimension_mismatch", window_index=window_index
        )
        if entry is None:
            return expected_cols
        mutated = expected_cols + 1
        self.report.record_hit(
            entry.id,
            f"窗 {window_index} 后验输入维度被改写 {expected_cols} -> {mutated}",
        )
        raise BusinessError(
            E_DIMENSION_MISMATCH,
            f"注入 {entry.id}: 后验输入维度 {mutated} 与投影列数 {expected_cols} 不符",
            stage="classify",
            window_index=window_index,
            injection_id=entry.id,
            evidence=f"dim {expected_cols} -> {mutated}",
        )

    def record_failure(self, error: BusinessError) -> None:
        self.report.record_failure(error)
