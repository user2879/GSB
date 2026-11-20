"""领域模型：指令种子、槽位来源、构造配置、打分参数、累计工作区状态。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import fingerprint


@dataclass(frozen=True)
class Seed:
    seed_id: str
    task_type: str
    template_id: str
    template: str
    slots: tuple[str, ...]
    answer_template: str
    line_no: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "task_type": self.task_type,
            "template_id": self.template_id,
            "template": self.template,
            "slots": list(self.slots),
            "answer_template": self.answer_template,
            "line_no": self.line_no,
        }


@dataclass(frozen=True)
class SlotSource:
    source_id: str
    task_type: str
    slot_name: str
    values: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "task_type": self.task_type,
            "slot_name": self.slot_name,
            "values": list(self.values),
        }


@dataclass(frozen=True)
class BuildConfig:
    task_types: tuple[str, ...]
    target_counts: dict[str, int]
    template_weights: dict[str, dict[str, float]]
    slot_weights: dict[str, dict[str, float]]
    dedup_threshold: float
    boundary_rule: str  # "merge" 表示 ==阈值 视为近重复；"split" 表示不合并
    max_workers: int
    snapshot_interval_batches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_types": list(self.task_types),
            "target_counts": dict(self.target_counts),
            "template_weights": {
                task: dict(weights) for task, weights in self.template_weights.items()
            },
            "slot_weights": {
                task: dict(weights) for task, weights in self.slot_weights.items()
            },
            "dedup_threshold": float(self.dedup_threshold),
            "boundary_rule": self.boundary_rule,
            "max_workers": int(self.max_workers),
            "snapshot_interval_batches": int(self.snapshot_interval_batches),
        }


@dataclass(frozen=True)
class DimensionSpec:
    weight: float
    baseline: float

    def to_dict(self) -> dict[str, Any]:
        return {"weight": float(self.weight), "baseline": float(self.baseline)}


@dataclass(frozen=True)
class ScoreParams:
    dimensions: tuple[str, ...]
    specs: dict[str, DimensionSpec]
    length_lower: int
    length_upper: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": list(self.dimensions),
            "specs": {name: spec.to_dict() for name, spec in self.specs.items()},
            "length_lower": int(self.length_lower),
            "length_upper": int(self.length_upper),
        }


@dataclass(frozen=True)
class Vocab:
    tokens: tuple[str, ...]
    rules: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": list(self.tokens), "rules": dict(self.rules)}


@dataclass
class WorkspaceState:
    """加载 + 逐批次应用后的累计输入状态。"""

    seeds: dict[str, Seed]
    slot_sources: dict[str, SlotSource]
    config: BuildConfig
    score_params: ScoreParams
    vocab: Vocab

    def fingerprints(self) -> dict[str, str]:
        seed_payload = [self.seeds[key].to_dict() for key in sorted(self.seeds)]
        slot_payload = [
            self.slot_sources[key].to_dict() for key in sorted(self.slot_sources)
        ]
        return {
            "seeds_fp": fingerprint(seed_payload),
            "slots_fp": fingerprint(slot_payload),
            "config_fp": fingerprint(self.config.to_dict()),
            "params_fp": fingerprint(self.score_params.to_dict()),
            "vocab_fp": fingerprint(self.vocab.to_dict()),
        }

    def combined_fingerprint(self) -> str:
        return fingerprint(self.fingerprints())

    def seeds_by_task(self, task_type: str) -> list[Seed]:
        items = [s for s in self.seeds.values() if s.task_type == task_type]
        return sorted(items, key=lambda s: s.template_id)

    def sources_by_task_slot(self, task_type: str, slot_name: str) -> list[SlotSource]:
        items = [
            source
            for source in self.slot_sources.values()
            if source.task_type == task_type and source.slot_name == slot_name
        ]
        return sorted(items, key=lambda source: source.source_id)
