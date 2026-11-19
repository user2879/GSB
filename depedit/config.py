"""运行配置：定宽字段、编号起点、并发与注入配置。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import HEAD_FILLER

# v1 定宽字段（按字符宽度切分，左对齐右侧补空格）。
DEFAULT_FIXED_WIDTHS: dict[str, int] = {
    "sent_id": 10,
    "word_idx": 4,
    "form": 12,
    "pos": 6,
    "head": 4,
    "relation": 16,
    "score": 6,
}
FIXED_ORDER = ["sent_id", "word_idx", "form", "pos", "head", "relation", "score"]

# 注入阶段 -> 业务码
INJECTION_BUSINESS_CODES: dict[str, str] = {
    "fixed_width_missing": "E_FIXED_WIDTH_FIELD",
    "head_out_of_range": "E_HEAD_OUT_OF_RANGE",
    "score_over_1000": "E_SCORE_RANGE",
    "relation_not_in_map": "E_RELATION_UNKNOWN",
    "v2_non_finite_score": "E_SCORE_NON_FINITE",
    "cycle_dependency": "E_CYCLE_DETECTED",
}
VALID_STAGES = {"parse", "edit", "convert"}


@dataclass
class AppConfig:
    fixed_widths: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_FIXED_WIDTHS))
    head_filler: str = HEAD_FILLER
    # v2 浮点偏移的编号起点：0 表示 0 起始，1 表示 1 起始。
    v2_head_origin: int = 0
    # 统一表示固定为 1 起始，根用 0。
    unified_origin: int = 1
    max_workers: int = 4
    workers_hard_limit: int = 16
    score_decimals: int = 4  # v1 千分比换算后保留精度标签位数

    @staticmethod
    def load(path: str | Path | None) -> "AppConfig":
        cfg = AppConfig()
        if path is None:
            return cfg
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.max_workers = max(1, min(int(cfg.max_workers), int(cfg.workers_hard_limit)))
        return cfg


@dataclass
class Injection:
    inject_id: str
    sent_id: str
    stage: str  # parse / edit / convert
    fault: str  # INJECTION_BUSINESS_CODES 的键
    target_word: int | None = None  # 针对特定词的故障

    @staticmethod
    def load_list(path: str | Path | None) -> list["Injection"]:
        if not path:
            return []
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = raw.get("injections", raw if isinstance(raw, list) else [])
        out: list[Injection] = []
        for item in items:
            fault = item["fault"]
            if fault not in INJECTION_BUSINESS_CODES:
                raise ValueError(f"未知注入故障类型: {fault}")
            stage = item["stage"]
            if stage not in VALID_STAGES:
                raise ValueError(f"未知注入阶段: {stage}")
            out.append(
                Injection(
                    inject_id=item["inject_id"],
                    sent_id=item["sent_id"],
                    stage=stage,
                    fault=fault,
                    target_word=item.get("target_word"),
                )
            )
        return out


def load_concurrency(path: str | Path | None, default: AppConfig | None = None) -> AppConfig:
    cfg = default or AppConfig()
    if path is None:
        return cfg
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "max_workers" in data:
        cfg.max_workers = max(1, min(int(data["max_workers"]), int(cfg.workers_hard_limit)))
    if "workers_hard_limit" in data:
        cfg.workers_hard_limit = int(data["workers_hard_limit"])
        cfg.max_workers = min(cfg.max_workers, cfg.workers_hard_limit)
    return cfg
