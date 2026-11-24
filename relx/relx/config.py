"""抽取配置 YAML 的加载与校验。"""

from dataclasses import dataclass

from . import mini_yaml
from .errors import E_CONFIG_INVALID, RelxError
from .fixedpoint import ALLOWED_Q_LEVELS, ALLOWED_WIDTHS


@dataclass(frozen=True)
class AppConfig:
    entity_types: tuple
    relation_types: tuple
    max_term_length: int
    score_threshold: float
    fixedpoint_width: int
    quant_level: int
    thread_count: int
    feature_budget_per_batch: int
    snapshot_commit_batches: int

    def fingerprint_material(self):
        return {
            "entity_types": list(self.entity_types),
            "relation_types": list(self.relation_types),
            "max_term_length": self.max_term_length,
            "score_threshold": self.score_threshold,
            "fixedpoint_width": self.fixedpoint_width,
            "quant_level": self.quant_level,
            "thread_count": self.thread_count,
            "feature_budget_per_batch": self.feature_budget_per_batch,
            "snapshot_commit_batches": self.snapshot_commit_batches,
        }


def _require_positive_int(data, key):
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RelxError(E_CONFIG_INVALID, f"{key} 必须为正整数，实际为 {value!r}")
    return value


def load_config(path: str) -> AppConfig:
    data = mini_yaml.load(path)
    required = (
        "entity_types",
        "relation_types",
        "max_term_length",
        "score_threshold",
        "fixedpoint_width",
        "quant_level",
        "thread_count",
        "feature_budget_per_batch",
        "snapshot_commit_batches",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise RelxError(E_CONFIG_INVALID, f"配置缺少必填字段: {missing}")

    entity_types = data["entity_types"]
    relation_types = data["relation_types"]
    if not isinstance(entity_types, list) or not entity_types:
        raise RelxError(E_CONFIG_INVALID, "entity_types 必须为非空列表")
    if not isinstance(relation_types, list) or not relation_types:
        raise RelxError(E_CONFIG_INVALID, "relation_types 必须为非空列表")
    if any(not isinstance(name, str) for name in entity_types + relation_types):
        raise RelxError(E_CONFIG_INVALID, "类型集合元素必须为字符串")
    if len(set(entity_types)) != len(entity_types):
        raise RelxError(E_CONFIG_INVALID, "entity_types 存在重复类型")
    if len(set(relation_types)) != len(relation_types):
        raise RelxError(E_CONFIG_INVALID, "relation_types 存在重复类型")

    max_term_length = _require_positive_int(data, "max_term_length")
    thread_count = _require_positive_int(data, "thread_count")
    feature_budget = _require_positive_int(data, "feature_budget_per_batch")
    snapshot_batches = _require_positive_int(data, "snapshot_commit_batches")

    threshold = data["score_threshold"]
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise RelxError(E_CONFIG_INVALID, "score_threshold 必须为数值")

    width = data["fixedpoint_width"]
    quant = data["quant_level"]
    if width not in ALLOWED_WIDTHS:
        raise RelxError(E_CONFIG_INVALID, f"定点位宽仅允许 {ALLOWED_WIDTHS}，实际 {width}")
    if quant not in ALLOWED_Q_LEVELS:
        raise RelxError(E_CONFIG_INVALID, f"量化档位仅允许 {ALLOWED_Q_LEVELS}，实际 {quant}")

    return AppConfig(
        entity_types=tuple(entity_types),
        relation_types=tuple(relation_types),
        max_term_length=max_term_length,
        score_threshold=float(threshold),
        fixedpoint_width=width,
        quant_level=quant,
        thread_count=thread_count,
        feature_budget_per_batch=feature_budget,
        snapshot_commit_batches=snapshot_batches,
    )
