"""领域数据模型：样本、规则资源、候选、分数分解、编辑记录。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# 四类错误类型（受控字表内的确定性合成错误）
WRONG_CHAR = "wrong_char"        # 错别字（混淆字对替换）
REPEAT = "repeat"                # 相邻重复词/叠字
TERM = "term"                    # 术语前后不一致
OMISSION = "omission"            # 疑似漏字

ERROR_TYPES = (WRONG_CHAR, REPEAT, TERM, OMISSION)

# 决策边界处的固定优先级（数值越小优先级越高）
ETYPE_PRIORITY = {
    WRONG_CHAR: 0,
    REPEAT: 1,
    OMISSION: 2,
    TERM: 3,
}

# 编辑动作类型
EDIT_REPLACE = "replace"
EDIT_DELETE = "delete"
EDIT_INSERT = "insert"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    context: Optional[str]
    truth: Optional[str]
    line_no: int
    rule_hints: List[str] = field(default_factory=list)

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "context": self.context,
            "truth": self.truth,
            "rule_hints": list(self.rule_hints),
        }


@dataclass(frozen=True)
class Rule:
    """统一规则模型，覆盖四类校对规则。

    edit_type:
      replace  -> 以 replacement 替换 [start, end)
      delete   -> 删除 [start, end)（重复词消解）
      insert   -> 在 start 位置插入 replacement（漏字补全）
    """
    rule_id: str
    etype: str
    pattern: str
    replacement: str
    edit_type: str
    weight: float
    insert_offset: int = 0

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuleResource:
    char_table: frozenset
    rules: List[Rule]
    rule_weights: Dict[str, float]
    source_fingerprint: str

    def rule_map(self) -> Dict[str, Rule]:
        return {r.rule_id: r for r in self.rules}


@dataclass(frozen=True)
class ScoringParams:
    detect_weights: Dict[str, float]
    base_scores: Dict[str, float]
    decision_threshold: float
    source_fingerprint: str


@dataclass
class Candidate:
    sample_id: str
    start: int
    end: int
    etype: str
    rule_id: str
    evidence: str
    window: str
    similarity: float
    suppressed: bool = False
    suppressed_by: Optional[str] = None

    @property
    def length(self) -> int:
        return self.end - self.start

    def order_tuple(self):
        # 消解顺序：区间长度降序、规则 ID 字典序、起始位置升序
        return (-self.length, self.rule_id, self.start)

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SuppressedItem:
    sample_id: str
    start: int
    end: int
    etype: str
    rule_id: str
    suppressed_by: str

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FixedPointQuant:
    """单条候选某一分项的定点量化中间量。"""
    raw: float
    quantized_int: int
    scale: float
    dequantized: float
    precision: str

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreBreakdown:
    sample_id: str
    rule_id: str
    etype: str
    start: int
    end: int
    detect_raw: float
    confidence_raw: float
    detect_quant: Dict[str, Any]
    confidence_quant: Dict[str, Any]
    weighted_detect: float
    weighted_confidence: float
    total_confidence: float
    decision: str
    boundary: bool
    precision: str

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageDiff:
    stage: str
    fp32_value: Optional[float]
    fp64_value: Optional[float]
    abs_diff: float
    rel_diff: float
    exceeds_tolerance: bool

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrecisionAudit:
    sample_id: str
    rule_id: str
    start: int
    stage_diffs: List[Dict[str, Any]]
    first_breach_stage: Optional[str]
    decision_flip: bool
    ranking_contradiction: bool
    attribution: str
    max_abs_diff: float

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EditRecord:
    sample_id: str
    start: int
    end: int
    original: str
    corrected: str
    edit_type: str
    rule_id: str
    etype: str
    confidence: float
    precision: str
    boundary: bool

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SampleResult:
    sample: Sample
    candidates: List[Candidate] = field(default_factory=list)
    suppressed: List[SuppressedItem] = field(default_factory=list)
    scores_fp32: List[ScoreBreakdown] = field(default_factory=list)
    scores_fp64: List[ScoreBreakdown] = field(default_factory=list)
    audits: List[PrecisionAudit] = field(default_factory=list)
    edits: List[EditRecord] = field(default_factory=list)
    corrected_text: str = ""
    boundary: bool = False
    used_rules: List[str] = field(default_factory=list)
