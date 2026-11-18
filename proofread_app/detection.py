"""候选错误检测：n-gram 窗口、规则匹配、相似度过滤、重叠消解。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .config import AppConfig
from .fingerprint import content_fingerprint
from .models import (
    Candidate,
    Rule,
    RuleResource,
    Sample,
    SuppressedItem,
)


@dataclass
class SegmentedText:
    sample_id: str
    text: str
    windows: List[Dict[str, int]]

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text_fingerprint": content_fingerprint(self.text),
            "windows": self.windows,
        }


def segment_sample(sample: Sample, config: AppConfig) -> SegmentedText:
    """按字符 n-gram 阶数切分字级候选窗口（1..N 阶全覆盖）。"""
    text = sample.text
    windows: List[Dict[str, int]] = []
    limit = min(config.ngram_order, len(text))
    for order in range(1, limit + 1):
        for start in range(0, len(text) - order + 1):
            windows.append({"start": start, "end": start + order, "order": order})
    return SegmentedText(sample.sample_id, text, windows)


def _jaccard(left: str, right: str) -> float:
    set_left = set(left)
    set_right = set(right)
    if not set_left and not set_right:
        return 1.0
    union = set_left | set_right
    return len(set_left & set_right) / len(union)


def _window_for(text: str, start: int, pattern_len: int, ngram_order: int) -> str:
    """取覆盖匹配区间的 n-gram 窗口（靠近句尾时左扩，保证窗口长度尽量为 N）。"""
    window_len = min(ngram_order, len(text))
    if window_len <= pattern_len:
        return text[start:start + pattern_len]
    end = min(len(text), start + window_len)
    begin = max(0, end - window_len)
    return text[begin:end]


def detect_candidates(sample: Sample, segmented: SegmentedText,
                      resource: RuleResource, config: AppConfig
                      ) -> Tuple[List[Candidate], List[SuppressedItem]]:
    """对单条样本执行候选生成与重叠消解。

    匹配完全基于规则 pattern 子串定位，相似度为确定性字符集合 Jaccard，
    不依赖任何外部模型或随机过程。
    """
    text = sample.text
    raw: List[Candidate] = []
    rule_map = resource.rule_map()
    for rule in resource.rules:
        if len(rule.pattern) > config.ngram_order:
            # 超过 n-gram 阶数的规则无法被窗口覆盖，跳过
            continue
        cursor = 0
        while True:
            index = text.find(rule.pattern, cursor)
            if index < 0:
                break
            end = index + len(rule.pattern)
            window = _window_for(text, index, len(rule.pattern), config.ngram_order)
            similarity = _jaccard(rule.pattern, window)
            if similarity + 1e-12 >= config.similarity_threshold:
                raw.append(Candidate(
                    sample_id=sample.sample_id,
                    start=index,
                    end=end,
                    etype=rule.etype,
                    rule_id=rule.rule_id,
                    evidence=rule.pattern,
                    window=window,
                    similarity=similarity,
                ))
            cursor = index + 1

    # 消解顺序：区间长度降序、规则 ID 字典序、起始位置升序
    ordered = sorted(raw, key=lambda item: item.order_tuple())
    accepted: List[Candidate] = []
    suppressed: List[SuppressedItem] = []
    occupied: List[Candidate] = []
    for candidate in ordered:
        overlapper = next(
            (keeper for keeper in occupied
             if candidate.start < keeper.end and keeper.start < candidate.end),
            None,
        )
        if overlapper is None:
            occupied.append(candidate)
            accepted.append(candidate)
        else:
            candidate.suppressed = True
            candidate.suppressed_by = overlapper.rule_id
            suppressed.append(SuppressedItem(
                sample_id=sample.sample_id,
                start=candidate.start,
                end=candidate.end,
                etype=candidate.etype,
                rule_id=candidate.rule_id,
                suppressed_by=overlapper.rule_id,
            ))

    # 最终对外顺序统一按起始位置、规则 ID 排列，保证并发/串行一致
    accepted.sort(key=lambda item: (item.start, item.rule_id))
    suppressed.sort(key=lambda item: (item.start, item.rule_id))
    return accepted, suppressed


def referenced_rules(candidates: Sequence[Candidate],
                     suppressed: Sequence[SuppressedItem]) -> List[str]:
    ids = {item.rule_id for item in candidates}
    ids.update(item.rule_id for item in suppressed)
    return sorted(ids)
