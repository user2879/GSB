"""编辑决策落地：生成修正文本与逐处编辑记录。"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .models import (
    Candidate,
    EditRecord,
    RuleResource,
    Sample,
    ScoreBreakdown,
    EDIT_REPLACE,
    EDIT_DELETE,
    EDIT_INSERT,
)
from .scoring import DECISION_APPLY


def build_edits(sample: Sample, candidates: Sequence[Candidate],
                scores: Sequence[ScoreBreakdown], resource: RuleResource,
                precision: str) -> Tuple[List[EditRecord], str]:
    """根据决策结果应用编辑，返回编辑记录与修正后文本。

    编辑从右向左落地，避免区间位移；重叠候选已在检测阶段消解，
    因此 apply 集合互不重叠。
    """
    score_map = {(item.rule_id, item.start): item for item in scores}
    rule_map = resource.rule_map()
    edits: List[EditRecord] = []
    applied: List[Tuple[int, int, EditRecord]] = []

    for candidate in candidates:
        breakdown = score_map.get((candidate.rule_id, candidate.start))
        if breakdown is None or breakdown.decision != DECISION_APPLY:
            continue
        rule = rule_map[candidate.rule_id]
        original = sample.text[candidate.start:candidate.end]
        if rule.edit_type == EDIT_INSERT:
            # replacement 中以 "|" 分隔：锚点字符|插入字；插在锚点之前
            if "|" in rule.replacement:
                anchor, insert_char = rule.replacement.split("|", 1)
                anchor_index = original.find(anchor)
                insert_at = anchor_index + len(anchor)
                corrected = original[:insert_at] + insert_char + original[insert_at:]
            else:
                corrected = rule.replacement + original
        elif rule.edit_type == EDIT_DELETE:
            corrected = ""
        else:
            corrected = rule.replacement
        record = EditRecord(
            sample_id=sample.sample_id,
            start=candidate.start,
            end=candidate.end,
            original=original,
            corrected=corrected,
            edit_type=rule.edit_type,
            rule_id=rule.rule_id,
            etype=rule.etype,
            confidence=breakdown.total_confidence,
            precision=precision,
            boundary=breakdown.boundary,
        )
        applied.append((candidate.start, candidate.end, record))

    # 右向左应用，插入/替换/删除统一处理
    text = sample.text
    for start, end, record in sorted(applied, key=lambda item: item[0], reverse=True):
        rule = rule_map[record.rule_id]
        if rule.edit_type == EDIT_INSERT:
            insert_pos = start + rule.insert_offset
            text = text[:insert_pos] + rule.replacement + text[insert_pos:]
        elif rule.edit_type == EDIT_DELETE:
            text = text[:start] + text[end:]
        else:
            text = text[:start] + rule.replacement + text[end:]
        edits.append(record)

    edits.sort(key=lambda item: (item.start, item.rule_id))
    return edits, text
