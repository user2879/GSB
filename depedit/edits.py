"""四类编辑操作：改头词、换关系、头词整体平移、标记重解析。

每个编辑先在统一表示上做深拷贝试写，结构校验通过后才提交；
违反约束抛 E_HEAD_STRUCTURE 且原状态保持不变。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .errors import EHeadStructure
from .models import Sentence
from .validation import validate_sentence


@dataclass
class EditResult:
    op: str
    sent_id: str
    affected_words: list[int] = field(default_factory=list)
    affected_arcs: list[int] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def _commit_if_valid(sent: Sentence, trial: Sentence, result: EditResult) -> EditResult:
    try:
        validate_sentence(trial)
    except EHeadStructure:
        raise
    except Exception as exc:  # 统一归一为 E_HEAD_STRUCTURE
        raise EHeadStructure(str(exc)) from exc
    sent.arcs = trial.arcs
    sent.words = trial.words
    sent.needs_reparse = trial.needs_reparse
    return result


def set_head(sent: Sentence, word_idx: int, new_head: int) -> EditResult:
    if not 1 <= word_idx <= sent.word_count():
        raise EHeadStructure(f"词序号 {word_idx} 越界")
    trial = copy.deepcopy(sent)
    old_head = trial.arc_by_word(word_idx).normalized_head()
    trial.arc_by_word(word_idx).head = new_head
    arc = trial.arc_by_word(word_idx)
    arc.referenced_words = sorted({word_idx, new_head} - {0})
    result = EditResult(
        op="set_head",
        sent_id=sent.sent_id,
        affected_words=sorted({word_idx, old_head, new_head} - {0}),
        affected_arcs=[word_idx],
        detail={"word_idx": word_idx, "old_head": old_head, "new_head": new_head},
    )
    return _commit_if_valid(sent, trial, result)


def replace_relation(
    sent: Sentence, relation_old: str, relation_new: str, *, word_idx: int | None = None
) -> EditResult:
    trial = copy.deepcopy(sent)
    affected: list[int] = []
    for arc in trial.arcs:
        if arc.relation == relation_old and (word_idx is None or arc.word_idx == word_idx):
            arc.relation = relation_new
            affected.append(arc.word_idx)
    if not affected:
        raise EHeadStructure(
            f"句 {sent.sent_id} 未找到关系 {relation_old}"
            + (f"（词 {word_idx}）" if word_idx else "")
        )
    result = EditResult(
        op="replace_relation",
        sent_id=sent.sent_id,
        affected_words=list(affected),
        affected_arcs=list(affected),
        detail={"old": relation_old, "new": relation_new},
    )
    return _commit_if_valid(sent, trial, result)


def shift_heads(sent: Sentence, delta: int) -> EditResult:
    if delta == 0:
        return EditResult(op="shift_heads", sent_id=sent.sent_id, detail={"delta": 0})
    trial = copy.deepcopy(sent)
    affected: list[int] = []
    for arc in trial.arcs:
        head = arc.normalized_head()
        if head == 0:
            continue  # 根节点缺失头词不参与平移
        arc.head = head + delta
        arc.referenced_words = sorted({arc.word_idx, arc.head} - {0})
        affected.append(arc.word_idx)
    result = EditResult(
        op="shift_heads",
        sent_id=sent.sent_id,
        affected_words=list(affected),
        affected_arcs=list(affected),
        detail={"delta": delta},
    )
    return _commit_if_valid(sent, trial, result)


def mark_reparse(sent: Sentence) -> EditResult:
    sent.needs_reparse = True
    return EditResult(
        op="mark_reparse",
        sent_id=sent.sent_id,
        affected_words=[w.idx for w in sent.words],
        affected_arcs=[a.word_idx for a in sent.arcs],
        detail={"needs_reparse": True},
    )
