from __future__ import annotations

import re
from typing import Iterable

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> tuple[str, ...]:
    tokens: set[str] = {match.group(0).lower() for match in _WORD_RE.finditer(text)}
    for match in _CJK_RE.finditer(text):
        value = match.group(0)
        if len(value) == 1:
            tokens.add(value)
        else:
            tokens.update(value[index : index + 2] for index in range(len(value) - 1))
    return tuple(sorted(tokens))


def jaccard_bp(
    left: Iterable[str] | str, right: Iterable[str] | str
) -> int:
    left_set = set(tokenize(left) if isinstance(left, str) else left)
    right_set = set(tokenize(right) if isinstance(right, str) else right)
    union = left_set | right_set
    if not union:
        return 10000
    intersection = left_set & right_set
    return (len(intersection) * 10000) // len(union)


def combined_topic_score_bp(query_bp: int, evidence_bp: int) -> int:
    return (query_bp * 3 + evidence_bp * 2) // 5
