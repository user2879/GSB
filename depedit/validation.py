"""依存结构校验：头词不越界、无环、每句恰好一个根。"""
from __future__ import annotations

from .errors import ECycleDetected, EHeadOutOfRange, EMultipleRoots
from .models import Sentence


def validate_sentence(sent: Sentence) -> None:
    n = sent.word_count()
    roots: list[int] = []
    heads: dict[int, int] = {}
    for arc in sent.arcs:
        head = arc.normalized_head()
        if head < 0 or head > n:
            raise EHeadOutOfRange(
                f"句 {sent.sent_id} 词 {arc.word_idx} 的头词 {head} 越界（合法 0..{n}）"
            )
        if head == 0:
            roots.append(arc.word_idx)
        else:
            heads[arc.word_idx] = head
    if len(roots) != 1:
        raise EMultipleRoots(f"句 {sent.sent_id} 根节点数量为 {len(roots)}，要求恰好 1 个")
    _detect_cycle(sent.sent_id, heads, roots[0], n)


def _detect_cycle(sent_id: str, heads: dict[int, int], root: int, n: int) -> None:
    # 每个非根词沿头指针向上追溯，带访问色标记。
    color: dict[int, int] = {i: 0 for i in range(1, n + 1)}

    def visit(node: int) -> None:
        color[node] = 1
        nxt = heads.get(node)
        if nxt is not None:
            if color.get(nxt) == 1:
                raise ECycleDetected(f"句 {sent_id} 检测到环依赖，涉及词 {node} -> {nxt}")
            if color.get(nxt) == 0:
                visit(nxt)
        color[node] = 2

    for node in range(1, n + 1):
        if color[node] == 0:
            visit(node)
