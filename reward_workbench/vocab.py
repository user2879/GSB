"""受控词表。

样本生成器与加载校验层共用同一份词表，确保“词项落在受控词表内”可判定。
词表固定且版本化，保证跨运行、跨平台逐整数一致。
"""
from __future__ import annotations

VOCAB_VERSION = "vocab-1"

# 三个语义槽位各自的受控词；prompt/chosen/rejected 共用同一总词表，
# 但保留槽位前缀以便生成受控长度的句子。
_PROMPT_WORDS = (
    "summarize", "translate", "classify", "rewrite", "extract",
    "question", "document", "instruction", "context", "request",
)
_CHOSEN_WORDS = (
    "accurate", "concise", "relevant", "faithful", "complete",
    "answer", "summary", "result", "output", "response",
)
_REJECT_WORDS = (
    "verbose", "vague", "offtopic", "hallucinated", "partial",
    "guess", "noise", "error", "filler", "repeat",
)
_CONNECTORS = ("the", "a", "for", "of", "and", "with", "in", "to")

# 总受控词表（去重后排序，确定性）
CONTROLLED_VOCAB: tuple[str, ...] = tuple(
    sorted(set(_PROMPT_WORDS) | set(_CHOSEN_WORDS) | set(_REJECT_WORDS) | set(_CONNECTORS))
)
_VOCAB_SET = frozenset(CONTROLLED_VOCAB)


def vocabulary() -> tuple[str, ...]:
    return CONTROLLED_VOCAB


def slot_words(slot: str) -> tuple[str, ...]:
    if slot == "prompt":
        return _PROMPT_WORDS
    if slot == "chosen":
        return _CHOSEN_WORDS
    if slot == "rejected":
        return _REJECT_WORDS
    raise ValueError(f"unknown slot {slot!r}")


def connector_words() -> tuple[str, ...]:
    return _CONNECTORS


def all_in_vocab(tokens: list[str]) -> bool:
    return all(tok in _VOCAB_SET for tok in tokens)
