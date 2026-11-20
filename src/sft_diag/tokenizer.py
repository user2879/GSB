"""确定性子词切分：词表最长前缀匹配，未登录字符按单字符回退。"""
from __future__ import annotations

import unicodedata

from .models import Vocab


class Tokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab
        self._token_set = frozenset(vocab.tokens)
        self._by_first: dict[str, list[str]] = {}
        for token in vocab.tokens:
            self._by_first.setdefault(token[0], []).append(token)
        for options in self._by_first.values():
            options.sort(key=lambda item: (-len(item), item))

    def normalize(self, text: str) -> str:
        """NFKC 归一 + 小写，保证跨平台一致。"""
        return unicodedata.normalize("NFKC", text).lower()

    def tokenize(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        tokens: list[str] = []
        index = 0
        total = len(normalized)
        while index < total:
            char = normalized[index]
            if char.isspace():
                index += 1
                continue
            matched = None
            for candidate in self._by_first.get(char, ()):
                if normalized.startswith(candidate, index):
                    matched = candidate
                    break
            if matched is None:
                tokens.append(char)
                index += 1
            else:
                tokens.append(matched)
                index += len(matched)
        return tokens

    def token_set(self, text: str) -> frozenset[str]:
        return frozenset(self.tokenize(text))

    def vocab_coverage(self, text: str) -> float:
        tokens = self.tokenize(text)
        if not tokens:
            return 0.0
        in_vocab = sum(1 for token in tokens if token in self._token_set)
        return in_vocab / len(tokens)
