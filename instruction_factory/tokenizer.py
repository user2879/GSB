from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

from .util import fingerprint_object

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u3400-\u9fff]+")


class SubwordTokenizer:
    def __init__(self, rules: dict[str, Any]):
        self.rules = dict(rules)
        self.lowercase = bool(rules["lowercase"])
        self.ngram_size = int(rules["ngram_size"])
        self.drop_chars = str(rules.get("drop_chars", ""))
        self.min_chars = int(rules.get("min_chars", 1))
        self._cache: dict[str, tuple[str, ...]] = {}
        self.fingerprint = fingerprint_object(self.rules)

    def tokenize(self, text: str) -> tuple[str, ...]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        normalized = text.lower() if self.lowercase else text
        if self.drop_chars:
            translation = {ord(char): " " for char in self.drop_chars}
            normalized = normalized.translate(translation)
        tokens: set[str] = set()
        for part in TOKEN_RE.findall(normalized):
            if len(part) < self.min_chars:
                continue
            if len(part) <= self.ngram_size:
                tokens.add(part)
                continue
            for index in range(len(part) - self.ngram_size + 1):
                tokens.add(part[index : index + self.ngram_size])
        result = tuple(sorted(tokens))
        self._cache[text] = result
        return result

    def jaccard_decimal(self, left: str | tuple[str, ...], right: str | tuple[str, ...]) -> Decimal:
        left_tokens = set(self.tokenize(left) if isinstance(left, str) else left)
        right_tokens = set(self.tokenize(right) if isinstance(right, str) else right)
        union = left_tokens | right_tokens
        if not union:
            return Decimal(0)
        intersection = left_tokens & right_tokens
        return Decimal(len(intersection)) / Decimal(len(union))
