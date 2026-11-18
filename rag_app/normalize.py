"""Deterministic text normalization and tokenization.

Steps (each independently switchable via config ``normalization``):

* ``fold_full_half`` -- map full-width ASCII / full-width space to half
  width and full-width punctuation used in the samples to ASCII equivalents;
* ``case_fold`` -- lower-case all cased characters (``str.casefold``);
* ``whitespace_merge`` -- collapse runs of whitespace to a single space and
  strip leading / trailing whitespace.

Tokenization splits on whitespace after normalization, so the controlled
vocabulary generator only emits tokens that survive a round trip.
"""

from __future__ import annotations

import unicodedata


def fold_full_half(text: str) -> str:
    """Fold full-width ASCII and common full-width punctuation."""
    chars: list[str] = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:  # ideographic space
            chars.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:  # full-width ASCII block
            chars.append(chr(code - 0xFEE0))
        else:
            name = unicodedata.name(ch, "")
            if "FULLWIDTH" in name:
                chars.append(unicodedata.normalize("NFKC", ch))
            else:
                chars.append(ch)
    return "".join(chars)


def case_fold(text: str) -> str:
    return text.casefold()


def merge_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_text(text: str, rules: dict[str, bool]) -> str:
    if rules.get("fold_full_half", True):
        text = fold_full_half(text)
    if rules.get("case_fold", True):
        text = case_fold(text)
    if rules.get("whitespace_merge", True):
        text = merge_whitespace(text)
    return text


def tokenize(text: str, rules: dict[str, bool]) -> list[str]:
    normalized = normalize_text(text, rules)
    if not normalized:
        return []
    return normalized.split(" ")

