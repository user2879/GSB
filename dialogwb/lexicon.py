from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import fingerprint


class LexiconError(ValueError):
    pass


@dataclass(frozen=True)
class TokenMatch:
    token: str
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {"token": self.token, "start": self.start, "end": self.end}


class Lexicon:
    def __init__(self, value: dict[str, Any]) -> None:
        if not isinstance(value, dict):
            raise LexiconError("lexicon must be a JSON object")
        tokens_value = value.get("tokens")
        if not isinstance(tokens_value, list) or not tokens_value:
            raise LexiconError("tokens must be a non-empty array")
        tokens: list[str] = []
        seen: set[str] = set()
        for item in tokens_value:
            if not isinstance(item, str) or not item:
                raise LexiconError("tokens must be non-empty strings")
            if item not in seen:
                seen.add(item)
                tokens.append(item)
        rules = value.get("rules", {})
        if not isinstance(rules, dict):
            raise LexiconError("rules must be an object")
        case_sensitive = rules.get("case_sensitive", False)
        if not isinstance(case_sensitive, bool):
            raise LexiconError("rules.case_sensitive must be boolean")
        punctuation = rules.get("punctuation", [])
        if not isinstance(punctuation, list) or any(
            not isinstance(item, str) or not item for item in punctuation
        ):
            raise LexiconError("rules.punctuation must be an array of strings")
        subwords_value = rules.get("subwords", {})
        if not isinstance(subwords_value, dict):
            raise LexiconError("rules.subwords must be an object")
        subwords: dict[str, list[str]] = {}
        for key in sorted(subwords_value):
            components = subwords_value[key]
            if not isinstance(key, str) or not key or not isinstance(components, list):
                raise LexiconError("subword rules must map strings to arrays")
            if not components or any(
                not isinstance(component, str) or not component
                for component in components
            ):
                raise LexiconError(f"invalid subword components for {key}")
            unknown = [item for item in components if item not in seen]
            if unknown:
                raise LexiconError(
                    f"subword components not in token list: {', '.join(unknown)}"
                )
            if "".join(components) != key:
                raise LexiconError(
                    f"subword components for {key} must concatenate to the key"
                )
            subwords[key] = list(components)
        self.value = {
            "version": value.get("version", "unversioned"),
            "tokens": tokens,
            "rules": {
                "case_sensitive": case_sensitive,
                "punctuation": list(punctuation),
                "subwords": subwords,
            },
        }
        self.version = str(self.value["version"])
        self.case_sensitive = case_sensitive
        self.punctuation = set(punctuation)
        self.tokens = tokens
        self.token_set = set(tokens)
        self._fold_map = {
            (token if case_sensitive else token.casefold()): token for token in tokens
        }
        self._fold_tokens = sorted(self._fold_map, key=lambda item: (-len(item), item))
        self._subwords = sorted(
            (
                (key if case_sensitive else key.casefold(), key, components)
                for key, components in subwords.items()
            ),
            key=lambda item: (-len(item[0]), item[0]),
        )
        self.fingerprint = fingerprint(self.value)

    @classmethod
    def from_path(cls, path: str | Path) -> "Lexicon":
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise LexiconError(f"cannot load lexicon: {exc}") from exc
        return cls(value)

    def is_separator(self, char: str) -> bool:
        if char.isspace() or char in self.punctuation:
            return True
        category = unicodedata.category(char)
        return category.startswith("P") or category.startswith("Z")

    def split(self, text: str) -> tuple[list[TokenMatch], list[dict[str, Any]]]:
        matches: list[TokenMatch] = []
        unknown: list[dict[str, Any]] = []
        index = 0
        length = len(text)
        while index < length:
            if self.is_separator(text[index]):
                index += 1
                continue
            subword = self._match_subword(text, index)
            if subword is not None:
                matches.extend(subword)
                index = subword[-1].end
                continue
            token = self._match_token(text, index)
            if token is not None:
                matches.append(token)
                index = token.end
                continue
            end = self._unknown_end(text, index)
            unknown.append(
                {
                    "token": text[index:end],
                    "start": index,
                    "end": end,
                }
            )
            index = end
        matches.sort(key=lambda item: (item.start, item.end, item.token))
        return matches, unknown

    def _normalized(self, text: str) -> str:
        return text if self.case_sensitive else text.casefold()

    def _match_subword(self, text: str, index: int) -> list[TokenMatch] | None:
        normalized = self._normalized(text)
        for folded_key, original_key, components in self._subwords:
            if not normalized.startswith(folded_key, index):
                continue
            if not self._boundary_ok(text, index, index + len(folded_key)):
                continue
            cursor = index
            result: list[TokenMatch] = []
            for component in components:
                component_length = len(component)
                normalized_piece = normalized[cursor : cursor + component_length]
                original_piece = text[cursor : cursor + component_length]
                canonical = self._fold_map.get(
                    normalized_piece if not self.case_sensitive else original_piece
                )
                if canonical is None:
                    result = []
                    break
                result.append(TokenMatch(canonical, cursor, cursor + component_length))
                cursor += component_length
            if result and cursor == index + len(original_key):
                return result
        return None

    def _match_token(self, text: str, index: int) -> TokenMatch | None:
        normalized = self._normalized(text)
        for folded_token in self._fold_tokens:
            if normalized.startswith(folded_token, index):
                end = index + len(folded_token)
                if not self._boundary_ok(text, index, end):
                    continue
                canonical = self._fold_map[folded_token]
                return TokenMatch(canonical, index, end)
        return None

    def _boundary_ok(self, text: str, start: int, end: int) -> bool:
        if start < len(text) and text[start].isascii() and text[start].isalnum():
            if start > 0 and text[start - 1].isascii() and text[start - 1].isalnum():
                return False
        if end > 0 and text[end - 1].isascii() and text[end - 1].isalnum():
            if end < len(text) and text[end].isascii() and text[end].isalnum():
                return False
        return True

    def _has_match(self, text: str, index: int) -> bool:
        return (
            self._match_subword(text, index) is not None
            or self._match_token(text, index) is not None
        )

    def _unknown_end(self, text: str, index: int) -> int:
        end = index + 1
        while end < len(text) and not self.is_separator(text[end]):
            if self._has_match(text, end):
                break
            end += 1
        return end


def load_lexicon(path: str | Path) -> Lexicon:
    return Lexicon.from_path(path)
