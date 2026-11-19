from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .errors import ValidationError


_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationRules:
    case_fold: bool = True
    width_fold: bool = True
    whitespace_collapse: bool = True

    @classmethod
    def from_mapping(cls, value: dict) -> "NormalizationRules":
        if not isinstance(value, dict):
            raise ValidationError("E_INDEX_INVALID", "normalization must be an object")
        width_fold = value.get("width_fold", value.get("fullwidth_fold", True))
        rules = cls(
            case_fold=value.get("case_fold", True),
            width_fold=width_fold,
            whitespace_collapse=value.get("whitespace_collapse", True),
        )
        if not all(isinstance(item, bool) for item in (rules.case_fold, rules.width_fold, rules.whitespace_collapse)):
            raise ValidationError("E_INDEX_INVALID", "normalization switches must be booleans")
        return rules

    def to_dict(self) -> dict:
        return {
            "case_fold": self.case_fold,
            "width_fold": self.width_fold,
            "whitespace_collapse": self.whitespace_collapse,
        }


def normalize_term(term: str, rules: NormalizationRules) -> str:
    if not isinstance(term, str):
        raise ValidationError("E_DOC_INVALID", f"term must be a string, got {type(term).__name__}")
    normalized = term
    if rules.width_fold:
        normalized = unicodedata.normalize("NFKC", normalized)
    if rules.case_fold:
        normalized = normalized.casefold()
    if rules.whitespace_collapse:
        normalized = _WHITESPACE.sub(" ", normalized).strip()
    else:
        normalized = normalized.strip()
    if not normalized:
        raise ValidationError("E_DOC_INVALID", "term becomes empty after normalization")
    if any(char.isspace() and char != " " for char in normalized):
        raise ValidationError("E_DOC_INVALID", "term contains unsupported whitespace")
    return normalized


def normalize_vocabulary(terms: list[str], rules: NormalizationRules) -> tuple[str, ...]:
    normalized = []
    for term in terms:
        if not isinstance(term, str):
            raise ValidationError("E_INDEX_INVALID", "controlled vocabulary entries must be strings")
        normalized.append(normalize_term(term, rules))
    if len(normalized) != len(set(normalized)):
        raise ValidationError("E_INDEX_INVALID", "controlled vocabulary collides after normalization")
    return tuple(sorted(normalized))

