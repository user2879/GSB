"""Query normalization: full/half-width folding, case folding, whitespace and
punctuation handling.

Every stage emits:
  * ``modifications`` - changed positions with old/new characters;
  * ``fingerprint``    - per-stage SHA-256 over stage input and output;
  * an alignment map from normalized indices back to original offsets.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

from .config import fingerprint
from .errors import EmptyQueryError


@dataclass
class NormalizationResult:
    original: str
    normalized: str
    modifications: List[dict] = field(default_factory=list)
    stage_fingerprints: List[dict] = field(default_factory=list)
    alignment: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "modifications": self.modifications,
            "stage_fingerprints": self.stage_fingerprints,
            "alignment": self.alignment,
        }


def _stage_fp(stage: str, text_in: str, text_out: str) -> dict:
    return {
        "stage": stage,
        "fingerprint": fingerprint({"in": text_in, "out": text_out}),
    }


def _fold_width(text: str) -> Tuple[str, List[dict], List[int]]:
    """NFKC folds full-width ASCII (e.g. ＡＢＣ, １２３） to half-width."""
    modifications: List[dict] = []
    alignment: List[int] = []
    out_chars: List[str] = []
    for idx, ch in enumerate(text):
        folded = unicodedata.normalize("NFKC", ch)
        # NFKC is a 1:N mapping only for a handful of compatibility glyphs;
        # every emitted output character keeps the source index.
        for out_idx, new_ch in enumerate(folded):
            if new_ch != ch or len(folded) > 1:
                modifications.append(
                    {
                        "stage": "width_fold",
                        "position": len(out_chars),
                        "original_position": idx,
                        "old": ch,
                        "new": new_ch,
                    }
                )
            out_chars.append(new_ch)
            alignment.append(idx)
    return "".join(out_chars), modifications, alignment


def _fold_case(text: str, alignment: List[int]) -> Tuple[str, List[dict], List[int]]:
    modifications: List[dict] = []
    out_chars: List[str] = []
    new_alignment: List[int] = []
    for idx, ch in enumerate(text):
        low = ch.casefold()
        for new_ch in low:
            if new_ch != ch:
                modifications.append(
                    {
                        "stage": "case_fold",
                        "position": len(out_chars),
                        "original_position": alignment[idx],
                        "old": ch,
                        "new": new_ch,
                    }
                )
            out_chars.append(new_ch)
            new_alignment.append(alignment[idx])
    return "".join(out_chars), modifications, new_alignment


def _merge_whitespace(text: str, alignment: List[int]) -> Tuple[str, List[dict], List[int]]:
    modifications: List[dict] = []
    out_chars: List[str] = []
    new_alignment: List[int] = []
    prev_space = False
    for idx, ch in enumerate(text):
        is_space = ch.isspace()
        if is_space and prev_space:
            modifications.append(
                {
                    "stage": "whitespace_merge",
                    "position": len(out_chars),
                    "original_position": alignment[idx],
                    "old": ch,
                    "new": "",
                }
            )
            continue
        if is_space and ch != " ":
            modifications.append(
                {
                    "stage": "whitespace_merge",
                    "position": len(out_chars),
                    "original_position": alignment[idx],
                    "old": ch,
                    "new": " ",
                }
            )
            ch = " "
        out_chars.append(ch)
        new_alignment.append(alignment[idx])
        prev_space = is_space
    return "".join(out_chars), modifications, new_alignment


def _strip_punctuation(text: str, alignment: List[int]) -> Tuple[str, List[dict], List[int]]:
    modifications: List[dict] = []
    out_chars: List[str] = []
    new_alignment: List[int] = []
    for idx, ch in enumerate(text):
        category = unicodedata.category(ch)
        if category.startswith("P") or category.startswith("S"):
            modifications.append(
                {
                    "stage": "punctuation_strip",
                    "position": len(out_chars),
                    "original_position": alignment[idx],
                    "old": ch,
                    "new": "",
                }
            )
            continue
        out_chars.append(ch)
        new_alignment.append(alignment[idx])
    return "".join(out_chars), modifications, new_alignment


def normalize_query(raw: str) -> NormalizationResult:
    """Run the four deterministic normalization stages.

    Raises :class:`EmptyQueryError` for empty / punctuation-only input after
    folding.
    """
    if not isinstance(raw, str) or raw == "":
        raise EmptyQueryError("query text is empty", stage="normalize")

    stage_inputs: List[str] = []
    all_modifications: List[dict] = []
    fingerprints: List[dict] = []

    stage_inputs.append(raw)
    text, mods, alignment = _fold_width(raw)
    all_modifications.extend(mods)
    fingerprints.append(_stage_fp("width_fold", stage_inputs[0], text))

    stage_inputs.append(text)
    text, mods, alignment = _fold_case(text, alignment)
    all_modifications.extend(mods)
    fingerprints.append(_stage_fp("case_fold", stage_inputs[1], text))

    stage_inputs.append(text)
    text, mods, alignment = _merge_whitespace(text, alignment)
    all_modifications.extend(mods)
    fingerprints.append(_stage_fp("whitespace_merge", stage_inputs[2], text))

    stage_inputs.append(text)
    text, mods, alignment = _strip_punctuation(text, alignment)
    all_modifications.extend(mods)
    fingerprints.append(_stage_fp("punctuation_strip", stage_inputs[3], text))

    # Punctuation removal may leave dangling spaces; collapse again and trim.
    stage_inputs.append(text)
    merged, mods, alignment = _merge_whitespace(text, alignment)
    if mods:
        all_modifications.extend(mods)
    lead = len(merged) - len(merged.lstrip())
    trail = len(merged) - len(merged.rstrip())
    text = merged.strip()
    if trail:
        alignment = alignment[lead : len(merged) - trail]
    else:
        alignment = alignment[lead:]
    if merged != stage_inputs[4]:
        fingerprints.append(_stage_fp("final_trim", stage_inputs[4], text))

    if text == "":
        raise EmptyQueryError("query is empty after normalization", stage="normalize")

    return NormalizationResult(
        original=raw,
        normalized=text,
        modifications=all_modifications,
        stage_fingerprints=fingerprints,
        alignment=alignment,
    )
