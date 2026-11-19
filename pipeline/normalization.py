from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional


HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_text(text: str, rules: dict) -> str:
    value = text
    if rules.get("nfkc", True):
        value = unicodedata.normalize("NFKC", value)
    if rules.get("casefold", True):
        value = value.casefold()
    if rules.get("collapse_whitespace", True):
        value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str, rules: dict) -> tuple[str, ...]:
    normalized = normalize_text(text, rules)
    return tuple(re.findall(rules["token_pattern"], normalized))


def token_script_language(token: str) -> Optional[str]:
    if HAN_RE.search(token):
        return "zh"
    if LATIN_RE.search(token):
        return "en"
    return None


def _ratio(part: int, total: int) -> float:
    return 0.0 if total == 0 else round(part / total, 12)


def detect_language(
    tokens: Iterable[str],
    *,
    declared: str,
    config: dict,
    translation_resolver,
) -> dict:
    tokens = tuple(tokens)
    zh_latin: list[str] = []
    en_latin: list[str] = []
    neutral = 0
    for token in tokens:
        lang = token_script_language(token)
        if lang == "zh":
            zh_latin.append(token)
        elif lang == "en":
            en_latin.append(token)
        else:
            neutral += 1
    script_total = len(zh_latin) + len(en_latin)
    zh_ratio = _ratio(len(zh_latin), script_total)
    en_ratio = _ratio(len(en_latin), script_total)
    mixed_ratio = min(zh_ratio, en_ratio) if script_total else 0.0

    zh_translated = sum(1 for token in zh_latin if translation_resolver.has_translation("zh", token))
    en_translated = sum(1 for token in en_latin if translation_resolver.has_translation("en", token))
    coverage_zh = _ratio(zh_translated, len(zh_latin))
    coverage_en = _ratio(en_translated, len(en_latin))

    lid = config["language_detection"]
    allowed = set(config["allowed_languages"])
    index_languages = tuple(config["index_languages"])
    if not lid["enabled"]:
        used = declared if declared in allowed else index_languages[0]
        detected = declared
        rule = "declared_language_detection_disabled"
    elif script_total == 0:
        candidates = sorted(index_languages, key=lambda lang: (-(coverage_zh if lang == "zh" else coverage_en), lang))
        used = candidates[0] if candidates else declared
        detected = used
        rule = "translation_coverage_tiebreak_no_script_tokens"
    elif mixed_ratio >= float(lid["mixed_ratio_threshold"]) and len(zh_latin) and len(en_latin):
        detected = "mixed"
        used = "mixed" if "mixed" in allowed else declared
        rule = "mixed_script_threshold"
    elif zh_ratio >= en_ratio:
        detected = "zh"
        used = "zh" if "zh" in index_languages else declared
        rule = "han_script_majority"
    else:
        detected = "en"
        used = "en" if "en" in index_languages else declared
        rule = "latin_script_majority"

    return {
        "declared": declared,
        "detected": detected,
        "used": used,
        "evidence": {
            "token_count": len(tokens),
            "han_token_count": len(zh_latin),
            "latin_token_count": len(en_latin),
            "neutral_token_count": neutral,
            "han_ratio": zh_ratio,
            "latin_ratio": en_ratio,
            "mixed_ratio": round(mixed_ratio, 12),
            "mixed_ratio_threshold": float(lid["mixed_ratio_threshold"]),
            "translation_coverage_zh": coverage_zh,
            "translation_coverage_en": coverage_en,
            "coverage_used_as": "tiebreak_only" if script_total else "primary",
            "rule": rule,
        },
    }


def token_counts(tokens: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(tokens).items()))


def quantize(value: float, fixed_config: dict) -> int:
    import numpy as np

    if not math.isfinite(value):
        raise ValueError("cannot quantize a non-finite value")
    bits = int(fixed_config["bits"])
    scale = int(fixed_config["scale"])
    maximum = (1 << (bits - 1)) - 1
    scaled = np.rint(np.float64(value) * np.float64(scale))
    return int(np.clip(scaled, 0.0, float(maximum)))

