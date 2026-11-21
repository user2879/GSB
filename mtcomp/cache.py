from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import sha256_value
from .model import SourceSentence, TranslationState

ALGORITHM_VERSION = "mtcomp-fixed-v1"


@dataclass(frozen=True)
class SentenceKeys:
    content_hash: str
    referenced_entries: tuple[tuple[str, str], ...]
    referenced_sources: tuple[str, ...]
    alignment_key: str
    decode_key: str
    confidence_key: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceKeys":
        return cls(
            content_hash=data["content_hash"],
            referenced_entries=tuple(
                (item["key"], item["fingerprint"])
                for item in data["referenced_entries"]
            ),
            referenced_sources=tuple(data["referenced_sources"]),
            alignment_key=data["alignment_key"],
            decode_key=data["decode_key"],
            confidence_key=data["confidence_key"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "referenced_entries": [
                {"key": key, "fingerprint": fingerprint}
                for key, fingerprint in self.referenced_entries
            ],
            "referenced_sources": list(self.referenced_sources),
            "alignment_key": self.alignment_key,
            "decode_key": self.decode_key,
            "confidence_key": self.confidence_key,
        }


def sentence_keys(
    state: TranslationState,
    sentence: SourceSentence,
    *,
    bits: int | None = None,
    tier: int | None = None,
) -> SentenceKeys:
    if state.params is None or state.config is None:
        raise ValueError("translation state is incomplete")
    effective_bits = state.params.fixed_point_bits if bits is None else bits
    effective_tier = state.params.quantization_tier if tier is None else tier
    entries: list[tuple[str, str]] = []
    sources: set[str] = set()
    for token in sentence.tokens:
        bucket = state.lex_bucket(token)
        if bucket:
            sources.add(token)
        for entry in bucket:
            entries.append((entry.key, entry.fingerprint()))
    entries.sort(key=lambda item: item[0])
    params_fp = state.params.fingerprint()
    config_fp = state.config.result_fingerprint()
    common = {
        "algorithm": ALGORITHM_VERSION,
        "content_hash": sentence.content_hash,
        "entries": entries,
        "params": params_fp,
        "compile": config_fp,
        "bits": effective_bits,
        "tier": effective_tier,
    }
    alignment_key = sha256_value({"stage": "alignment", **common})
    decode_key = sha256_value({"stage": "decode", "alignment_key": alignment_key, **common})
    confidence_key = sha256_value(
        {"stage": "confidence", "decode_key": decode_key, **common}
    )
    return SentenceKeys(
        content_hash=sentence.content_hash,
        referenced_entries=tuple(entries),
        referenced_sources=tuple(sorted(sources)),
        alignment_key=alignment_key,
        decode_key=decode_key,
        confidence_key=confidence_key,
    )
