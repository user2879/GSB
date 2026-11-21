from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .alignment import AlignmentResult, CandidateSpan
from .canonical import fixed_to_permille


class NoAlignmentError(ValueError):
    def __init__(self, missing_tokens: tuple[str, ...]):
        self.missing_tokens = missing_tokens
        super().__init__(
            "no complete lexical alignment path; missing tokens: "
            + ", ".join(missing_tokens)
        )


@dataclass(frozen=True)
class TranslationFragment:
    source_start: int
    source_length: int
    source_tokens: tuple[str, ...]
    target_tokens: tuple[str, ...]
    used_entry_keys: tuple[str, ...]
    score_fixed: int
    confidence_sum_fixed: int
    target_chars: int

    @property
    def source_end(self) -> int:
        return self.source_start + self.source_length

    def to_dict(self, bits: int) -> dict[str, Any]:
        scale = 1 << bits
        return {
            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_tokens": list(self.source_tokens),
            "target_tokens": list(self.target_tokens),
            "translation": " ".join(self.target_tokens),
            "used_entry_keys": list(self.used_entry_keys),
            "score_fixed": self.score_fixed,
            "alignment_confidence_fixed": self.confidence_sum_fixed,
            "alignment_confidence_permille": fixed_to_permille(
                self.confidence_sum_fixed, bits
            ),
            "alignment_confidence_scale": scale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslationFragment":
        return cls(
            source_start=int(data["source_start"]),
            source_length=int(data["source_end"]) - int(data["source_start"]),
            source_tokens=tuple(data["source_tokens"]),
            target_tokens=tuple(data["target_tokens"]),
            used_entry_keys=tuple(data["used_entry_keys"]),
            score_fixed=int(data["score_fixed"]),
            confidence_sum_fixed=int(data["alignment_confidence_fixed"]),
            target_chars=sum(len(token) for token in data["target_tokens"]),
        )

    def signature_tuple(self) -> tuple[Any, ...]:
        return (
            self.source_start,
            self.source_length,
            self.target_tokens,
            self.source_tokens,
            self.used_entry_keys,
        )


@dataclass(frozen=True)
class DecodeResult:
    sentence_id: str
    decode_key: str
    fragments: tuple[TranslationFragment, ...]
    score_fixed: int
    second_score_fixed: int
    margin_fixed: int
    mean_alignment_conf_fixed: int
    used_entry_keys: tuple[str, ...]
    path_signature: str
    candidate_signature: str

    @property
    def target_tokens(self) -> tuple[str, ...]:
        tokens: list[str] = []
        for fragment in self.fragments:
            tokens.extend(fragment.target_tokens)
        return tuple(tokens)

    def path_tuple(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(fragment.signature_tuple() for fragment in self.fragments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence_id": self.sentence_id,
            "decode_key": self.decode_key,
            "fragments": [
                {
                    "source_start": fragment.source_start,
                    "source_length": fragment.source_length,
                    "source_tokens": list(fragment.source_tokens),
                    "target_tokens": list(fragment.target_tokens),
                    "used_entry_keys": list(fragment.used_entry_keys),
                    "score_fixed": fragment.score_fixed,
                    "alignment_confidence_fixed": fragment.confidence_sum_fixed,
                }
                for fragment in self.fragments
            ],
            "score_fixed": self.score_fixed,
            "second_score_fixed": self.second_score_fixed,
            "margin_fixed": self.margin_fixed,
            "mean_alignment_conf_fixed": self.mean_alignment_conf_fixed,
            "used_entry_keys": list(self.used_entry_keys),
            "path_signature": self.path_signature,
            "candidate_signature": self.candidate_signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecodeResult":
        fragments = []
        for item in data["fragments"]:
            fragments.append(
                TranslationFragment(
                    source_start=int(item["source_start"]),
                    source_length=int(item["source_length"]),
                    source_tokens=tuple(item["source_tokens"]),
                    target_tokens=tuple(item["target_tokens"]),
                    used_entry_keys=tuple(item["used_entry_keys"]),
                    score_fixed=int(item["score_fixed"]),
                    confidence_sum_fixed=int(item["alignment_confidence_fixed"]),
                    target_chars=sum(len(token) for token in item["target_tokens"]),
                )
            )
        return cls(
            sentence_id=data["sentence_id"],
            decode_key=data["decode_key"],
            fragments=tuple(fragments),
            score_fixed=int(data["score_fixed"]),
            second_score_fixed=int(data["second_score_fixed"]),
            margin_fixed=int(data["margin_fixed"]),
            mean_alignment_conf_fixed=int(data["mean_alignment_conf_fixed"]),
            used_entry_keys=tuple(data["used_entry_keys"]),
            path_signature=data["path_signature"],
            candidate_signature=data["candidate_signature"],
        )


@dataclass(frozen=True)
class _Path:
    fragments: tuple[TranslationFragment, ...]
    score: int

    @property
    def target_tokens(self) -> tuple[str, ...]:
        output: list[str] = []
        for fragment in self.fragments:
            output.extend(fragment.target_tokens)
        return tuple(output)

    @property
    def total_target_chars(self) -> int:
        return sum(fragment.target_chars for fragment in self.fragments)

    def path_key(self) -> tuple[Any, ...]:
        return (
            -self.score,
            self.target_tokens,
            self.total_target_chars,
            tuple(fragment.source_length for fragment in self.fragments),
            tuple(fragment.source_start for fragment in self.fragments),
            tuple(fragment.used_entry_keys for fragment in self.fragments),
        )

    def signature(self) -> str:
        payload = "|".join(
            f"{fragment.source_start}:{fragment.source_length}:"
            + "\x1f".join(fragment.target_tokens)
            for fragment in self.fragments
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fragment_from_candidate(candidate: CandidateSpan) -> TranslationFragment:
    return TranslationFragment(
        source_start=candidate.start,
        source_length=candidate.length,
        source_tokens=candidate.source_tokens,
        target_tokens=candidate.target_tokens,
        used_entry_keys=candidate.used_entry_keys,
        score_fixed=candidate.score_fixed,
        confidence_sum_fixed=candidate.confidence_sum_fixed,
        target_chars=candidate.target_chars,
    )


def viterbi_decode(
    alignment: AlignmentResult,
    *,
    decode_key: str,
    bits: int,
    scale: int,
) -> DecodeResult:
    if alignment.missing_tokens:
        raise NoAlignmentError(alignment.missing_tokens)
    source_length = len(alignment.tokens)
    by_end: dict[int, list[CandidateSpan]] = {}
    for candidate in alignment.candidates:
        by_end.setdefault(candidate.end, []).append(candidate)

    best: list[_Path | None] = [None] * (source_length + 1)
    second: list[_Path | None] = [None] * (source_length + 1)
    best_score_np = np.full(source_length + 1, np.iinfo(np.int64).min, dtype=np.int64)
    second_score_np = np.full(source_length + 1, np.iinfo(np.int64).min, dtype=np.int64)
    best[0] = _Path(fragments=(), score=0)
    best_score_np[0] = 0

    for end in range(1, source_length + 1):
        outgoing: list[_Path] = []
        for candidate in by_end.get(end, []):
            predecessors: list[_Path] = []
            if best[candidate.start] is not None:
                predecessors.append(best[candidate.start])  # type: ignore[arg-type]
            if second[candidate.start] is not None:
                predecessors.append(second[candidate.start])  # type: ignore[arg-type]
            for previous in predecessors:
                fragment = _fragment_from_candidate(candidate)
                path = _Path(
                    fragments=previous.fragments + (fragment,),
                    score=previous.score + candidate.score_fixed,
                )
                outgoing.append(path)
        if not outgoing:
            continue
        unique: dict[str, _Path] = {}
        for path in outgoing:
            signature = path.signature()
            existing = unique.get(signature)
            if existing is None or path.path_key() < existing.path_key():
                unique[signature] = path
        ordered = sorted(unique.values(), key=lambda path: path.path_key())
        if ordered:
            best[end] = ordered[0]
            best_score_np[end] = ordered[0].score
        if len(ordered) > 1:
            second[end] = ordered[1]
            second_score_np[end] = ordered[1].score

    final = best[source_length]
    if final is None:
        missing = alignment.missing_tokens or tuple(
            sorted(
                {
                    alignment.tokens[index]
                    for index in range(source_length)
                    if not any(
                        candidate.start <= index < candidate.end
                        for candidate in alignment.candidates
                    )
                }
            )
        )
        raise NoAlignmentError(missing)
    second_final = second[source_length]
    if second_final is None:
        margin = max(scale * 16, abs(int(best_score_np[source_length])) + scale)
        second_score = final.score - margin
    else:
        second_score = second_final.score
        margin = max(0, final.score - second_score)

    confidence_sum = 0
    for fragment in final.fragments:
        confidence_sum += fragment.confidence_sum_fixed
    mean_alignment_conf = confidence_sum // max(1, source_length)
    used_entries: set[str] = set()
    for fragment in final.fragments:
        used_entries.update(fragment.used_entry_keys)
    path_signature = final.signature()
    return DecodeResult(
        sentence_id=alignment.sentence_id,
        decode_key=decode_key,
        fragments=final.fragments,
        score_fixed=final.score,
        second_score_fixed=second_score,
        margin_fixed=margin,
        mean_alignment_conf_fixed=mean_alignment_conf,
        used_entry_keys=tuple(sorted(used_entries)),
        path_signature=path_signature,
        candidate_signature=alignment.candidate_signature,
    )
