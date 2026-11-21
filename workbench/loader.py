"""Loading and validation for vocabulary, hierarchy config and JSONL samples."""
from __future__ import annotations

from .errors import (
    E_DOMAIN_UNKNOWN,
    E_INTENT_OUT_OF_DOMAIN,
    E_SAMPLE_INVALID,
    E_SLOT_SCHEMA_INVALID,
)
from .io_utils import read_json, read_jsonl
from .models import ConfigError, HierarchyConfig, Sample, Vocabulary


def load_vocabulary(path: str) -> Vocabulary:
    return Vocabulary.from_dict(read_json(path))


def load_config(path: str, vocab: Vocabulary) -> HierarchyConfig:
    try:
        return HierarchyConfig.from_dict(read_json(path), vocab)
    except ConfigError:
        raise


def _failure(code: str, ref: str, line: int, message: str) -> dict:
    return {"code": code, "ref": ref, "line": line, "message": message}


def validate_sample_row(row: dict, line: int, seen_ids: set,
                        vocab: Vocabulary, config: HierarchyConfig):
    ref = str(row.get("utterance_id", f"line-{line}"))
    required = ("utterance_id", "tokens", "domain", "intent")
    if not all(key in row for key in required):
        return None, [_failure(E_SAMPLE_INVALID, ref, line, "missing required field(s)")]
    utterance_id = row["utterance_id"]
    if not isinstance(utterance_id, str) or not utterance_id:
        return None, [_failure(E_SAMPLE_INVALID, ref, line, "utterance_id must be a non-empty string")]
    if utterance_id in seen_ids:
        return None, [_failure(E_SAMPLE_INVALID, ref, line, "duplicate utterance_id")]
    tokens = row["tokens"]
    if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
        return None, [_failure(E_SAMPLE_INVALID, ref, line, "tokens must be a non-empty string list")]
    illegal = [token for token in tokens if token not in vocab.tokens]
    if illegal:
        return None, [_failure(
            E_SAMPLE_INVALID, ref, line, f"token(s) outside controlled vocabulary: {illegal}")]
    domain = row["domain"]
    if domain not in config.domains:
        return None, [_failure(E_DOMAIN_UNKNOWN, ref, line, f"domain label unknown: {domain}")]
    intent = row["intent"]
    if intent not in config.domain_intents.get(domain, []):
        return None, [_failure(
            E_INTENT_OUT_OF_DOMAIN, ref, line,
            f"intent {intent} not mapped under domain {domain}")]
    gold = row.get("gold_slots", [])
    if not isinstance(gold, list):
        return None, [_failure(E_SAMPLE_INVALID, ref, line, "gold_slots must be a list")]
    failures = []
    norm_gold = []
    for triple in gold:
        if (not isinstance(triple, list) or len(triple) != 3
                or not isinstance(triple[0], str) or not isinstance(triple[1], list)
                or not isinstance(triple[2], int)):
            failures.append(_failure(E_SAMPLE_INVALID, ref, line, f"bad gold triple: {triple}"))
            continue
        slot, value, start = triple
        if slot not in config.slot_defs:
            failures.append(_failure(
                E_SLOT_SCHEMA_INVALID, ref, line, f"gold references undeclared slot: {slot}"))
            continue
        if any(token not in vocab.tokens for token in value):
            failures.append(_failure(
                E_SAMPLE_INVALID, ref, line, f"gold value token outside vocabulary: {value}"))
            continue
        norm_gold.append((slot, tuple(value), start))
    if failures:
        return None, failures
    sample = Sample(
        utterance_id=utterance_id,
        tokens=tuple(tokens),
        domain=domain,
        intent=intent,
        gold_slots=tuple(norm_gold),
        source_line=line,
    )
    return sample, []


def load_samples(path: str, vocab: Vocabulary, config: HierarchyConfig):
    samples = {}
    failures = []
    seen = set()
    for line, row in read_jsonl(path):
        sample, row_failures = validate_sample_row(row, line, seen, vocab, config)
        if row_failures:
            failures.extend(row_failures)
            continue
        seen.add(sample.utterance_id)
        samples[sample.utterance_id] = sample
    return samples, failures
