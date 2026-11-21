from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .loader import validate_sentence_payload
from .model import (
    CompileConfig,
    Failure,
    LexiconEntry,
    ScoreParams,
    SourceSentence,
    TranslationState,
    entry_key,
)


@dataclass
class CommandOutcome:
    affected_ids: set[str] = field(default_factory=set)
    changed_entry_keys: set[str] = field(default_factory=set)
    changed_sources: set[str] = field(default_factory=set)
    params_changed: bool = False
    deleted_ids: set[str] = field(default_factory=set)
    failure: Failure | None = None
    no_op: bool = False


def _command_failure(
    command: dict[str, Any], index: int, message: str
) -> Failure:
    command_id = str(command.get("id", command.get("command_id", f"cmd-{index}")))
    return Failure(
        code="E_EDIT_INVALID",
        phase="apply_command",
        message=message,
        file="edits.json",
        line=index + 1,
        command_id=command_id,
        order=(4, index + 1, "E_EDIT_INVALID", command_id),
    )


def _parse_entry_fields(command: dict[str, Any]) -> tuple[str, str] | None:
    source = command.get("source")
    target = command.get("target")
    if isinstance(source, str) and isinstance(target, str) and source and target:
        return source, target
    return None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _batch_counts(sentences: dict[str, SourceSentence]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for sentence in sentences.values():
        counts[sentence.batch] = counts.get(sentence.batch, 0) + 1
    return counts


def _check_batch_limit(
    sentences: dict[str, SourceSentence], config: CompileConfig
) -> str | None:
    for batch, count in sorted(_batch_counts(sentences).items()):
        if count > config.max_sentences_per_batch:
            return (
                f"batch {batch} would contain {count} sentences, exceeding "
                f"max_sentences_per_batch={config.max_sentences_per_batch}"
            )
    return None


def _param_field(name: str) -> str | None:
    aliases = {
        "alignment_weight": "alignment_weight",
        "word_alignment_weight": "alignment_weight",
        "order_weight": "order_weight",
        "word_order_weight": "order_weight",
        "length_penalty": "length_penalty",
        "length_penalty_coefficient": "length_penalty",
        "fusion_temperature": "fusion_temperature",
        "temperature": "fusion_temperature",
        "confidence_threshold": "confidence_threshold",
        "threshold": "confidence_threshold",
        "fixed_point_bits": "fixed_point_bits",
        "fixed_bits": "fixed_point_bits",
        "quantization_tier": "quantization_tier",
        "quant_tier": "quantization_tier",
    }
    return aliases.get(name)


def apply_command(
    state: TranslationState,
    command: dict[str, Any],
    index: int,
    config: CompileConfig,
) -> CommandOutcome:
    outcome = CommandOutcome()
    op = str(command.get("op", command.get("operation", command.get("type", "")))).lower()
    if state.params is None:
        outcome.failure = _command_failure(command, index, "state has no scoring parameters")
        return outcome

    try:
        if op in {"lex_add", "lexicon_add", "add_lexicon", "lexicon_upsert"}:
            pair = _parse_entry_fields(command)
            if pair is None:
                raise ValueError("lex_add requires non-empty source and target")
            source, target = pair
            key = entry_key(source, target)
            if key in state.lexicon:
                raise ValueError("lexicon entry already exists; use an update command")
            frequency = _finite_number(command.get("frequency", command.get("freq")))
            confidence = _finite_number(
                command.get("alignment_confidence", command.get("confidence"))
            )
            if frequency is None or frequency <= 0:
                raise ValueError("frequency must be a positive finite number")
            if confidence is None or not 0 <= confidence <= 1:
                raise ValueError("alignment_confidence must be finite and in [0, 1]")
            state.lexicon[key] = LexiconEntry(
                source=source,
                target=target,
                frequency=frequency,
                alignment_confidence=confidence,
            )
            outcome.changed_entry_keys.add(key)
            outcome.changed_sources.add(source)
            outcome.affected_ids.update(
                sentence_id
                for sentence_id, sentence in state.sentences.items()
                if source in sentence.tokens
            )
            return outcome

        if op in {"lex_delete", "lexicon_delete", "remove_lexicon"}:
            pair = _parse_entry_fields(command)
            if pair is None:
                raise ValueError("lex_delete requires source and target")
            source, target = pair
            key = entry_key(source, target)
            if key not in state.lexicon:
                raise ValueError("lexicon entry does not exist")
            del state.lexicon[key]
            outcome.changed_entry_keys.add(key)
            outcome.changed_sources.add(source)
            outcome.affected_ids.update(
                sentence_id
                for sentence_id, sentence in state.sentences.items()
                if source in sentence.tokens
            )
            return outcome

        if op in {
            "lex_freq",
            "lex_frequency",
            "lex_set_frequency",
            "set_lexicon_frequency",
            "adjust_frequency",
            "lexicon_set_frequency",
        }:
            pair = _parse_entry_fields(command)
            if pair is None:
                raise ValueError("frequency command requires source and target")
            source, target = pair
            key = entry_key(source, target)
            entry = state.lexicon.get(key)
            if entry is None:
                raise ValueError("lexicon entry does not exist")
            if "delta" in command:
                delta = _finite_number(command["delta"])
                if delta is None:
                    raise ValueError("frequency delta must be finite")
                frequency = entry.frequency + delta
            else:
                raw = _finite_number(command.get("value", command.get("frequency")))
                if raw is None:
                    raise ValueError("frequency value must be finite")
                frequency = raw
            if not math.isfinite(frequency) or frequency <= 0:
                raise ValueError("frequency must remain positive and finite")
            state.lexicon[key] = LexiconEntry(
                source=entry.source,
                target=entry.target,
                frequency=frequency,
                alignment_confidence=entry.alignment_confidence,
            )
            outcome.changed_entry_keys.add(key)
            outcome.changed_sources.add(source)
            outcome.affected_ids.update(
                sentence_id
                for sentence_id, sentence in state.sentences.items()
                if source in sentence.tokens
            )
            return outcome

        if op in {
            "lex_conf",
            "lex_confidence",
            "lex_set_confidence",
            "set_lexicon_confidence",
            "adjust_confidence",
            "lexicon_set_confidence",
        }:
            pair = _parse_entry_fields(command)
            if pair is None:
                raise ValueError("confidence command requires source and target")
            source, target = pair
            key = entry_key(source, target)
            entry = state.lexicon.get(key)
            if entry is None:
                raise ValueError("lexicon entry does not exist")
            if "delta" in command:
                delta = _finite_number(command["delta"])
                if delta is None:
                    raise ValueError("confidence delta must be finite")
                confidence = entry.alignment_confidence + delta
            else:
                raw = _finite_number(command.get("value", command.get("alignment_confidence")))
                if raw is None:
                    raise ValueError("confidence value must be finite")
                confidence = raw
            if not 0 <= confidence <= 1:
                raise ValueError("alignment_confidence must remain in [0, 1]")
            state.lexicon[key] = LexiconEntry(
                source=entry.source,
                target=entry.target,
                frequency=entry.frequency,
                alignment_confidence=confidence,
            )
            outcome.changed_entry_keys.add(key)
            outcome.changed_sources.add(source)
            outcome.affected_ids.update(
                sentence_id
                for sentence_id, sentence in state.sentences.items()
                if source in sentence.tokens
            )
            return outcome

        if op in {"param_set", "set_param", "set_parameter"}:
            field_name = str(command.get("field", command.get("name", "")))
            canonical = _param_field(field_name)
            if canonical is None:
                raise ValueError(f"unknown scoring parameter: {field_name}")
            if "value" not in command:
                raise ValueError("param_set requires value")
            raw = command["value"]
            if canonical in {"fixed_point_bits", "quantization_tier"}:
                if isinstance(raw, bool):
                    raise ValueError("integer parameter cannot be boolean")
                value: float | int = int(raw)
            else:
                numeric = _finite_number(raw)
                if numeric is None:
                    raise ValueError("scoring parameter value must be finite")
                value = numeric
            params = state.params.with_field(canonical, value)
            errors = params.validate()
            if errors:
                raise ValueError("; ".join(errors))
            state.params = params
            outcome.params_changed = True
            outcome.affected_ids.update(state.sentences)
            return outcome

        if op in {"sent_replace", "replace_sentence", "sentence_replace"}:
            payload = command.get("sentence", command.get("value"))
            if payload is None and "id" in command and "tokens" in command:
                payload = command
            if not isinstance(payload, dict):
                raise ValueError("sentence_replace requires a sentence object")
            target_id = payload.get("id", payload.get("sentence_id"))
            if not isinstance(target_id, str):
                raise ValueError("replacement sentence requires an id")
            existing_without = {
                sentence_id: sentence
                for sentence_id, sentence in state.sentences.items()
                if sentence_id != target_id
            }
            sentence, error, details = validate_sentence_payload(
                payload, config, existing_ids=existing_without
            )
            if error or sentence is None:
                raise ValueError(error or "invalid replacement sentence")
            trial = dict(existing_without)
            trial[sentence.sentence_id] = sentence
            batch_error = _check_batch_limit(trial, config)
            if batch_error:
                raise ValueError(batch_error)
            state.sentences = trial
            outcome.affected_ids.add(sentence.sentence_id)
            return outcome

        if op in {"sent_delete", "delete_sentence", "sentence_delete"}:
            sentence_id = command.get("sentence_id", command.get("target_id"))
            if sentence_id is None:
                legacy_id = command.get("id")
                if isinstance(legacy_id, str) and legacy_id in state.sentences:
                    sentence_id = legacy_id
            if not isinstance(sentence_id, str) or sentence_id not in state.sentences:
                raise ValueError("sentence_id does not exist; sent_delete requires sentence_id")
            del state.sentences[sentence_id]
            outcome.deleted_ids.add(sentence_id)
            outcome.affected_ids.add(sentence_id)
            return outcome

        if op in {"batch_append", "append_batch", "sent_append"}:
            payloads = command.get("sentences", command.get("items", []))
            if not isinstance(payloads, list) or not payloads:
                raise ValueError("batch_append requires a non-empty sentences list")
            trial = dict(state.sentences)
            new_ids: set[str] = set()
            for payload in payloads:
                sentence, error, details = validate_sentence_payload(
                    payload, config, existing_ids=trial
                )
                if error or sentence is None:
                    raise ValueError(error or "invalid appended sentence")
                trial[sentence.sentence_id] = sentence
                new_ids.add(sentence.sentence_id)
            batch_error = _check_batch_limit(trial, config)
            if batch_error:
                raise ValueError(batch_error)
            state.sentences = trial
            outcome.affected_ids.update(new_ids)
            return outcome

        if op in {"noop", "no_op"}:
            outcome.no_op = True
            return outcome

        raise ValueError(f"unsupported command op: {op or '<missing>'}")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        outcome.failure = _command_failure(command, index, str(exc))
        return outcome


def parse_score_value(value: Any) -> ScoreParams | None:
    try:
        return ScoreParams.from_mapping(value)
    except (KeyError, TypeError, ValueError):
        return None
