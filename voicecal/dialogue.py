"""Dialogue script loading, validation, and stable ordering."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .constants import DIALOGUE_SCHEMA_VERSION, E_DIALOGUE_INVALID
from .errors import VoicecalError
from .util import read_json


def dialogue_error(message: str, **details: Any) -> VoicecalError:
    return VoicecalError(E_DIALOGUE_INVALID, message, **details)


def normalize_turn(turn: dict[str, Any], source: str = "dialogue") -> dict[str, Any]:
    if not isinstance(turn, dict):
        raise dialogue_error("turn must be an object", source=source)
    required = ("turn_id", "turn_index", "audio", "speaker_id")
    missing = [field for field in required if field not in turn]
    if missing:
        raise dialogue_error("turn is missing required fields", source=source, missing=missing)
    turn_id = turn["turn_id"]
    if not isinstance(turn_id, str) or not turn_id:
        raise dialogue_error("turn_id must be a non-empty string", source=source)
    if isinstance(turn["turn_index"], bool) or not isinstance(turn["turn_index"], int) or turn["turn_index"] < 1:
        raise dialogue_error("turn_index must be a positive integer", turn_id=turn_id)
    audio = turn["audio"]
    if not isinstance(audio, str) or not audio:
        raise dialogue_error("audio must be a non-empty relative path", turn_id=turn_id)
    speaker_id = turn["speaker_id"]
    if not isinstance(speaker_id, str) or not speaker_id:
        raise dialogue_error("speaker_id must be a non-empty string", turn_id=turn_id)
    candidates_raw = turn.get("speaker_candidates", [speaker_id])
    if not isinstance(candidates_raw, list) or not candidates_raw or not all(isinstance(item, str) and item for item in candidates_raw):
        raise dialogue_error("speaker_candidates must be a non-empty string array", turn_id=turn_id)
    candidates = sorted(set(candidates_raw) | {speaker_id})
    gap_ms = turn.get("gap_ms", 0)
    if isinstance(gap_ms, bool) or not isinstance(gap_ms, (int, float)) or gap_ms < 0:
        raise dialogue_error("gap_ms must be a non-negative number", turn_id=turn_id)
    previous_turn_id = turn.get("previous_turn_id")
    if previous_turn_id is not None and not isinstance(previous_turn_id, str):
        raise dialogue_error("previous_turn_id must be null or a string", turn_id=turn_id)
    normalized = copy.deepcopy(turn)
    normalized.update(
        {
            "turn_id": turn_id,
            "turn_index": turn["turn_index"],
            "conversation_id": str(turn.get("conversation_id", "default")),
            "audio": audio,
            "speaker_id": speaker_id,
            "speaker_candidates": candidates,
            "gap_ms": float(gap_ms),
            "previous_turn_id": previous_turn_id,
            "link_type": str(turn.get("link_type", "continuation")),
        }
    )
    return normalized


def validate_dialogue(raw: dict[str, Any], source: str = "dialogue") -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise dialogue_error("dialogue root must be an object", source=source)
    if int(raw.get("schema_version", -1)) != DIALOGUE_SCHEMA_VERSION:
        raise dialogue_error("unsupported dialogue schema version", expected=DIALOGUE_SCHEMA_VERSION)
    turns_raw = raw.get("turns")
    if not isinstance(turns_raw, list) or not turns_raw:
        raise dialogue_error("dialogue.turns must be a non-empty array")
    turns = [normalize_turn(turn, source) for turn in turns_raw]
    turns.sort(key=lambda item: item["turn_index"])
    ids = [turn["turn_id"] for turn in turns]
    indexes = [turn["turn_index"] for turn in turns]
    if len(set(ids)) != len(ids):
        raise dialogue_error("turn_id values must be unique")
    if len(set(indexes)) != len(indexes):
        raise dialogue_error("turn_index values must be globally unique")
    known_ids = set(ids)
    previous_in_conversation: dict[str, str | None] = {}
    for turn in turns:
        previous = turn["previous_turn_id"]
        conversation = turn["conversation_id"]
        if previous is not None and previous not in known_ids:
            raise dialogue_error("previous_turn_id is unknown", turn_id=turn["turn_id"], previous_turn_id=previous)
        expected_previous = previous_in_conversation.get(conversation)
        if previous is not None and previous != expected_previous:
            raise dialogue_error(
                "previous_turn_id does not match prior turn in the same conversation",
                turn_id=turn["turn_id"],
                expected=expected_previous,
                actual=previous,
            )
        previous_in_conversation[conversation] = turn["turn_id"]
    by_conversation: dict[str, int] = {}
    for turn in turns:
        conversation = turn["conversation_id"]
        by_conversation[conversation] = by_conversation.get(conversation, 0) + 1
        turn["conversation_position"] = by_conversation[conversation]
    return {
        "schema_version": DIALOGUE_SCHEMA_VERSION,
        "dialogue_id": str(raw.get("dialogue_id", source)),
        "turns": turns,
    }


def load_dialogue(path: Path) -> dict[str, Any]:
    try:
        raw = read_json(path)
    except FileNotFoundError as exc:
        raise dialogue_error("dialogue file is missing", file=str(path)) from exc
    except ValueError as exc:
        raise dialogue_error("dialogue file is not valid JSON", file=str(path)) from exc
    return validate_dialogue(raw, str(path))
