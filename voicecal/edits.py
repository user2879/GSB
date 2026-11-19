"""Command-stream application with deterministic, explicit effective state."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .constants import EDITS_SCHEMA_VERSION, E_EDIT_INVALID
from .dialogue import normalize_turn, validate_dialogue
from .errors import VoicecalError
from .util import read_json


def edit_error(message: str, **details: Any) -> VoicecalError:
    return VoicecalError(E_EDIT_INVALID, message, **details)


def load_edit_stream(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"schema_version": EDITS_SCHEMA_VERSION, "batches": []}
    try:
        raw = read_json(path)
    except FileNotFoundError as exc:
        raise edit_error("edit stream file is missing", file=str(path)) from exc
    except ValueError as exc:
        raise edit_error("edit stream is not valid JSON", file=str(path)) from exc
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != EDITS_SCHEMA_VERSION:
        raise edit_error("unsupported edit stream schema version", expected=EDITS_SCHEMA_VERSION)
    if "batches" in raw:
        batches_raw = raw["batches"]
    elif "commands" in raw:
        batches_raw = [{"batch_id": "batch-1", "commands": raw["commands"]}]
    else:
        raise edit_error("edit stream must contain batches or commands")
    if not isinstance(batches_raw, list):
        raise edit_error("edit stream batches must be an array")
    batches: list[dict[str, Any]] = []
    for index, batch_raw in enumerate(batches_raw, start=1):
        if not isinstance(batch_raw, dict):
            raise edit_error("edit batch must be an object", batch_index=index)
        commands = batch_raw.get("commands")
        if not isinstance(commands, list):
            raise edit_error("edit batch commands must be an array", batch_index=index)
        batches.append({"batch_id": str(batch_raw.get("batch_id", f"batch-{index}")), "commands": copy.deepcopy(commands)})
    return {"schema_version": EDITS_SCHEMA_VERSION, "batches": batches}


def initial_state(dialogue: dict[str, Any]) -> dict[str, Any]:
    return {
        "dialogue": copy.deepcopy(dialogue),
        "config_overrides": {"gain": None, "turn_gain": {}},
        "applied_batch_count": 0,
        "applied_batches": [],
    }


def apply_batch(state: dict[str, Any], batch: dict[str, Any], batch_index: int) -> dict[str, Any]:
    effective = copy.deepcopy(state)
    turns = effective["dialogue"]["turns"]
    by_id = {turn["turn_id"]: turn for turn in turns}
    events: list[dict[str, Any]] = []
    for command_index, command in enumerate(batch["commands"], start=1):
        if not isinstance(command, dict) or not isinstance(command.get("op"), str):
            raise edit_error("edit command must contain an op string", batch_index=batch_index, command_index=command_index)
        op = command["op"]
        if op == "replace_audio":
            turn_id = command.get("turn_id")
            if turn_id not in by_id:
                raise edit_error("replace_audio references unknown turn", turn_id=turn_id)
            audio = command.get("audio")
            if not isinstance(audio, str) or not audio:
                raise edit_error("replace_audio.audio must be a non-empty relative path", turn_id=turn_id)
            old = by_id[turn_id]["audio"]
            by_id[turn_id]["audio"] = audio
            events.append({"op": op, "turn_id": turn_id, "turn_index": by_id[turn_id]["turn_index"], "from": old, "to": audio})
        elif op == "delete_turn":
            turn_id = command.get("turn_id")
            if turn_id not in by_id:
                raise edit_error("delete_turn references unknown turn", turn_id=turn_id)
            removed = by_id[turn_id]
            remaining = [turn for turn in turns if turn["turn_id"] != turn_id]
            deleted_position = removed["conversation_position"]
            for candidate in remaining:
                if (
                    candidate["conversation_id"] == removed["conversation_id"]
                    and candidate["previous_turn_id"] == turn_id
                ):
                    candidate["previous_turn_id"] = removed.get("previous_turn_id")
                    break
            for candidate in remaining:
                if candidate["conversation_id"] == removed["conversation_id"] and candidate["conversation_position"] > deleted_position:
                    candidate["conversation_position"] -= 1
            effective["dialogue"] = validate_dialogue(
                {"schema_version": effective["dialogue"]["schema_version"], "dialogue_id": effective["dialogue"]["dialogue_id"], "turns": remaining},
                f"after delete at batch {batch_index}",
            )
            turns = effective["dialogue"]["turns"]
            by_id = {turn["turn_id"]: turn for turn in turns}
            events.append({"op": op, "turn_id": turn_id, "turn_index": removed["turn_index"], "speaker_id": removed["speaker_id"]})
        elif op == "adjust_gain":
            gain = command.get("gain")
            if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not 0 <= float(gain) <= 1:
                raise edit_error("adjust_gain.gain must be in [0, 1]", batch_index=batch_index, command_index=command_index)
            turn_id = command.get("turn_id")
            if turn_id is None:
                old = effective["config_overrides"].get("gain")
                effective["config_overrides"]["gain"] = float(gain)
                events.append({"op": op, "scope": "global", "from": old, "to": float(gain)})
            else:
                if turn_id not in by_id:
                    raise edit_error("adjust_gain references unknown turn", turn_id=turn_id)
                old = effective["config_overrides"]["turn_gain"].get(turn_id)
                effective["config_overrides"]["turn_gain"][turn_id] = float(gain)
                events.append({"op": op, "scope": "turn", "turn_id": turn_id, "turn_index": by_id[turn_id]["turn_index"], "from": old, "to": float(gain)})
        elif op == "modify_speaker":
            turn_id = command.get("turn_id")
            if turn_id not in by_id:
                raise edit_error("modify_speaker references unknown turn", turn_id=turn_id)
            speaker_id = command.get("speaker_id")
            if not isinstance(speaker_id, str) or not speaker_id:
                raise edit_error("modify_speaker.speaker_id must be a non-empty string", turn_id=turn_id)
            candidates_raw = command.get("speaker_candidates")
            if candidates_raw is None:
                candidates = sorted(set(by_id[turn_id].get("speaker_candidates", [])) | {speaker_id})
            else:
                if not isinstance(candidates_raw, list) or not all(isinstance(item, str) and item for item in candidates_raw):
                    raise edit_error("modify_speaker.speaker_candidates must be a string array", turn_id=turn_id)
                candidates = sorted(set(candidates_raw) | {speaker_id})
            old = by_id[turn_id]["speaker_id"]
            by_id[turn_id]["speaker_id"] = speaker_id
            by_id[turn_id]["speaker_candidates"] = candidates
            events.append({"op": op, "turn_id": turn_id, "turn_index": by_id[turn_id]["turn_index"], "from": old, "to": speaker_id})
        elif op == "append_turn":
            turn_raw = command.get("turn")
            turn = normalize_turn(turn_raw, f"append_turn at batch {batch_index}")
            if turn["turn_id"] in by_id or any(item["turn_index"] == turn["turn_index"] for item in turns):
                raise edit_error("append_turn duplicates an existing id or index", turn_id=turn["turn_id"], turn_index=turn["turn_index"])
            combined = validate_dialogue(
                {
                    "schema_version": effective["dialogue"]["schema_version"],
                    "dialogue_id": effective["dialogue"]["dialogue_id"],
                    "turns": turns + [turn],
                },
                f"after append at batch {batch_index}",
            )
            effective["dialogue"] = combined
            turns = effective["dialogue"]["turns"]
            by_id = {item["turn_id"]: item for item in turns}
            events.append({"op": op, "turn_id": turn["turn_id"], "turn_index": turn["turn_index"]})
        else:
            raise edit_error("unknown edit command", op=op, batch_index=batch_index, command_index=command_index)
    effective["dialogue"] = validate_dialogue(
        {
            "schema_version": effective["dialogue"]["schema_version"],
            "dialogue_id": effective["dialogue"]["dialogue_id"],
            "turns": effective["dialogue"]["turns"],
        },
        f"after batch {batch_index}",
    )
    effective["applied_batch_count"] = int(state.get("applied_batch_count", 0)) + 1
    effective["applied_batches"] = list(state.get("applied_batches", [])) + [
        {"batch_id": batch["batch_id"], "batch_index": batch_index, "events": events}
    ]
    return effective
