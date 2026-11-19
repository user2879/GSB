"""Deterministic fault injection independent from sample and calibration inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import E_CONFIG_INVALID
from .errors import VoicecalError
from .util import json_hash, read_json

SUPPORTED_STAGES = {"extract", "initial", "calibration", "attribution"}
SUPPORTED_KINDS = {"corrupt_format", "force_exception"}


def empty_injections() -> dict[str, Any]:
    clean = {"schema_version": 1, "injections": []}
    clean["injection_hash"] = json_hash(clean)
    return clean


def validate_injections(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or int(raw.get("schema_version", -1)) != 1:
        raise VoicecalError(E_CONFIG_INVALID, "unsupported injection configuration schema")
    items_raw = raw.get("injections")
    if not isinstance(items_raw, list):
        raise VoicecalError(E_CONFIG_INVALID, "injections must be an array")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item_raw in items_raw:
        if not isinstance(item_raw, dict):
            raise VoicecalError(E_CONFIG_INVALID, "injection item must be an object")
        injection_id = item_raw.get("injection_id")
        turn_id = item_raw.get("turn_id")
        stage = item_raw.get("stage")
        kind = item_raw.get("kind")
        if not isinstance(injection_id, str) or not injection_id:
            raise VoicecalError(E_CONFIG_INVALID, "injection_id must be a non-empty string")
        if injection_id in seen:
            raise VoicecalError(E_CONFIG_INVALID, "injection_id must be unique", injection_id=injection_id)
        if not isinstance(turn_id, str) or not turn_id:
            raise VoicecalError(E_CONFIG_INVALID, "injection turn_id must be a non-empty string", injection_id=injection_id)
        if stage not in SUPPORTED_STAGES:
            raise VoicecalError(E_CONFIG_INVALID, "injection stage is unsupported", injection_id=injection_id, stage=stage)
        if kind not in SUPPORTED_KINDS:
            raise VoicecalError(E_CONFIG_INVALID, "injection kind is unsupported", injection_id=injection_id, kind=kind)
        if kind == "corrupt_format" and stage != "extract":
            raise VoicecalError(E_CONFIG_INVALID, "corrupt_format is only valid at extract stage", injection_id=injection_id)
        seen.add(injection_id)
        items.append(
            {
                "injection_id": injection_id,
                "turn_id": turn_id,
                "stage": stage,
                "kind": kind,
                "enabled": bool(item_raw.get("enabled", True)),
                "message": str(item_raw.get("message", f"injected {kind}")),
            }
        )
    items.sort(key=lambda item: (item["turn_id"], item["stage"], item["injection_id"]))
    clean = {"schema_version": 1, "injections": items}
    clean["injection_hash"] = json_hash(clean)
    return clean


def load_injections(path: Path | None) -> dict[str, Any]:
    if path is None:
        return empty_injections()
    try:
        raw = read_json(path)
    except FileNotFoundError as exc:
        raise VoicecalError(E_CONFIG_INVALID, "injection file is missing", file=str(path)) from exc
    except ValueError as exc:
        raise VoicecalError(E_CONFIG_INVALID, "injection file is not valid JSON", file=str(path)) from exc
    return validate_injections(raw)


def injections_by_turn_stage(config: dict[str, Any], turn_id: str, stage: str) -> list[dict[str, Any]]:
    return [
        item
        for item in config["injections"]
        if item["enabled"] and item["turn_id"] == turn_id and item["stage"] == stage
    ]
