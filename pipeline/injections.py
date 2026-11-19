from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import sha256_json
from .errors import ConfigError


ALLOWED_INJECTIONS = {
    "retrieval_scoring": {"nonfinite_confidence", "translation_term_missing", "recall_limit_zero"},
    "evaluation": {"relevance_grade_out_of_range"},
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    query_id: str
    stage: str
    kind: str
    enabled: bool = True
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "injection_id": self.injection_id,
            "query_id": self.query_id,
            "stage": self.stage,
            "kind": self.kind,
            "enabled": self.enabled,
            "params": self.params,
        }


def _validate(injection: Injection) -> None:
    if not injection.injection_id:
        raise ConfigError("injection_id must be non-empty")
    if not injection.query_id:
        raise ConfigError("injection query_id must be non-empty")
    if injection.stage not in ALLOWED_INJECTIONS:
        raise ConfigError(f"unsupported injection stage: {injection.stage}")
    if injection.kind not in ALLOWED_INJECTIONS[injection.stage]:
        raise ConfigError(f"injection kind {injection.kind!r} is not valid for stage {injection.stage}")
    if not isinstance(injection.params, dict):
        raise ConfigError("injection params must be an object")


def load_injections(path: str | Path | None) -> tuple[list[Injection], str]:
    if path is None:
        return [], sha256_json([])
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load injections: {exc}") from exc
    if isinstance(payload, dict):
        raw_items = payload.get("injections", [])
    else:
        raw_items = payload
    if not isinstance(raw_items, list):
        raise ConfigError("injections must be a JSON list or contain an injections list")
    result: list[Injection] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ConfigError(f"injections[{index}] must be an object")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"injections[{index}].enabled must be boolean")
        injection = Injection(
            injection_id=str(raw.get("injection_id", raw.get("id", ""))),
            query_id=str(raw.get("query_id", "")),
            stage=str(raw.get("stage", "")),
            kind=str(raw.get("kind", "")),
            enabled=enabled,
            params=raw.get("params", {}),
        )
        _validate(injection)
        if injection.injection_id in seen:
            raise ConfigError(f"duplicate injection_id: {injection.injection_id}")
        seen.add(injection.injection_id)
        result.append(injection)
    result.sort(key=lambda item: (item.query_id, item.stage, item.injection_id))
    return result, sha256_json([item.to_dict() for item in result])


def injections_for(injections: Iterable[Injection], query_id: str, stage: str) -> list[Injection]:
    return [item for item in injections if item.enabled and item.query_id == query_id and item.stage == stage]

