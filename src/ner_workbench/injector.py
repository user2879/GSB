from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import sha256_file
from .errors import E_CONFIG_INVALID, WorkbenchError


ALLOWED_INJECTIONS: dict[str, str] = {
    "nonfinite_score": "emission",
    "truncated_labels": "decode",
    "hash_bucket_mismatch": "emission",
    "quota_lt_min_chars": "schedule",
}


@dataclass(frozen=True)
class InjectionSpec:
    injection_id: str
    sample_id: str
    stage: str
    kind: str

    def as_dict(self) -> dict[str, str]:
        return {
            "injection_id": self.injection_id,
            "sample_id": self.sample_id,
            "stage": self.stage,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class InjectionSet:
    by_sample: dict[str, tuple[InjectionSpec, ...]]
    all: tuple[InjectionSpec, ...]
    source_hash: str

    def for_sample(self, sample_id: str, stage: str | None = None) -> tuple[InjectionSpec, ...]:
        items = self.by_sample.get(sample_id, tuple())
        if stage is None:
            return items
        return tuple(item for item in items if item.stage == stage)


def load_injections(path: Path) -> InjectionSet:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WorkbenchError(E_CONFIG_INVALID, "load", f"cannot read injections: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkbenchError(E_CONFIG_INVALID, "load", f"invalid injections JSON: {exc}") from exc
    if isinstance(value, dict):
        raw_items = value.get("injections", [])
    else:
        raw_items = value
    if not isinstance(raw_items, list):
        raise WorkbenchError(E_CONFIG_INVALID, "load", "injections must be a list or {'injections': [...]}")

    seen_ids: set[str] = set()
    items: list[InjectionSpec] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise WorkbenchError(E_CONFIG_INVALID, "load", f"injection #{index} must be an object")
        injection_id = str(raw.get("injection_id", "")).strip()
        sample_id = str(raw.get("sample_id", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        stage = str(raw.get("stage", "")).strip() or ALLOWED_INJECTIONS.get(kind, "")
        if not injection_id or not sample_id or not kind:
            raise WorkbenchError(E_CONFIG_INVALID, "load", f"injection #{index} missing required fields")
        if injection_id in seen_ids:
            raise WorkbenchError(E_CONFIG_INVALID, "load", f"duplicate injection_id: {injection_id}")
        seen_ids.add(injection_id)
        if kind not in ALLOWED_INJECTIONS:
            raise WorkbenchError(E_CONFIG_INVALID, "load", f"unsupported injection kind: {kind}")
        expected_stage = ALLOWED_INJECTIONS[kind]
        if stage != expected_stage:
            raise WorkbenchError(
                E_CONFIG_INVALID,
                "load",
                f"injection {injection_id} kind {kind} must target stage {expected_stage}, got {stage}",
            )
        items.append(InjectionSpec(injection_id, sample_id, stage, kind))

    items.sort(key=lambda item: (item.sample_id, item.stage, item.injection_id))
    grouping: dict[str, list[InjectionSpec]] = {}
    for item in items:
        grouping.setdefault(item.sample_id, []).append(item)
    by_sample = {
        sample_id: tuple(sorted(group, key=lambda item: (item.stage, item.injection_id)))
        for sample_id, group in grouping.items()
    }
    return InjectionSet(by_sample=by_sample, all=tuple(items), source_hash=sha256_file(path))


def injection_hit(
    injection: InjectionSpec,
    code: str,
    detail: str,
    *,
    position: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "injection_id": injection.injection_id,
        "sample_id": injection.sample_id,
        "stage": injection.stage,
        "kind": injection.kind,
        "code": code,
        "detail": detail,
    }
    if position is not None:
        payload["position"] = position
    return payload
