"""Isolated fault-injection configuration.

Injections are evaluated per keypoint, not by mutating the shared camera files.
That is what lets one deliberately broken keypoint fail while all other
keypoints continue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import (
    ConfigError,
    E_CFG_INVALID,
    E_CAMERA_INVALID,
    E_OBS_INVALID,
)
from .hashing import sha256_json


FAULT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "nonfinite_pixel": {
        "stages": {"load", "loading"},
        "business_code": E_OBS_INVALID,
        "description": "Replace a target observation coordinate with NaN before validation.",
    },
    "nonorthogonal_rotation": {
        "stages": {"load", "loading"},
        "business_code": E_CAMERA_INVALID,
        "description": "Skew a target camera rotation and validate it for the target keypoint.",
    },
    "min_disparity_above_limit": {
        "stages": {"triangulation", "triangulate"},
        "business_code": E_CFG_INVALID,
        "description": "Raise the point-local minimum disparity beyond the legal upper bound.",
    },
    "zero_outlier_threshold": {
        "stages": {"rejection", "outlier_rejection"},
        "business_code": E_CFG_INVALID,
        "description": "Set the point-local outlier threshold to zero.",
    },
}


@dataclass(frozen=True)
class InjectionItem:
    id: str
    keypoint_id: str
    stage: str
    fault: str
    business_code: str
    value: Any = None
    view_index: int | None = None
    coordinate_field: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "id": self.id,
                "keypoint_id": self.keypoint_id,
                "stage": self.stage,
                "fault": self.fault,
                "value": self.value,
                "view_index": self.view_index,
                "field": self.coordinate_field,
            }
        )

    def report_record(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "keypoint_id": self.keypoint_id,
            "stage": self.stage,
            "fault": self.fault,
            "business_code": self.business_code,
            "hit": False,
        }
        if self.view_index is not None:
            out["view_index"] = self.view_index
        if self.coordinate_field is not None:
            out["field"] = self.coordinate_field
        if self.value is not None:
            out["value"] = self.value
        return out


@dataclass
class InjectionSet:
    version: str = "none"
    items: list[InjectionItem] = field(default_factory=list)
    load_errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return sha256_json(
            {
                "version": self.version,
                "items": [item.fingerprint for item in self.items],
            }
        )

    def for_keypoint(self, keypoint_id: str) -> list[InjectionItem]:
        return [
            item
            for item in self.items
            if item.keypoint_id == keypoint_id
        ]

    def fingerprint_for(self, keypoint_id: str) -> str:
        return sha256_json(
            [item.fingerprint for item in self.for_keypoint(keypoint_id)]
        )

    def initial_report(self) -> list[dict[str, Any]]:
        return [item.report_record() for item in sorted(self.items, key=lambda item: item.id)]


def parse_injections(data: dict[str, Any]) -> InjectionSet:
    version = str(data.get("injection_version", data.get("version", "unnamed")))
    raw_items = data.get("items", data.get("injections", []))
    if not isinstance(raw_items, list):
        raise ConfigError("injection items must be a list")
    items: list[InjectionItem] = []
    seen: set[str] = set()
    allowed_stage_aliases = {
        "load": "load",
        "loading": "load",
        "triangulation": "triangulation",
        "triangulate": "triangulation",
        "rejection": "rejection",
        "outlier_rejection": "rejection",
        "finalization": "finalization",
        "finalize": "finalization",
    }
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ConfigError(f"injection item {index} must be an object")
        item_id = str(raw.get("id", "")).strip()
        if not item_id or item_id in seen:
            raise ConfigError(f"injection item {index} has an empty or duplicate id")
        seen.add(item_id)
        keypoint_id = str(raw.get("keypoint_id", "")).strip()
        if not keypoint_id:
            raise ConfigError(f"injection {item_id} must specify keypoint_id")
        fault = str(raw.get("fault", ""))
        if fault not in FAULT_DEFINITIONS:
            raise ConfigError(f"injection {item_id} has unknown fault {fault!r}")
        stage = allowed_stage_aliases.get(str(raw.get("stage", "")).strip())
        if stage is None or stage not in {
            "load",
            "triangulation",
            "rejection",
            "finalization",
        }:
            raise ConfigError(f"injection {item_id} has invalid stage")
        expected_stage = next(iter(FAULT_DEFINITIONS[fault]["stages"]))
        expected_stage = allowed_stage_aliases.get(expected_stage, expected_stage)
        if stage != expected_stage:
            raise ConfigError(
                f"injection {item_id} fault {fault} belongs to stage {expected_stage}"
            )
        business_code = str(
            raw.get("business_code", FAULT_DEFINITIONS[fault]["business_code"])
        )
        view_index = raw.get("view_index")
        if view_index is not None:
            if isinstance(view_index, bool) or not isinstance(view_index, int):
                raise ConfigError(f"injection {item_id} view_index must be an integer")
        field_name = raw.get("field")
        if field_name is not None and field_name not in {"u", "v"}:
            raise ConfigError(f"injection {item_id} field must be 'u' or 'v'")
        items.append(
            InjectionItem(
                id=item_id,
                keypoint_id=keypoint_id,
                stage=stage,
                fault=fault,
                business_code=business_code,
                value=raw.get("value"),
                view_index=view_index,
                coordinate_field=field_name,
                raw=dict(raw),
            )
        )
    items.sort(key=lambda item: item.id)
    return InjectionSet(version=version, items=items)


def load_injections(path: str | Path | None) -> InjectionSet:
    if path is None:
        return InjectionSet()
    injection_path = Path(path)
    if not injection_path.exists():
        raise ConfigError(f"injection config does not exist: {injection_path}")
    try:
        data = json.loads(injection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read injection config {injection_path}: {exc}") from exc
    return parse_injections(data)
