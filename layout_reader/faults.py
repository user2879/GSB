"""Fault-injection registry and page-local fault application helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import (
    E_BLOCK_BOUNDS,
    E_COLUMN_GAP_INVALID,
    E_IMAGE_INVALID,
    E_ORDER_DUPLICATE,
    ConfigError,
    WorkbenchError,
)


FAULT_BUSINESS_CODES = {
    "truncate_png_bytes": E_IMAGE_INVALID,
    "truncate_png": E_IMAGE_INVALID,
    "corrupt_png_container": E_IMAGE_INVALID,
    "block_out_of_bounds": E_BLOCK_BOUNDS,
    "column_gap_zero": E_COLUMN_GAP_INVALID,
    "duplicate_reading_order": E_ORDER_DUPLICATE,
}

FAULT_STAGES = {
    "truncate_png_bytes": "image",
    "truncate_png": "image",
    "corrupt_png_container": "image",
    "block_out_of_bounds": "detect",
    "column_gap_zero": "columns",
    "duplicate_reading_order": "order",
}

STAGE_ALIASES = {
    "load": "image",
    "loader": "image",
    "detection": "detect",
    "role": "role",
    "roles": "role",
    "column": "columns",
    "columns": "columns",
    "ordering": "order",
    "order": "order",
    "image": "image",
    "detect": "detect",
}

@dataclass(frozen=True)
class FaultItem:
    item_id: str
    page: str
    stage: str
    kind: str
    parameters: dict[str, Any]
    expected_code: str

    def fingerprint_value(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "page": self.page,
            "stage": self.stage,
            "kind": self.kind,
            "parameters": self.parameters,
        }

    def report_record(self) -> dict[str, Any]:
        return {
            "id": self.item_id,
            "page": self.page,
            "stage": self.stage,
            "kind": self.kind,
            "expected_code": self.expected_code,
            "hit": False,
            "business_code": None,
            "status": "not_evaluated",
        }


class FaultRegistry:
    def __init__(self, items: list[FaultItem]) -> None:
        self.items = sorted(items, key=lambda item: item.item_id)
        self._by_page_stage: dict[tuple[str, str], list[FaultItem]] = {}
        for item in self.items:
            self._by_page_stage.setdefault((item.page, item.stage), []).append(item)

    def validate_pages(self, pages: set[str]) -> None:
        for item in self.items:
            if item.page not in pages:
                # A missing page is reported as an unhit injection rather than
                # making the whole run unusable; page-set edits are common.
                continue

    def for_stage(self, page: str, stage: str) -> list[FaultItem]:
        return list(self._by_page_stage.get((page, stage), []))

    def stage_fingerprint(self, page: str, stage: str) -> list[dict[str, Any]]:
        return [item.fingerprint_value() for item in self.for_stage(page, stage)]

    def truncate_image_bytes(self, page: str, raw_bytes: bytes) -> bytes:
        current = raw_bytes
        for item in self.for_stage(page, "image"):
            if item.kind in {"truncate_png_bytes", "truncate_png"}:
                byte_count = int(item.parameters.get("bytes", 64))
                ratio = item.parameters.get("fraction")
                if ratio is not None:
                    byte_count = max(1, int(len(current) * float(ratio)))
                if byte_count <= 0:
                    raise ConfigError(
                        f"injection {item.item_id} truncation byte count must be positive"
                    )
                current = current[: max(0, len(current) - byte_count)]
            elif item.kind == "corrupt_png_container":
                mutable = bytearray(current)
                offset = int(item.parameters.get("offset", 1))
                if mutable:
                    mutable[min(max(offset, 0), len(mutable) - 1)] ^= 0xFF
                current = bytes(mutable)
        return current


def load_fault_registry(path: str | Path | None) -> FaultRegistry:
    if path is None:
        return FaultRegistry([])
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        raw_items = raw
    elif isinstance(raw, dict):
        raw_items = raw.get("items", [])
    else:
        raise ConfigError("fault-injection JSON must be an array or an object with items")
    if not isinstance(raw_items, list):
        raise ConfigError("fault-injection items must be an array")

    items: list[FaultItem] = []
    seen_ids: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ConfigError("each fault-injection item must be an object")
        item_id = str(raw_item.get("id", "")).strip()
        page = str(raw_item.get("page", "")).strip()
        stage = STAGE_ALIASES.get(str(raw_item.get("stage", "")).strip().lower())
        kind = str(raw_item.get("kind", raw_item.get("fault", "")).strip())
        if not item_id or item_id in seen_ids:
            raise ConfigError("fault-injection ids must be non-empty and unique")
        if not page:
            raise ConfigError(f"injection {item_id} has no page")
        if stage is None:
            raise ConfigError(f"injection {item_id} has unsupported stage")
        if kind not in FAULT_BUSINESS_CODES:
            raise ConfigError(f"injection {item_id} has unsupported fault kind {kind!r}")
        if FAULT_STAGES[kind] != stage:
            raise ConfigError(
                f"injection {item_id} uses {kind!r} with incompatible stage {stage!r}"
            )
        parameters = {
            key: copy.deepcopy(value)
            for key, value in raw_item.items()
            if key not in {"id", "page", "stage", "kind", "fault"}
        }
        items.append(
            FaultItem(
                item_id=item_id,
                page=page,
                stage=stage,
                kind=kind,
                parameters=parameters,
                expected_code=FAULT_BUSINESS_CODES[kind],
            )
        )
        seen_ids.add(item_id)
    return FaultRegistry(items)


def fault_error(
    code: str,
    message: str,
    stage: str,
    items: list[FaultItem],
    details: dict[str, Any] | None = None,
) -> WorkbenchError:
    merged_details = dict(details or {})
    merged_details["injection_ids"] = [item.item_id for item in items]
    return WorkbenchError(
        code,
        message,
        stage,
        injection_id=items[0].item_id if items else None,
        details=merged_details,
    )
