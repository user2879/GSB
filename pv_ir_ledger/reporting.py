from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .exchange import (
    build_sidecar_document,
    downgrade_v2_to_v1,
    write_v1_text,
    write_v2_document,
)
from .hashing import canonical_json


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def export_judgments(ledger: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    v2_judgments = []
    for component in ledger["components"]:
        judgment = component["exchange"]["judgment"]
        judgment["source_version"] = "v2"
        v2_judgments.append(judgment)
    v2_document = write_v2_document(v2_judgments, ledger.get("quantization_scheme_name"))
    atomic_write_json(output / "judgments_v2.json", v2_document)

    v1_judgments = []
    sidecar_entries = []
    for judgment in v2_judgments:
        v1_judgment, sidecar = downgrade_v2_to_v1(judgment)
        v1_judgments.append(v1_judgment)
        sidecar_entries.append(sidecar)
    atomic_write_text(output / "judgments_v1.txt", write_v1_text(v1_judgments))
    sidecar_document = build_sidecar_document(sidecar_entries)
    atomic_write_json(output / "judgments_v1.sidecar.json", sidecar_document)
    return {"v2": v2_document, "v1_count": len(v1_judgments), "sidecar": sidecar_document}


DOMAIN_LEDGER_KEYS = (
    "config_hash",
    "quantization_scheme_name",
    "image_fingerprints",
    "cells",
    "components",
    "array_conclusions",
    "image_conclusions",
    "boundary_grade_items",
    "failures",
)


def domain_canonical(ledger: dict[str, Any]) -> str:
    projection = {key: ledger.get(key) for key in DOMAIN_LEDGER_KEYS}
    return canonical_json(projection)


def event_order_canonical(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "image": event["image"],
            "component_id": event["component_id"],
            "stage": event["stage"],
            "cache_key": event["cache_key"],
            "event_order": event["event_order"],
        }
        for event in ledger.get("ledger_events", [])
    ]


def action_order_canonical(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "image": event["image"],
            "component_id": event["component_id"],
            "stage": event["stage"],
            "action": event["action"],
            "reason": event["reason"],
            "cache_key": event["cache_key"],
            "event_order": event["event_order"],
        }
        for event in ledger.get("ledger_events", [])
    ]
