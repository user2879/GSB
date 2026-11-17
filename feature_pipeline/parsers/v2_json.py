"""v2 JSON exchange format with named precision labels.

Schema::

    {
      "format_version": "v2",
      "entities": {
        "<entity_id>": {
          "<feature>": {"value": 1.234, "precision": "float64"},
          "<feature>": {"value": null, "precision": "float32",
                        "explicit_null": true}
        }
      }
    }

Missing values have two representations: an absent feature key and an
explicit null entry. Both are normalized to ``Cell(missing=True)`` while the
original representation is retained through ``raw_text``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..errors import E_NONFINITE_INPUT, NonFiniteInputError
from ..models import GLOBAL_PRECISION_LABELS, V2, Cell, EntityTable


MAGIC = "v2"


def build_payload(table: EntityTable) -> dict:
    entities: dict[str, dict] = {}
    order: list[str] = []
    for cell in table.cells:
        if cell.entity_id not in entities:
            entities[cell.entity_id] = {}
            order.append(cell.entity_id)
        label = cell.precision_label or "float64"
        if cell.missing:
            entities[cell.entity_id][cell.feature] = {
                "value": None,
                "precision": label,
                "explicit_null": True,
            }
        else:
            entities[cell.entity_id][cell.feature] = {
                "value": float(cell.value),
                "precision": label,
            }
    return {"format_version": MAGIC, "feature_order": list(table.feature_order),
            "entities": entities, "entity_order": order}


def write_file(path: str | Path, table: EntityTable) -> None:
    Path(path).write_text(
        json.dumps(build_payload(table), ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )


def parse_file(path: str | Path) -> EntityTable:
    text = Path(path).read_text(encoding="utf-8")
    return parse_text(text, source_path=str(path))


def parse_text(text: str, source_path: str = "<memory>") -> EntityTable:
    payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("format_version") != MAGIC:
        raise ValueError(f"{source_path} is not a v2 feature exchange document")
    entities = payload.get("entities")
    if not isinstance(entities, dict):
        raise ValueError(f"{source_path} missing object field 'entities'")

    table = EntityTable(source_version=V2)
    table.feature_order = list(payload.get("feature_order", []))

    for entity_id, feature_map in entities.items():
        if not isinstance(feature_map, dict):
            raise ValueError(
                f"{source_path}: entity {entity_id!r} must map to an object"
            )
        for feature, entry in feature_map.items():
            if not isinstance(entry, dict) or "precision" not in entry:
                raise ValueError(
                    f"{source_path}: {entity_id}/{feature} entry must contain "
                    "value and precision"
                )
            precision = entry["precision"]
            if precision not in GLOBAL_PRECISION_LABELS:
                # Parser keeps structural validity; metadata cross-check happens in
                # validation, but an unknown label can never be legal anywhere.
                from ..errors import PrecisionInvalidError

                raise PrecisionInvalidError(
                    f"unknown precision label {precision!r} for {entity_id}/{feature}",
                    entity_id=str(entity_id),
                    stage="v2_parse",
                )
            raw_token = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            value = entry.get("value")
            explicit_null = bool(entry.get("explicit_null", False))
            if value is None:
                table.cells.append(
                    Cell(
                        entity_id=str(entity_id),
                        feature=feature,
                        source_version=V2,
                        value=float("nan"),
                        missing=True,
                        precision_label=precision,
                        raw_text=raw_token,
                    )
                )
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise NonFiniteInputError(
                    f"non-finite v2 value for {entity_id}/{feature}: {value}",
                    entity_id=str(entity_id),
                    stage="v2_parse",
                )
            table.cells.append(
                Cell(
                    entity_id=str(entity_id),
                    feature=feature,
                    source_version=V2,
                    value=numeric,
                    missing=False,
                    precision_label=precision,
                    raw_text=raw_token,
                )
            )

    _append_missing_key_cells(table, entities)
    return table


def _append_missing_key_cells(table: EntityTable, entities: dict) -> None:
    """Represent absent feature keys explicitly with missing cells.

    Every entity receives one cell per feature observed anywhere in the
    document so downstream missing-position comparisons are complete.
    """
    known: set[tuple[str, str]] = {
        (cell.entity_id, cell.feature) for cell in table.cells
    }
    all_features = table.feature_names()
    for entity_id in entities:
        for feature in all_features:
            if (str(entity_id), feature) not in known:
                table.cells.append(
                    Cell(
                        entity_id=str(entity_id),
                        feature=feature,
                        source_version=V2,
                        value=float("nan"),
                        missing=True,
                        precision_label=None,
                        raw_text='{"absent_key":true}',
                    )
                )


BUSINESS_CODES = (E_NONFINITE_INPUT,)
