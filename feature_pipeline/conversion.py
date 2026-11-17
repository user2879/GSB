"""Version conversion with explicit information-retention sidecars."""

from __future__ import annotations

import json
from pathlib import Path

from .errors import PrecisionInvalidError
from .models import V1, V2, V3, Cell, EntityTable, FeatureMeta
from .parsers import v1_fixedwidth, v2_json, v3_binary


def validate_precision(cell: Cell, metadata: dict[str, FeatureMeta]) -> None:
    meta = metadata.get(cell.feature)
    if meta is None or not meta.allowed_precisions or cell.precision_label is None:
        return
    if cell.precision_label not in meta.allowed_precisions:
        raise PrecisionInvalidError(
            f"precision label {cell.precision_label!r} not allowed for feature "
            f"{cell.feature!r} (allowed: {list(meta.allowed_precisions)})",
            entity_id=cell.entity_id,
            stage="precision_validate",
        )


def v1_to_v2(table: EntityTable, metadata: dict[str, FeatureMeta]) -> EntityTable:
    """Promote per-mille integers to floats with metadata-derived labels."""
    promoted = EntityTable(
        source_version=V2,
        feature_order=list(table.feature_order),
        partition_boundaries=dict(table.partition_boundaries),
    )
    for cell in table.cells:
        meta = metadata.get(cell.feature)
        label = (
            meta.allowed_precisions[0]
            if meta and meta.allowed_precisions
            else "float64"
        )
        if label == "per_mille":
            label = "float64"
        if cell.missing:
            new_value = float("nan")
        else:
            permille = int(round(cell.value * 1000))
            new_value = permille / 1000.0  # exact decimal scaling
        promoted.cells.append(
            cell.clone(
                source_version=V2,
                value=new_value,
                precision_label=label,
            )
        )
    return promoted


def v2_to_v1(table: EntityTable, sidecar_path: str | Path) -> EntityTable:
    """Round floats to per-mille and record truncation residuals in a sidecar."""
    downgraded = EntityTable(
        source_version=V1,
        feature_order=list(table.feature_order),
        partition_boundaries=dict(table.partition_boundaries),
    )
    sidecar: dict[str, dict] = {}
    for cell in table.cells:
        if cell.missing:
            downgraded.cells.append(
                cell.clone(source_version=V1, value=float("nan"),
                           precision_label="per_mille")
            )
            continue
        permille = int(round(cell.value * 1000))
        restored = permille / 1000.0
        residual = cell.value - restored
        downgraded.cells.append(
            cell.clone(
                source_version=V1,
                value=restored,
                precision_label="per_mille",
            )
        )
        if residual != 0.0:
            sidecar[f"{cell.entity_id}|{cell.feature}"] = {
                "entity_id": cell.entity_id,
                "feature": cell.feature,
                "original_value": cell.value,
                "permille": permille,
                "truncation_residual": residual,
                "precision_label": cell.precision_label,
                "source_version": cell.source_version,
                "source_marker": "v2_to_v1_downgrade",
            }

    downgrade_losses = []
    if sidecar:
        downgrade_losses.append(
            {
                "lost_attribute": "sub_permille_fraction",
                "reason": "v1 stores integer per-mille units only",
                "mitigation": "residual retained in this sidecar",
                "affected_record_count": len(sidecar),
            }
        )
    payload = {
        "sidecar_type": "v2_to_v1_residuals",
        "record_count": len(sidecar),
        "records": sidecar,
        "downgrade_losses": downgrade_losses,
    }
    Path(sidecar_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return downgraded


def to_v3(table: EntityTable) -> tuple[EntityTable, list[dict]]:
    """Convert any table to the v3 partitioned representation."""
    losses: list[dict] = []
    target = EntityTable(
        source_version=V3,
        feature_order=list(table.feature_order),
        partition_boundaries=dict(table.partition_boundaries)
        if table.partition_boundaries
        else {0: table.entity_ids()},
    )
    for cell in table.cells:
        if cell.missing:
            narrowed = float("nan")
        else:
            narrowed = struct_roundtrip_float32(cell.value)
        if not cell.missing and narrowed != cell.value:
            losses.append(
                {
                    "entity_id": cell.entity_id,
                    "feature": cell.feature,
                    "lost_attribute": "float64_precision",
                    "reason": "v3 stores float32 only",
                    "original_value": cell.value,
                    "float32_value": narrowed,
                    "absolute_error": abs(narrowed - cell.value),
                }
            )
        target.cells.append(
            cell.clone(
                source_version=V3,
                value=narrowed,
                precision_label="float32",
            )
        )
    if table.source_version != V3:
        losses.append(
            {
                "lost_attribute": "source_precision_label",
                "reason": "v3 only carries float32 slots; original labels are "
                "not representable",
                "mitigation": "labels retained in merge provenance records",
            }
        )
    return target, losses


def v3_to_v2(table: EntityTable) -> tuple[EntityTable, list[dict]]:
    """Promote v3 slots to v2 while preserving partition boundaries/order."""
    losses = [
        {
            "lost_attribute": "partition_offsets",
            "reason": "v2 JSON has no offset table; partition boundary order is "
            "kept on feature_order/entity ordering instead",
            "mitigation": "partition_boundaries retained in conversion metadata",
        }
    ]
    target = EntityTable(
        source_version=V2,
        feature_order=list(table.feature_order),
        partition_boundaries=dict(table.partition_boundaries),
    )
    for cell in table.cells:
        target.cells.append(
            cell.clone(source_version=V2, precision_label="float32")
        )
    return target, losses


def v3_to_v1(table: EntityTable, sidecar_path: str | Path) -> tuple[EntityTable, list[dict]]:
    v2_table, losses = v3_to_v2(table)
    downgraded = v2_to_v1(v2_table, sidecar_path)
    losses.append(
        {
            "lost_attribute": "partition_structure",
            "reason": "v1 fixed-width rows cannot encode partition headers",
            "mitigation": "partition boundaries written to residual sidecar",
        }
    )
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    sidecar["partition_boundaries"] = {
        str(pid): entity_ids for pid, entity_ids in sorted(table.partition_boundaries.items())
    }
    sidecar.setdefault("downgrade_losses", []).extend(losses)
    Path(sidecar_path).write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return downgraded, losses


def struct_roundtrip_float32(value: float) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def serialize(table: EntityTable, path: str | Path) -> None:
    if table.source_version == V1:
        v1_fixedwidth.write_file(path, table)
    elif table.source_version == V2:
        v2_json.write_file(path, table)
    elif table.source_version == V3:
        v3_binary.write_file(path, table)
    else:
        raise ValueError(f"unknown source version {table.source_version}")
