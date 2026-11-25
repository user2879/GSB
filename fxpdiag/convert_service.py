"""双版本互转编排：v1->v2、v2->v1（旁路）、再升回的往返损耗统计。"""

from __future__ import annotations

import os

from .conversion import (
    SIDECAR_SUFFIX,
    ConversionRecord,
    layer_roundtrip_loss,
    read_sidecar,
    v1_to_v2_with_sidecar,
    v2_to_v1_row,
    write_sidecar,
)
from .models import V1WeightRow, V2WeightRow
from .weights import write_v1, write_v2


def downgrade_v2_to_v1(
    v2_rows: list[V2WeightRow], out_v1_path: str
) -> tuple[list[V1WeightRow], list, dict[str, float]]:
    v1_rows: list[V1WeightRow] = []
    sidecars = []
    records: list[ConversionRecord] = []
    for row in v2_rows:
        v1_row, sidecar = v2_to_v1_row(row)
        v1_rows.append(v1_row)
        sidecars.append(sidecar)
        records.append(
            ConversionRecord(
                layer=row.layer,
                row_index=row.row_index,
                direction="v2->v1",
                scale_before=row.scale,
                scale_after=v1_row.scale_permille / 1000.0,
                loss_abs=abs(row.scale - v1_row.scale_permille / 1000.0),
                precision_tag=sidecar.precision_tag,
            )
        )
    write_v1(out_v1_path, v1_rows)
    write_sidecar(out_v1_path + SIDECAR_SUFFIX, sidecars)
    return v1_rows, sidecars, layer_roundtrip_loss(records)


def upgrade_v1_to_v2(
    v1_rows: list[V1WeightRow],
    out_v2_path: str,
    sidecar_path: str | None = None,
) -> tuple[list[V2WeightRow], list[ConversionRecord], dict[str, float]]:
    sidecars = read_sidecar(sidecar_path) if sidecar_path and os.path.exists(sidecar_path) else {}
    v2_rows: list[V2WeightRow] = []
    records: list[ConversionRecord] = []
    for row in v1_rows:
        sidecar = sidecars.get((row.layer, row.row_index))
        v2_row, record = v1_to_v2_with_sidecar(row, sidecar)
        v2_rows.append(v2_row)
        records.append(record)
    write_v2(out_v2_path, v2_rows)
    return v2_rows, records, layer_roundtrip_loss(records)


def full_roundtrip(
    v2_rows: list[V2WeightRow], work_dir: str
) -> dict:
    """v2 -> v1（旁路） -> v2 还原，返回逐行/逐层往返损耗。"""

    os.makedirs(work_dir, exist_ok=True)
    v1_path = os.path.join(work_dir, "roundtrip_v1.txt")
    restored_v2_path = os.path.join(work_dir, "roundtrip_v2_restored.json")
    v1_rows, _, down_loss = downgrade_v2_to_v1(v2_rows, v1_path)
    _, records, up_loss = upgrade_v1_to_v2(
        v1_rows, restored_v2_path, v1_path + SIDECAR_SUFFIX
    )
    combined: dict[str, float] = {}
    for layer, value in down_loss.items():
        combined[layer] = combined.get(layer, 0.0) + value
    for layer, value in up_loss.items():
        combined[layer] = combined.get(layer, 0.0) + value
    return {
        "v1_path": v1_path,
        "restored_v2_path": restored_v2_path,
        "records": [item.to_dict() for item in records],
        "layer_roundtrip_loss": combined,
    }
