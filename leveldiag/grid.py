"""离散关卡网格装配。

网格是量化后的最终图块表示：地形类别 + 量化高程/密度档级 + 资源/出生/目标标记。
网格内容哈希覆盖全部图块，供账本与复算器引用。
"""

from __future__ import annotations

from .spec import Spec
from .util import content_hash


def build_grid(spec: Spec, continuous: dict, mapping: dict) -> dict:
    width, height = spec.width, spec.height
    terrain_grid = continuous["terrain_grid"]
    cells: dict[str, dict] = {}
    for rid, shard in mapping["region_shards"].items():
        for entry in shard["mappings"]:
            x, y = entry["cell"]
            cells[f"{x},{y}"] = {
                "cell": [x, y],
                "region_id": rid,
                "terrain": entry["terrain"],
                "elevation_q": entry["elevation"]["quantized"],
                "elevation_level": entry["elevation"]["level_index"],
                "density_q": entry["density"]["quantized"],
                "density_level": entry["density"]["level_index"],
                "is_resource": False,
                "role": None,
            }

    for res in continuous["resources"]:
        key = f"{res['cell'][0]},{res['cell'][1]}"
        if key in cells:
            cells[key]["is_resource"] = True

    sx, sy = spec.raw["points"]["spawn"]
    tx, ty = spec.raw["points"]["target"]
    cells[f"{sx},{sy}"]["role"] = "spawn"
    cells[f"{tx},{ty}"]["role"] = "target"

    # 行主序输出，顺序确定
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            row.append(cells[f"{x},{y}"])
        rows.append(row)

    grid = {
        "spec_hash": spec.spec_hash,
        "mapping_hash": mapping["mapping_hash"],
        "width": width,
        "height": height,
        "tiles": rows,
        "resource_cells": [r["cell"] for r in continuous["resources"]],
        "spawn": [sx, sy],
        "target": [tx, ty],
    }
    grid["grid_hash"] = content_hash(
        {"width": width, "height": height, "tiles": rows,
         "resource_cells": grid["resource_cells"],
         "spawn": grid["spawn"], "target": grid["target"]})
    return grid
