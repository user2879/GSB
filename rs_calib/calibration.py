"""逐像素辐射定标阶段。"""
from __future__ import annotations

import numpy as np

from .errors import E_CALIB_FAILED, STAGE_CALIB, StageError
from .fingerprint import content_fingerprint, derive_cache_key, sensor_coeff_fingerprint
from .fixedpoint import to_fixed_point
from .models import CalibRecord, Tile


def stable_key(tile_id: str, band: int) -> str:
    """由瓦片 ID 与波段号派生的字典序稳定排序键（零填充保证数值位排序一致）。"""
    return f"{tile_id}|{band:06d}"


def calibrate_tile_band(tile: Tile, table: dict[str, dict], config: dict) -> CalibRecord:
    sensor = table[tile.sensor_id]
    coeff = sensor["bands"][tile.band]
    gain, offset = coeff["gain"], coeff["offset"]
    override = tile.fp_override or {}
    total_bits = int(override.get("fixed_point_bits", config["fixed_point_bits"]))
    frac_bits = int(override.get("fixed_point_frac_bits", config["fixed_point_frac_bits"]))

    counts = np.asarray(tile.counts, dtype=np.int64)
    try:
        # 行优先顺序即一维 counts 的天然顺序；饱和像素按饱和上限计数处理
        saturated = np.asarray(tile.saturated_flags, dtype=bool)
        if saturated.shape != counts.shape:
            saturated = np.zeros(counts.shape, dtype=bool)
        effective = np.where(saturated, np.int64(sensor["saturation_limit"]), counts)
        physical = effective.astype(np.float64) * gain + offset
        fixed_values = to_fixed_point(
            physical, total_bits, frac_bits,
            tile_id=tile.tile_id, band=tile.band, stage=STAGE_CALIB,
        )
    except StageError:
        raise
    except Exception as exc:  # 单瓦片异常隔离
        raise StageError(E_CALIB_FAILED, f"定标异常: {exc}",
                         tile_id=tile.tile_id, band=tile.band,
                         stage=STAGE_CALIB, sensor_id=tile.sensor_id) from exc

    upstream = {
        "tile": tile.to_dict(),
        "sensor_fingerprint": sensor_coeff_fingerprint(sensor),
        "fixed_point_bits": total_bits,
        "fixed_point_frac_bits": frac_bits,
        "quantization_levels": int(config["quantization_levels"]),
    }
    upstream_fp = content_fingerprint(upstream)
    key = derive_cache_key("calib", upstream_fp)
    return CalibRecord(
        tile_id=tile.tile_id,
        scene_id=tile.scene_id,
        sensor_id=tile.sensor_id,
        band=tile.band,
        fixed_values=[int(v) for v in fixed_values],
        saturation_count=int(np.count_nonzero(saturated)),
        pixel_count=int(counts.size),
        key=key,
        upstream_fingerprint=upstream_fp,
    )
