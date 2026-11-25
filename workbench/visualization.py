from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .configuration import WorkbenchConfig


def render_tile_status(
    config: WorkbenchConfig,
    base_final: np.ndarray,
    pending_ids: list[str],
    failed: list[dict],
) -> np.ndarray:
    image = Image.fromarray(base_final, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pending_base = [tile_id for tile_id in pending_ids if tile_id.startswith(f"L{config.levels - 1:02d}")]
    for tile_id in pending_base:
        _, row_text, col_text = tile_id.split("R")[0], tile_id.split("R")[1][:3], tile_id.split("C")[1][:3]
        row, col = int(row_text), int(col_text)
        size = config.tile_sizes[-1]
        x0, y0 = col * size, row * size
        x1, y1 = x0 + size, y0 + size
        draw.rectangle([x0, y0, x1, y1], fill=(255, 184, 28, 72), outline=(255, 196, 64, 255), width=2)
        for offset in range(-size, size * 2, 10):
            draw.line([(x0 + offset, y1), (x0 + offset + size, y0)], fill=(255, 220, 120, 150), width=1)

    failed_keys = {item.get("task_key", "") for item in failed}
    for key in failed_keys:
        parts = key.split(":")
        if len(parts) != 2 or not parts[1].startswith("L"):
            continue
        try:
            level = int(parts[1][1:3])
            row = int(parts[1].split("R")[1][:4])
            col = int(parts[1].split("C")[1][:4])
        except (IndexError, ValueError):
            continue
        if level != config.levels - 1:
            continue
        size = config.tile_sizes[level]
        x0, y0 = col * size, row * size
        draw.rectangle([x0, y0, x0 + size, y0 + size], fill=(220, 38, 38, 95), outline=(255, 90, 90, 255), width=3)
        draw.line([x0 + 5, y0 + 5, x0 + size - 5, y0 + size - 5], fill=(255, 230, 230, 255), width=3)
        draw.line([x0 + size - 5, y0 + 5, x0 + 5, y0 + size - 5], fill=(255, 230, 230, 255), width=3)

    composited = Image.alpha_composite(image, overlay).convert("RGB")
    return np.asarray(composited, dtype=np.uint8)
