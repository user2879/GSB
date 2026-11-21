"""Deterministic SVG heatmap visualization stage (no external dependencies)."""

from __future__ import annotations

from .cache import derived_key, fingerprint

CELL_W = 72
CELL_H = 28
MARGIN_X = 150
MARGIN_Y = 60


def _heat_color(value: float) -> str:
    clamped = max(0.0, min(1.0, value))
    red = int(99 + clamped * 156)
    green = int(190 - clamped * 140)
    blue = int(222 - clamped * 180)
    return f"rgb({red},{green},{blue})"


def build_heatmap_svg(decision_payload: dict) -> tuple[str, str]:
    decisions = decision_payload["decisions"]
    columns = sorted({row["column"] for row in decisions})
    windows = sorted({row["window_index"] for row in decisions})
    primary_tier = decision_payload["tier_order"][len(decision_payload["tier_order"]) // 2]
    score_map = {
        (row["column"], row["window_index"]): next(
            t["score_retained"] for t in row["tiers"] if t["tier"] == primary_tier
        )
        for row in decisions
    }
    width = MARGIN_X + len(windows) * CELL_W + 20
    height = MARGIN_Y + len(columns) * CELL_H + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="10" y="24" font-family="monospace" font-size="16">'
        f"Drift heatmap (tier={primary_tier})</text>",
    ]
    for j, window in enumerate(windows):
        x = MARGIN_X + j * CELL_W
        parts.append(
            f'<text x="{x + CELL_W / 2}" y="{MARGIN_Y - 12}" '
            f'font-family="monospace" font-size="12" text-anchor="middle">w{window}</text>'
        )
    max_score = max((abs(v) for v in score_map.values()), default=1.0) or 1.0
    for i, column in enumerate(columns):
        y = MARGIN_Y + i * CELL_H
        parts.append(
            f'<text x="10" y="{y + CELL_H / 2 + 4}" font-family="monospace" '
            f'font-size="11">{column[:20]}</text>'
        )
        for j, window in enumerate(windows):
            value = score_map.get((column, window), 0.0)
            x = MARGIN_X + j * CELL_W
            parts.append(
                f'<rect x="{x}" y="{y}" width="{CELL_W - 2}" height="{CELL_H - 2}" '
                f'fill="{_heat_color(abs(value) / max_score)}" stroke="#333"/>'
            )
            parts.append(
                f'<text x="{x + CELL_W / 2}" y="{y + CELL_H / 2 + 4}" '
                f'font-family="monospace" font-size="10" text-anchor="middle">'
                f"{value:.4f}</text>"
            )
    parts.append("</svg>")
    svg = "".join(parts)
    key = derived_key(fingerprint(decision_payload), {"viz": "svg_heatmap"}, "visual")
    return key, svg
