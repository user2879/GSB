"""Optional Tkinter visualization with graceful degradation.

Tkinter ships with the Python standard library on Windows/macOS official
installers. On minimal Linux builds the native Tk runtime may be absent; in
that case :func:`show_timeline` returns ``False`` instead of crashing so the
component stays usable in headless environments.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

CLUSTER_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
                  "#B279A2", "#FF9DA6", "#9D755D")


def tkinter_available() -> bool:
    try:
        import tkinter  # noqa: F401
        _ = tkinter
        return True
    except Exception:
        return False


def render_html_timeline(result: Dict[str, Any], path: str) -> str:
    """Headless fallback: produce a self-contained HTML/SVG timeline."""
    segments = result.get("segments", [])
    if not segments:
        max_ms = 1.0
    else:
        max_ms = max(seg["end_ms"] for seg in segments) or 1.0
    speaker_ids = sorted({seg["speaker"] for seg in segments})
    color_map = {sid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                 for i, sid in enumerate(speaker_ids)}
    width, row_h, pad = 900, 34, 60
    height = pad + len(speaker_ids) * row_h + 40
    bars = []
    for seg in segments:
        row = speaker_ids.index(seg["speaker"])
        x = pad + seg["start_ms"] / max_ms * (width - pad - 20)
        w = max(2, (seg["end_ms"] - seg["start_ms"]) / max_ms
                * (width - pad - 20))
        y = pad // 2 + row * row_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="22" '
            f'rx="4" fill="{color_map[seg["speaker"]]}"/>'
            f'<text x="{x + 3:.1f}" y="{y + 15}" font-size="11" '
            f'fill="white">{seg["speaker"]}</text>')
    labels = [
        f'<text x="10" y="{pad // 2 + i * row_h + 16}" font-size="12">'
        f'{sid}</text>' for i, sid in enumerate(speaker_ids)]
    svg = (f'<svg width="{width}" height="{height}" '
           f'xmlns="http://www.w3.org/2000/svg">'
           + "".join(labels) + "".join(bars) + "</svg>")
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Diarization timeline</title></head><body>"
            f"<h3>Speaker timeline ({len(segments)} segments)</h3>" + svg
            + "</body></html>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path


def show_timeline(result: Dict[str, Any], html_fallback: str) -> Dict[str, Any]:
    """Open a Tk window when available; otherwise write an HTML fallback."""
    if not tkinter_available():
        path = render_html_timeline(result, html_fallback)
        return {"backend": "html", "path": path,
                "reason": "tkinter native runtime unavailable"}
    import tkinter as tk  # local import keeps headless imports cheap

    segments = result.get("segments", [])
    speaker_ids = sorted({seg["speaker"] for seg in segments})
    color_map = {sid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                 for i, sid in enumerate(speaker_ids)}
    root = tk.Tk()
    root.title("Speaker diarization timeline")
    canvas = tk.Canvas(root, width=960,
                       height=80 + len(speaker_ids) * 40, bg="white")
    canvas.pack()
    max_ms = max((seg["end_ms"] for seg in segments), default=1.0) or 1.0
    for row, sid in enumerate(speaker_ids):
        y = 30 + row * 40
        canvas.create_text(10, y + 12, text=sid, anchor="w")
        for seg in segments:
            if seg["speaker"] != sid:
                continue
            x0 = 90 + seg["start_ms"] / max_ms * 820
            x1 = max(x0 + 2, 90 + seg["end_ms"] / max_ms * 820)
            canvas.create_rectangle(x0, y, x1, y + 24,
                                    fill=color_map[sid], outline="")
            canvas.create_text(x0 + 4, y + 12, text=sid, fill="white",
                               anchor="w")
    root.mainloop()
    return {"backend": "tkinter", "path": ""}


def load_summary(summary_path: str) -> Dict[str, Any]:
    with open(summary_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
