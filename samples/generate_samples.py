#!/usr/bin/env python3
"""Generate deterministic local subtitle descriptions.

No real subtitle files, attachments, network access, or media tools are used.
The generated JSON is stable across operating systems and Python processes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"


def anchor_bottom_center() -> dict[str, object]:
    return {"x": 160, "y": 160, "horizontal": "center", "vertical": "bottom"}


def base_description(name: str, *, concurrency: int = 3, duration_ms: int = 1600) -> dict:
    return {
        "schema_version": "subtitle-engine/v1",
        "duration_ms": duration_ms,
        "frame": {
            "width": 320,
            "height": 180,
            "frame_ms": 20,
            "background": 18,
        },
        "safe_area": {"x": 16, "y": 16, "width": 288, "height": 144},
        "style": {
            "font_size": 12,
            "line_height": 16,
            "max_lines": 5,
            "padding": 6,
            "contrast_enhance": True,
            "mask_opacity": 0.62,
        },
        "events": [
            {
                "id": "e1",
                "start_ms": 0,
                "end_ms": 420,
                "text": "System alert: hull pressure rising.",
                "speaker": "COMPUTER",
                "language": "en",
                "max_chars_per_sec": 16,
                "critical_sound": True,
                "anchor": anchor_bottom_center(),
            },
            {
                "id": "e2",
                "start_ms": 280,
                "end_ms": 860,
                "text": "Kira: We need to reach the bridge before the gate closes.",
                "speaker": "Kira",
                "language": "en",
                "max_chars_per_sec": 22,
                "critical_sound": False,
                "anchor": anchor_bottom_center(),
            },
            {
                "id": "e3",
                "start_ms": 760,
                "end_ms": 1260,
                "text": "Critical sound detected near the east airlock.",
                "speaker": "SFX",
                "language": "en",
                "max_chars_per_sec": 30,
                "critical_sound": True,
                "anchor": anchor_bottom_center(),
            },
            {
                "id": "e4",
                "start_ms": 1180,
                "end_ms": 1600,
                "text": "All crew report to stations.",
                "speaker": "Tora",
                "language": "en",
                "max_chars_per_sec": 20,
                "critical_sound": False,
                "anchor": anchor_bottom_center(),
            },
            {
                "id": "e5",
                "start_ms": 200,
                "end_ms": 1500,
                "text": "Ready.",
                "speaker": "Pilot",
                "language": "en",
                "max_chars_per_sec": 10,
                "critical_sound": False,
                "anchor": {"x": 20, "y": 20, "horizontal": "left", "vertical": "top"},
            },
        ],
        "edits": [
            {
                "op": "set_text",
                "event_id": "e2",
                "text": "Kira: Bridge is sealed. Use the maintenance shaft.",
            },
            {"op": "set_style", "field": "font_size", "value": 14},
            {"op": "set_time", "event_id": "e3", "start_ms": 800, "end_ms": 1300},
            {"op": "set_style", "field": "contrast_enhance", "value": False},
        ],
        "runtime": {
            "concurrency": concurrency,
            "memory_budget_bytes": 4 * 1024 * 1024,
            "block_ms": 100,
            "state_dir": f"state/{name}",
            "output_dir": f"output/{name}",
        },
    }


def write(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    main_desc = base_description("main", concurrency=3)
    write("main", main_desc)

    serial = copy.deepcopy(main_desc)
    serial["runtime"]["concurrency"] = 1
    serial["runtime"]["state_dir"] = "state/serial"
    serial["runtime"]["output_dir"] = "output/serial"
    write("serial", serial)

    parallel = copy.deepcopy(main_desc)
    parallel["runtime"]["concurrency"] = 3
    parallel["runtime"]["state_dir"] = "state/parallel"
    parallel["runtime"]["output_dir"] = "output/parallel"
    write("parallel", parallel)

    interrupt = base_description("interrupt", concurrency=1, duration_ms=2400)
    write("interrupt", interrupt)

    invalid = base_description("invalid", concurrency=1)
    invalid["events"][0]["anchor"] = {
        "x": 2,
        "y": 2,
        "horizontal": "left",
        "vertical": "top",
    }
    invalid["runtime"]["state_dir"] = "state/invalid"
    invalid["runtime"]["output_dir"] = "output/invalid"
    write("invalid_anchor", invalid)

    index = {
        "generator": "samples/generate_samples.py",
        "deterministic": True,
        "samples": [
            "main.json",
            "serial.json",
            "parallel.json",
            "interrupt.json",
            "invalid_anchor.json",
        ],
    }
    (OUT / "index.json").write_text(
        json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
