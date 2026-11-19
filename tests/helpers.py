from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from subtitle_engine.engine import EngineResult, SignalEvent, run_engine


ROOT = Path(__file__).resolve().parent.parent


def event(
    event_id: str,
    start: int,
    end: int,
    text: str = "Alpha beta",
    *,
    rate: int = 100,
    critical: bool = False,
    anchor: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "start_ms": start,
        "end_ms": end,
        "text": text,
        "speaker": f"speaker-{event_id}",
        "language": "en",
        "max_chars_per_sec": rate,
        "critical_sound": critical,
        "anchor": anchor
        or {"x": 100, "y": 110, "horizontal": "center", "vertical": "bottom"},
    }


def description(
    *,
    name: str = "case",
    events: list[dict[str, Any]] | None = None,
    edits: list[dict[str, Any]] | None = None,
    style: dict[str, Any] | None = None,
    duration_ms: int = 400,
    frame_ms: int = 20,
    block_ms: int = 100,
    concurrency: int = 2,
    memory_budget_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    result = {
        "schema_version": "subtitle-engine/v1",
        "duration_ms": duration_ms,
        "frame": {"width": 200, "height": 120, "frame_ms": frame_ms, "background": 18},
        "safe_area": {"x": 10, "y": 10, "width": 180, "height": 100},
        "style": {
            "font_size": 10,
            "line_height": 14,
            "max_lines": 4,
            "padding": 5,
            "contrast_enhance": True,
            "mask_opacity": 0.65,
        },
        "events": events or [event("e1", 0, 200), event("e2", 200, 400, "Delta")],
        "edits": edits or [],
        "runtime": {
            "concurrency": concurrency,
            "memory_budget_bytes": memory_budget_bytes,
            "block_ms": block_ms,
            "state_dir": f"state/{name}",
            "output_dir": f"output/{name}",
        },
    }
    if style:
        result["style"].update(style)
    return result


def write_description(root: Path, name: str, value: dict[str, Any]) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "input.json"
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return path


def run_case(
    root: Path,
    name: str,
    value: dict[str, Any],
    *,
    concurrency: int | None = None,
    memory_budget_bytes: int | None = None,
    delay_ms: int = 0,
    stop_event: SignalEvent | None = None,
) -> EngineResult:
    path = write_description(root, name, value)
    return run_engine(
        path,
        state_dir=root / name / "state",
        output_dir=root / name / "output",
        concurrency=concurrency,
        memory_budget_bytes=memory_budget_bytes,
        test_block_delay_ms=delay_ms,
        stop_event=stop_event,
    )


def run_cli(root: Path, name: str, value: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    path = write_description(root, name, value)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "subtitle_engine",
            "--input",
            str(path),
            "--state-dir",
            str(root / name / "state"),
            "--output-dir",
            str(root / name / "output"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )


def clone(value: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(value)
