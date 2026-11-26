"""Interactive budget response semantics.

Each edit prefers the freshest completed summary within the latency budget.
If the latest parameters cannot be rendered in time, the most recent stale
summary is returned with E_STALE_SUMMARY and the stale objects are labelled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

from .codes import BizCode
from .pipeline import RenderEngine, RenderResult, render_incremental


@dataclass
class Summary:
    code: str
    global_version: int
    completed_frames: list[int]
    affected_frame_range: list[int]
    tile_completion: dict
    progress: float
    stale_objects: list[str] = field(default_factory=list)
    consumed_budget: int = 0
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _affected_range(frames: list[int]) -> list[int]:
    if not frames:
        return []
    return [min(frames), max(frames)]


def _tile_completion(engine: RenderEngine, result: RenderResult) -> dict:
    completion = {}
    for frame_id in range(engine.state.frame_count):
        done = sorted(result.completed_tiles.get(frame_id, []))
        completion[str(frame_id)] = {
            "done": done,
            "total": engine.cfg.tile_rows,
        }
    return completion


def _progress(engine: RenderEngine, result: RenderResult) -> float:
    total = engine.state.frame_count * engine.cfg.tile_rows
    if total == 0:
        return 0.0
    done = sum(len(rows) for rows in result.completed_tiles.values())
    return round(done / total, 6)


def handle_edit(engine: RenderEngine, command: dict,
                last_summary: Summary | None) -> tuple[Summary, list[str], RenderResult]:
    budget_seconds = engine.cfg.interact_budget_ms / 1000.0
    started = time.perf_counter()
    change = __import__("fxrender.pipeline", fromlist=["apply_command"]).apply_command(
        engine.state, command)
    if "error" in change:
        code = change["error"].split(":", 1)[0]
        stale = last_summary or Summary(
            code, engine.state.version, [], [], {}, 0.0)
        stale.code = code
        stale.detail = change["error"]
        return stale, [], RenderResult()

    invalidated, result = render_incremental(engine, change)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = Summary(
        code=BizCode.OK.value,
        global_version=engine.state.version,
        completed_frames=sorted(result.completed_frames),
        affected_frame_range=_affected_range(change.get("frames", [])),
        tile_completion=_tile_completion(engine, result),
        progress=_progress(engine, result),
        consumed_budget=result.consumed_budget,
        detail=f"elapsed_ms={elapsed_ms:.3f}",
    )
    if elapsed_ms > engine.cfg.interact_budget_ms:
        # The just-built summary reflects new params but exceeded the budget;
        # callers that cannot wait should retain the previous stale summary.
        summary.code = BizCode.E_STALE_SUMMARY.value
        summary.stale_objects = [
            f"frame:{fid}" for fid in range(engine.state.frame_count)
            if fid not in result.completed_frames]
        summary.detail += "; budget exceeded"
        if last_summary is not None:
            last_summary.code = BizCode.E_STALE_SUMMARY.value
            last_summary.stale_objects = [
                f"frame:{fid}" for fid in range(engine.state.frame_count)]
            return last_summary, invalidated, result
    return summary, invalidated, result
