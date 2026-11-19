"""High-level resumable engine orchestration."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ._numpy import np
from .blocks import schedule_blocks
from .errors import EngineError
from .exporter import existing_output_matches, existing_output_version, export_outputs
from .incremental import IncrementalPlan, build_incremental_plan, materialize_final_spec
from .layout import EventLayout
from .spec import Spec, load_spec
from .state import FORMAT_VERSION, BlockRecord, StateStore
from .versioning import compute_version


INTERRUPTED_EXIT_CODE = 130


class SignalEvent(threading.Event):
    def __init__(self) -> None:
        super().__init__()
        self.signal_number: int | None = None

    def request(self, signal_number: int) -> None:
        self.signal_number = signal_number
        self.set()


@dataclass(frozen=True)
class EngineResult:
    status: str
    exit_code: int
    version: str
    frame_count: int
    report: dict[str, Any] | None
    frames_path: Path | None
    report_path: Path | None
    message: str

    def summary(self) -> dict[str, Any]:
        if self.status == "interrupted":
            return {
                "ok": False,
                "status": "interrupted",
                "exit_code": self.exit_code,
                "version": self.version,
                "frame_count": self.frame_count,
                "message": self.message,
            }
        assert self.report is not None
        return {
            "ok": True,
            "status": self.status,
            "exit_code": self.exit_code,
            "version": self.version,
            "frame_count": self.frame_count,
            "blocks": {
                "total": self.report["block_count"],
                "reused": self.report["state"]["reused_block_count"],
                "recomputed": self.report["state"]["recomputed_block_count"],
                "invalidated": self.report["state"]["invalidated_block_count"],
            },
            "business_codes": self.report["business_codes"],
            "frames_path": str(self.frames_path),
            "report_path": str(self.report_path),
        }


def _resolve_runtime_overrides(
    spec: Spec,
    *,
    state_dir: Path | None,
    output_dir: Path | None,
    concurrency: int | None,
    memory_budget_bytes: int | None,
) -> Spec:
    runtime = spec.runtime
    new_state = state_dir.resolve() if state_dir else runtime.state_dir
    new_output = output_dir.resolve() if output_dir else runtime.output_dir
    if new_state == new_output:
        raise EngineError(
            "E_PATH",
            "state_dir and output_dir must be distinct",
            details={"state_dir": str(new_state), "output_dir": str(new_output)},
        )
    if concurrency is not None and concurrency <= 0:
        raise EngineError("E_RUNTIME", "concurrency must be positive")
    if memory_budget_bytes is not None and memory_budget_bytes <= 0:
        raise EngineError("E_RUNTIME", "memory budget must be positive")
    runtime = replace(
        runtime,
        state_dir=new_state,
        output_dir=new_output,
        concurrency=concurrency if concurrency is not None else runtime.concurrency,
        memory_budget_bytes=(
            memory_budget_bytes if memory_budget_bytes is not None else runtime.memory_budget_bytes
        ),
    )
    return replace(spec, runtime=runtime)


def _manifest_declares_block(manifest: dict[str, Any] | None, block: BlockRecord) -> bool:
    if not manifest:
        return True
    entry = manifest.get("blocks", {}).get(str(block.index))
    if not isinstance(entry, dict):
        return True
    return (
        entry.get("digest") == block.digest
        and entry.get("frame_count") == block.frame_count
    )


def _validate_all_blocks(
    store: StateStore,
    manifest: dict[str, Any] | None,
) -> tuple[dict[int, BlockRecord], list[int], list[str]]:
    valid: dict[int, BlockRecord] = {}
    pending: list[int] = []
    codes: list[str] = []
    for index in range(store.spec.total_blocks):
        block, reason = store.read_block(index)
        if block is None:
            pending.append(index)
            if reason != "missing":
                codes.append("E_DIGEST")
            store.quarantine_invalid(index)
            continue
        record = BlockRecord.from_stored(block)
        if not _manifest_declares_block(manifest, record):
            pending.append(index)
            codes.append("E_DIGEST")
            store.quarantine_invalid(index)
            continue
        valid[index] = record
    return valid, pending, codes


def _event_report(
    layout: EventLayout,
    *,
    spec: Spec,
    plan: IncrementalPlan,
    actual_recomputed: set[int],
) -> dict[str, Any]:
    start_frame, end_frame = layout.visible_frame_range(spec.frame.frame_ms)
    start_frame = max(0, min(spec.total_frames, start_frame))
    end_frame = max(start_frame, min(spec.total_frames, end_frame))
    visible_blocks = layout.visible_block_indices(spec.frame.frame_ms, spec.frames_per_block)
    actual_touch = len(visible_blocks & actual_recomputed)
    codes: list[str] = []
    if layout.speed_limited:
        codes.append("E_SPEED_LIMIT")
    return {
        "id": layout.event.id,
        "speaker": layout.event.speaker,
        "language": layout.event.language,
        "text": layout.event.text,
        "lines": list(layout.lines),
        "line_widths_px": list(layout.line_widths),
        "wrap_width_px": layout.wrap_width,
        "reading_characters": layout.reading_chars,
        "min_display_ms": layout.min_display_ms,
        "actual_display_ms": layout.duration_ms,
        "speed_limited": layout.speed_limited,
        "critical_sound": layout.event.critical_sound,
        "bbox": layout.bbox.as_list(),
        "bbox_start_frame": start_frame,
        "bbox_end_frame": end_frame,
        "visible_block_indices": sorted(visible_blocks),
        "dependencies": layout.dependencies(),
        "invalidated_block_count": len(plan.event_invalidated_blocks.get(layout.event.id, ())),
        "recomputed_block_count": actual_touch,
        "business_codes": codes,
    }


def _build_report(
    *,
    spec: Spec,
    plan: IncrementalPlan,
    version: str,
    blocks: dict[int, BlockRecord],
    reused: set[int],
    recomputed: set[int],
    business_codes: set[str],
    export_mode: str,
) -> dict[str, Any]:
    subtitles = [
        _event_report(
            plan.layouts[event.id],
            spec=spec,
            plan=plan,
            actual_recomputed=recomputed,
        )
        for event in plan.final_events
    ]
    block_entries = []
    for index in range(spec.total_blocks):
        block = blocks[index]
        block_entries.append(
            {
                "index": index,
                "start_frame": block.metadata["start_frame"],
                "end_frame": block.metadata["end_frame"],
                "frame_count": block.metadata["frame_count"],
                "visible_event_ids": block.metadata["visible_event_ids"],
                "critical_event_ids": block.metadata["critical_event_ids"],
                "frame_visible_event_ids": block.metadata["frame_visible_event_ids"],
                "digest": block.digest,
                "reused": index in reused,
                "recomputed": index in recomputed,
            }
        )
    dependency_graph = {
        "style.font_size": [event.id for event in plan.final_events],
        "style.line_height": [event.id for event in plan.final_events],
        "style.max_lines": [event.id for event in plan.final_events],
        "style.padding": [event.id for event in plan.final_events],
        "style.contrast_enhance": [event.id for event in plan.final_events],
        "style.mask_opacity": [event.id for event in plan.final_events],
        "safe_area": [event.id for event in plan.final_events],
    }
    return {
        "schema_version": "subtitle-a11y-report/v1",
        "version": version,
        "frame_count": spec.total_frames,
        "frame_ms": spec.frame.frame_ms,
        "frame_size": [spec.frame.width, spec.frame.height],
        "background_value": spec.frame.background,
        "safe_area": spec.safe_area.as_list(),
        "block_ms": spec.runtime.block_ms,
        "block_count": spec.total_blocks,
        "subtitle_count": len(subtitles),
        "business_codes": sorted(business_codes),
        "state": {
            "reused_block_count": len(reused),
            "recomputed_block_count": len(recomputed),
            "invalidated_block_count": len(plan.invalidated_blocks),
            "invalidated_blocks": list(plan.invalidated_blocks),
        },
        "edit_steps": [step.as_dict() for step in plan.steps],
        "dependency_graph": dependency_graph,
        "subtitles": subtitles,
        "blocks": block_entries,
        "export": {"mode": export_mode},
    }


def _materialize_frames(store: StateStore, output_dir: Path) -> Path:
    """Stream persisted blocks into one .npy memmap without retaining them all."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f".frames-{uuid.uuid4().hex}.npy"
    frames = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.uint8,
        shape=(store.spec.total_frames, store.spec.frame.height, store.spec.frame.width),
    )
    try:
        for index in range(store.spec.total_blocks):
            block, reason = store.read_block(index)
            if block is None:
                raise EngineError(
                    "E_DIGEST",
                    "Block failed final read-back validation",
                    details={"block_index": index, "reason": reason},
                )
            start_frame, end_frame = store.expected_block_range(index)
            if int(block.frames.shape[0]) != end_frame - start_frame:
                raise EngineError(
                    "E_DIGEST",
                    "Declared frame count does not match assembled block",
                    details={"block_index": index},
                )
            frames[start_frame:end_frame] = block.frames
        frames.flush()
    except Exception:
        del frames
        try:
            path.unlink()
        except OSError:
            pass
        raise
    del frames
    return path


def run_engine(
    input_path: Path,
    *,
    state_dir: Path | None = None,
    output_dir: Path | None = None,
    concurrency: int | None = None,
    memory_budget_bytes: int | None = None,
    test_block_delay_ms: int = 0,
    stop_event: SignalEvent | None = None,
) -> EngineResult:
    stop_event = stop_event or SignalEvent()
    cancel_event = threading.Event()
    original_spec = load_spec(input_path)
    working_spec = _resolve_runtime_overrides(
        original_spec,
        state_dir=state_dir,
        output_dir=output_dir,
        concurrency=concurrency,
        memory_budget_bytes=memory_budget_bytes,
    )
    plan = build_incremental_plan(working_spec)
    spec = materialize_final_spec(working_spec, plan)
    version = compute_version(working_spec, plan)
    store = StateStore(spec, version)

    if stop_event.is_set():
        store.write_interrupted(stop_event.signal_number or 130, [])
        return EngineResult(
            "interrupted",
            INTERRUPTED_EXIT_CODE,
            version,
            0,
            None,
            None,
            None,
            "Termination accepted before the first block boundary",
        )

    manifest = store.read_manifest()
    business_codes: set[str] = set()
    if manifest is not None:
        expected_manifest = {
            "format_version": FORMAT_VERSION,
            "version": version,
            "frame_ms": spec.frame.frame_ms,
            "block_ms": spec.runtime.block_ms,
            "total_frames": spec.total_frames,
            "total_blocks": spec.total_blocks,
        }
        if any(manifest.get(key) != value for key, value in expected_manifest.items()):
            business_codes.add("E_DIGEST")
            manifest = None
    valid, pending, digest_codes = _validate_all_blocks(store, manifest)
    business_codes.update(digest_codes)
    if manifest and manifest.get("complete") and pending:
        business_codes.add("E_DIGEST")
    if manifest is None:
        manifest = store.base_manifest(complete=False)

    reused: set[int] = set(valid)
    recomputed: set[int] = set()

    def on_block(result: Any) -> None:
        nonlocal manifest
        stored = store.save_block(result.index, result.frames, result.metadata)
        manifest = store.record_block(manifest, stored)
        valid[result.index] = BlockRecord.from_stored(stored)
        recomputed.add(result.index)

    scheduler_result = schedule_blocks(
        spec,
        plan,
        pending,
        stop_event=stop_event,
        cancel_event=cancel_event,
        on_block=on_block,
        test_block_delay_ms=test_block_delay_ms,
    )
    if scheduler_result.interrupted:
        completed = sorted(set(valid))
        store.write_interrupted(
            scheduler_result.signal_number or stop_event.signal_number or 130, completed
        )
        return EngineResult(
            "interrupted",
            INTERRUPTED_EXIT_CODE,
            version,
            spec.total_frames,
            None,
            None,
            None,
            f"Persisted {len(completed)} complete blocks; resume will reuse them",
        )

    if stop_event.is_set():
        completed = sorted(set(valid))
        store.write_interrupted(
            stop_event.signal_number or 130, completed
        )
        return EngineResult(
            "interrupted",
            INTERRUPTED_EXIT_CODE,
            version,
            spec.total_frames,
            None,
            None,
            None,
            "Termination accepted at the final block boundary before export",
        )

    # Final digest audit.  Any declared count/digest mismatch is repaired by a
    # second bounded recomputation pass, while the report retains E_DIGEST.
    for _attempt in range(2):
        valid, pending, digest_codes = _validate_all_blocks(store, manifest)
        if digest_codes:
            business_codes.add("E_DIGEST")
        if not pending:
            break
        if stop_event.is_set():
            store.write_interrupted(
                stop_event.signal_number or 130, sorted(set(valid))
            )
            return EngineResult(
                "interrupted",
                INTERRUPTED_EXIT_CODE,
                version,
                spec.total_frames,
                None,
                None,
                None,
                "Termination accepted during digest repair",
            )
        for index in pending:
            reused.discard(index)
            store.quarantine_invalid(index)

        def on_repair(result: Any) -> None:
            nonlocal manifest
            stored = store.save_block(result.index, result.frames, result.metadata)
            manifest = store.record_block(manifest, stored)
            valid[result.index] = BlockRecord.from_stored(stored)
            recomputed.add(result.index)

        repaired = schedule_blocks(
            spec,
            plan,
            pending,
            stop_event=stop_event,
            cancel_event=cancel_event,
            on_block=on_repair,
            test_block_delay_ms=0,
        )
        if repaired.interrupted:
            completed = sorted(set(valid))
            store.write_interrupted(
                repaired.signal_number or stop_event.signal_number or 130, completed
            )
            return EngineResult(
                "interrupted",
                INTERRUPTED_EXIT_CODE,
                version,
                spec.total_frames,
                None,
                None,
                None,
                "Termination accepted while repairing block digests",
            )
    else:
        raise EngineError(
            "E_DIGEST",
            "Block digest validation did not converge after repair",
            details={"pending_blocks": pending},
        )

    if len(valid) != spec.total_blocks:
        raise EngineError(
            "E_DIGEST",
            "Not all blocks are available after completion",
            details={"expected": spec.total_blocks, "actual": len(valid)},
        )
    rebuilt_manifest = store.base_manifest(complete=True)
    rebuilt_manifest["blocks"] = {
        str(index): {
            "digest": valid[index].digest,
            "frame_count": valid[index].frame_count,
        }
        for index in range(spec.total_blocks)
    }
    manifest = store.finalize_manifest(rebuilt_manifest)
    store.clear_interrupted()
    for layout in plan.layouts.values():
        if layout.speed_limited:
            business_codes.add("E_SPEED_LIMIT")

    output_matches = existing_output_matches(spec.runtime.output_dir, version)
    previous_output_version = existing_output_version(spec.runtime.output_dir)
    export_mode = "refreshed_same_version" if output_matches else "replaced"
    report = _build_report(
        spec=spec,
        plan=plan,
        version=version,
        blocks=valid,
        reused=reused,
        recomputed=recomputed,
        business_codes=business_codes,
        export_mode=export_mode,
    )
    report["export"]["previous_version"] = previous_output_version
    if output_matches:
        prior_commit = json.loads((spec.runtime.output_dir / "commit.json").read_text())
        frames_source = spec.runtime.output_dir / str(prior_commit["generation"]) / "frames.npy"
        temporary_frames: Path | None = None
    else:
        temporary_frames = _materialize_frames(store, spec.runtime.output_dir)
        frames_source = temporary_frames
    try:
        export_outputs(spec.runtime.output_dir, frames_source, report)
    finally:
        if temporary_frames is not None:
            try:
                temporary_frames.unlink()
            except OSError:
                pass
    frames_path = spec.runtime.output_dir / "frames.npy"
    report_path = spec.runtime.output_dir / "report.json"

    return EngineResult(
        "complete",
        0,
        version,
        spec.total_frames,
        report,
        frames_path,
        report_path,
        "All blocks committed and atomically exported",
    )
