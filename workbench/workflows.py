from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .commands import RuntimeState, apply_commands, validate_command_stream
from .configuration import WorkbenchConfig, load_config
from .engine import assemble_final
from .hashing import array_fingerprint
from .image_io import load_rgb_png
from .masks import load_mask_document
from .session import WorkbenchSession
from .snapshot import SnapshotStore


def load_bundle(config_path: str | Path):
    config = load_config(config_path)
    rectangles = load_mask_document(config.mask_path, config.width, config.height)
    pixels, image_fingerprint = load_rgb_png(config.image_path, config.width, config.height)
    return config, rectangles, pixels, image_fingerprint


def validated_commands(commands_path: str | Path, config: WorkbenchConfig, rectangles):
    return validate_command_stream(commands_path, config, rectangles)


def build_full_from_state(
    config: WorkbenchConfig,
    pixels: np.ndarray,
    image_fingerprint: str,
    rectangles,
    state: RuntimeState,
    max_workers: int = 1,
) -> WorkbenchSession:
    session = WorkbenchSession(config, pixels, image_fingerprint, rectangles, state=state)
    total = len(session.planned.order)
    session.applied_cursor = total_commands_hint(state)
    session.completed_cursor = session.applied_cursor
    session.run_batch(budget=total, max_workers=max_workers)
    return session


def total_commands_hint(state: RuntimeState) -> int:
    # The caller normally overwrites this cursor; state itself does not carry command count.
    return getattr(state, "command_count", 0)


def build_full_final(
    config: WorkbenchConfig,
    pixels: np.ndarray,
    image_fingerprint: str,
    rectangles,
    commands: list[dict[str, Any]],
    max_workers: int = 1,
) -> WorkbenchSession:
    state = RuntimeState.initial(config, rectangles)
    state = apply_commands(state, commands)
    session = WorkbenchSession(config, pixels, image_fingerprint, rectangles, state=state)
    session.applied_cursor = len(commands)
    session.completed_cursor = len(commands)
    session.run_batch(budget=len(session.planned.order), max_workers=max_workers)
    return session


def compare_sessions(incremental: WorkbenchSession, full: WorkbenchSession) -> dict[str, Any]:
    differences: list[str] = []
    if incremental.config.levels != full.config.levels:
        differences.append("level_count")
    layer_equal = []
    for level in range(incremental.config.levels):
        repair_equal = np.array_equal(incremental.repaired_levels[level], full.repaired_levels[level])
        style_equal = np.array_equal(incremental.styled_levels[level], full.styled_levels[level])
        layer_equal.append({"level": level, "repair_equal": repair_equal, "style_equal": style_equal})
        if not repair_equal:
            differences.append(f"repair:L{level}")
        if not style_equal:
            differences.append(f"style:L{level}")

    incremental_keys = {tile_id: record.cache_key for tile_id, record in incremental.records.items()}
    full_keys = {tile_id: record.cache_key for tile_id, record in full.records.items()}
    keys_equal = incremental_keys == full_keys
    if not keys_equal:
        differences.append("cache_keys")

    final_incremental = assemble_final(incremental.config, incremental.styled_levels)
    final_full = assemble_final(full.config, full.styled_levels)
    final_equal = np.array_equal(final_incremental, final_full)
    if not final_equal:
        differences.append("final_pixels")

    return {
        "equal": not differences,
        "differences": differences,
        "layer_equal": layer_equal,
        "cache_keys_equal": keys_equal,
        "final_pixels_equal": final_equal,
        "final_fingerprint_incremental": array_fingerprint(final_incremental),
        "final_fingerprint_full": array_fingerprint(final_full),
        "incremental_pending": list(incremental.pending),
        "full_pending": list(full.pending),
        "pending_equal": incremental.pending == full.pending,
    }


def open_snapshot_session(
    config: WorkbenchConfig,
    pixels: np.ndarray,
    image_fingerprint: str,
    rectangles,
    store: SnapshotStore,
):
    manifest = store.load_manifest()
    store.check_compatible(manifest)
    if manifest is None:
        return WorkbenchSession.fresh(config, pixels, image_fingerprint, rectangles), None
    products = store.load_products(manifest)
    session = WorkbenchSession.from_manifest(
        config, pixels, image_fingerprint, rectangles, manifest, products
    )
    return session, manifest
