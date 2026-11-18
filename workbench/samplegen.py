"""Deterministic synthetic frame-sequence sample generation (no external downloads)."""

import json
import os

import numpy as np

from .pngcodec import encode_png

WIDTH = 64
HEIGHT = 48
FPS = 25
FRAME_INTERVAL_MS = 1000 // FPS
SEED_BASE = 20261118

# (name, color, top-left (y, x), size (h, w), appear_frame, disappear_frame,
#  move_dy_per_frame, move_dx_per_frame, color_switch_frame, color_after)
PRIMITIVE_SCRIPTS = (
    ("rect_a", (200, 40, 40), (6, 6), (10, 12), 2, 30, 0, 1, None, None),
    ("rect_b", (40, 200, 60), (28, 40), (8, 10), 5, 34, 0, 0, 18, (220, 200, 40)),
    ("rect_c", (60, 80, 210), (20, 16), (12, 8), 8, 26, 1, 0, None, None),
    ("rect_d", (210, 120, 30), (34, 4), (6, 8), 12, 38, 0, 0, None, None),
)


def _render_frame(seq_index: int, frame_index: int) -> np.ndarray:
    rng = np.random.RandomState(SEED_BASE + seq_index * 100003 + frame_index)
    frame = rng.randint(0, 12, size=(HEIGHT, WIDTH, 3), dtype=np.uint8) + 12
    for name, color, (y0, x0), (h, w), appear, disappear, dy, dx, switch, after in PRIMITIVE_SCRIPTS:
        # Deterministic per-sequence variation of the script.
        shift = (seq_index * 3 + len(name)) % 5
        a = appear + shift
        d = min(disappear + shift, 39)
        if not (a <= frame_index < d):
            continue
        offset = frame_index - a
        y = y0 + dy * offset + (seq_index % 3)
        x = x0 + dx * offset + (seq_index % 2)
        c = after if (switch is not None and frame_index >= a + (switch - appear)) else color
        y2, x2 = min(y + h, HEIGHT), min(x + w, WIDTH)
        if 0 <= y < HEIGHT and 0 <= x < WIDTH and y2 > y and x2 > x:
            frame[y:y2, x:x2] = c
    return frame.astype(np.uint8)


def generate_samples(samples_dir: str, num_sequences: int = 4, num_frames: int = 40) -> list:
    """Generate deterministic PNG frame sequences plus per-sequence manifests."""
    os.makedirs(samples_dir, exist_ok=True)
    names = []
    for seq_index in range(num_sequences):
        name = f"seq_{seq_index:02d}"
        names.append(name)
        seq_dir = os.path.join(samples_dir, name)
        os.makedirs(seq_dir, exist_ok=True)
        manifest_lines = []
        for frame_index in range(num_frames):
            frame = _render_frame(seq_index, frame_index)
            with open(os.path.join(seq_dir, f"frame_{frame_index:04d}.png"), "wb") as fh:
                fh.write(encode_png(frame))
            timestamp = frame_index * FRAME_INTERVAL_MS
            # Deterministically omit timestamps for a few frames to exercise
            # fixed-interval interpolation.
            if (seq_index + frame_index) % 17 == 5:
                manifest_lines.append(f"{frame_index}")
            else:
                manifest_lines.append(f"{frame_index} {timestamp}")
        with open(os.path.join(seq_dir, "manifest.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(manifest_lines) + "\n")
    return names


def generate_configs(configs_dir: str) -> None:
    os.makedirs(configs_dir, exist_ok=True)
    event_params = {
        "change_pixel_ratio_threshold": 0.01,
        "color_match_tolerance": 24,
        "min_event_duration_ms": 120,
        "min_gap_ms": 80,
        "event_categories": ["appear", "disappear", "translate", "color_switch"],
        "change_pixel_value_threshold": 16,
        "track_max_gap_frames": 2,
        "block_size_frames": 8,
        "hist_bins": 4,
    }
    schedule = {
        "max_concurrent_sequences": 3,
        "global_frame_budget": 100000,
        "per_sequence_frame_quota": 1000,
        "frame_block_residency_budget": 6,
        "snapshot_commit_interval_frames": 16,
    }
    with open(os.path.join(configs_dir, "event_params.json"), "w", encoding="utf-8") as fh:
        json.dump(event_params, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(configs_dir, "schedule.json"), "w", encoding="utf-8") as fh:
        json.dump(schedule, fh, indent=2, sort_keys=True)
        fh.write("\n")
