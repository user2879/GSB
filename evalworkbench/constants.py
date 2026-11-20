"""Stable constants shared by generators, carriers, and evaluators."""

from __future__ import annotations

VERSION = "1.0.0"

WIDTH = 640
HEIGHT = 360
FPS = 30
DURATION_SECONDS = 10
FRAME_COUNT = FPS * DURATION_SECONDS
SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLES_PER_CHANNEL = SAMPLE_RATE * DURATION_SECONDS
TOTAL_AUDIO_SAMPLES = SAMPLES_PER_CHANNEL * CHANNELS
SEED = 20261120

PROJECT_SCHEMA = "eval-project/2.0"
LEGACY_PROJECT_SCHEMA = "eval-project/1.0"
BINARY_SCHEMA = "metric-frame/2.0"
LEDGER_SCHEMA = "eval-ledger/1.0"
INJECTION_SCHEMA = "injection/1.0"
TRUTH_SCHEMA = "truth/1.0"
METRIC_ENCODING_VERSION = 2

SHOTS = (
    {"shot_id": 0, "start_frame": 0, "end_frame": 89, "label": "pan-left"},
    {"shot_id": 1, "start_frame": 90, "end_frame": 179, "label": "pan-right"},
    {"shot_id": 2, "start_frame": 180, "end_frame": 239, "label": "vertical"},
    {"shot_id": 3, "start_frame": 240, "end_frame": 299, "label": "outro"},
)

METRIC_POLICY = {
    "policy_version": "quality/2.0",
    "pixel_range": [0, 255],
    "pixel_format": "yuv420p8",
    "channel_weights": {"y": 1.0, "u": 0.0, "v": 0.0},
    "downsample": "none",
    "ssim_window": 8,
    "ssim_c1": 6.5025,
    "ssim_c2": 58.5225,
    "psnr_ceiling_db": 99.0,
    "aggregation": {
        "psnr": "mean_frame_db",
        "ssim": "mean_frame",
    },
}

ENCODING_PARAMS = {
    "snapshot_version": "encoding/1.0",
    "codec": "deterministic-yuv420p-shaped",
    "qstep": 6,
    "block_size": 4,
    "blend": 0.18,
    "gop_frames": 30,
    "bitrate_kbps": 1200,
}

CONTENT_ASSET_ID = "synthetic-av-deterministic-v1"

METRIC_FIELD_BITS = {
    "frame_index": 1 << 0,
    "shot_id": 1 << 1,
    "psnr_db": 1 << 2,
    "ssim": 1 << 3,
    "timestamp_ms": 1 << 4,
    "flags": 1 << 5,
    "chroma_psnr_db": 1 << 6,
}
V1_FIELD_BITMAP = 0x3F
V2_FIELD_BITMAP = 0x7F
KNOWN_FIELD_BITMAP = 0x7F
REQUIRED_FIELD_BITMAP = {1: V1_FIELD_BITMAP, 2: V2_FIELD_BITMAP}

DEFAULT_INJECTION = {
    "schema_version": INJECTION_SCHEMA,
    "dropped_frames": [],
    "timestamp_jumps": {},
    "binary_fault": None,
    "audio_declared_sample_rate": None,
    "audio_sample_offset": 0,
    "audio_gain": 1.0,
    "audio_tolerance_ms": 1.0,
    "timestamp_tolerance_ms": 1.0,
    "force_closure_mismatch": False,
}

TIMEBASE = {
    "master": "global_content_start",
    "video": "frame_index/FPS",
    "audio": "sample_index/SAMPLE_RATE",
    "fps": FPS,
    "sample_rate": SAMPLE_RATE,
    "units": "milliseconds",
}


def shot_by_frame(frame_index: int) -> int:
    for shot in SHOTS:
        if shot["start_frame"] <= frame_index <= shot["end_frame"]:
            return int(shot["shot_id"])
    raise ValueError(f"frame {frame_index} is outside all shots")
