"""Deterministic synthetic audio/video fixture generator.

No ffmpeg, fonts, clocks, random entropy, or external files are used.  The same
seed and geometry produce byte-identical Y4M, WAV, reference PCM, and truth JSON.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

import numpy as np

from .constants import (
    CHANNELS,
    CONTENT_ASSET_ID,
    DURATION_SECONDS,
    ENCODING_PARAMS,
    FPS,
    FRAME_COUNT,
    HEIGHT,
    SAMPLE_RATE,
    SAMPLES_PER_CHANNEL,
    SEED,
    SHOTS,
    TOTAL_AUDIO_SAMPLES,
    TRUTH_SCHEMA,
    WIDTH,
)
from .util import (
    atomic_binary_stream,
    atomic_write_canonical_json,
    hash_object,
    relative_posix,
    sha256_file,
)
from .y4m import Y4MFrame, iter_y4m, write_y4m


_FONT: dict[str, tuple[str, ...]] = {
    " ": ("00000",) * 7,
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
}

_SHOT_MESSAGES = (
    "CODEC QUALITY LAB",
    "PSNR SSIM SEGMENT",
    "A/V SYNC DRIFT",
    "VERSION COMPAT",
)


def _clip_rect(x: int, y: int, width: int, height: int, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_width, x + width)
    y1 = min(frame_height, y + height)
    return x0, y0, x1, y1


def _fill_rect(canvas: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    height_px, width_px = canvas.shape[:2]
    x0, y0, x1, y1 = _clip_rect(x, y, width, height, width_px, height_px)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)


def _draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: int,
    color: tuple[int, int, int],
) -> int:
    cursor = x
    color_array = np.asarray(color, dtype=np.uint8)
    for character in text:
        glyph = _FONT.get(character, _FONT[" "])
        for row_index, row in enumerate(glyph):
            for column_index, bit in enumerate(row):
                if bit == "1":
                    _fill_rect(
                        canvas,
                        cursor + column_index * scale,
                        y + row_index * scale,
                        scale,
                        scale,
                        color_array.tolist(),
                    )
        cursor += 6 * scale
    return cursor - x


def _text_width(text: str, scale: int) -> int:
    return max(0, len(text) * 6 * scale - scale)


def _background(frame_index: int, shot_id: int) -> np.ndarray:
    x = np.arange(WIDTH, dtype=np.float32)[None, :]
    y = np.arange(HEIGHT, dtype=np.float32)[:, None]
    palettes = (
        ((25, 28, 62), (74, 38, 116), (12, 92, 130)),
        ((24, 70, 48), (128, 72, 38), (32, 112, 104)),
        ((62, 24, 72), (28, 82, 118), (126, 46, 96)),
        ((30, 30, 34), (104, 82, 32), (54, 64, 88)),
    )
    base, diagonal, pulse = palettes[shot_id]
    local = frame_index - SHOTS[shot_id]["start_frame"]
    wave = np.sin(x * 0.012 + y * 0.021 + local * 0.045 + shot_id * 0.8)
    radial = np.sqrt((x - WIDTH * 0.5) ** 2 + (y - HEIGHT * 0.55) ** 2)
    vignette = np.clip(radial / 430.0, 0.0, 1.0)
    canvas = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for channel in range(3):
        value = (
            base[channel]
            + (diagonal[channel] - base[channel]) * (x / (WIDTH - 1)) * 0.62
            + (pulse[channel] - base[channel]) * (y / (HEIGHT - 1)) * 0.38
            + wave * 8.0
            - vignette * 24.0
        )
        canvas[:, :, channel] = np.clip(np.floor(value + 0.5), 0, 255).astype(np.uint8)
    return canvas


def _render_frame(frame_index: int) -> tuple[np.ndarray, dict]:
    shot = SHOTS[frame_index // 90 if frame_index < 180 else (2 if frame_index < 240 else 3)]
    shot_id = int(shot["shot_id"])
    local = frame_index - int(shot["start_frame"])
    canvas = _background(frame_index, shot_id)
    block_colors = (
        ((245, 186, 48), (30, 210, 188), (230, 72, 112)),
        ((88, 220, 116), (240, 126, 56), (88, 156, 244)),
        ((242, 214, 92), (178, 88, 222), (64, 212, 198)),
        ((214, 170, 82), (74, 154, 226), (226, 92, 92)),
    )[shot_id]
    block_specs = (
        ((40, 54, 142, 86), (430, 206, 154, 92), (242, 118, 96, 58)),
        ((512, 62, 104, 98), (84, 214, 148, 72), (232, 98, 132, 66)),
        ((84, 220, 132, 72), (364, 46, 112, 92), (248, 168, 86, 60)),
        ((190, 52, 78, 76), (410, 164, 150, 88), (58, 250, 122, 62)),
    )[shot_id]
    velocities = ((3, 1), (-2, 2), (1, 3), (2, -2))[shot_id]
    blocks: list[dict] = []
    for index, ((bx, by, bw, bh), color) in enumerate(zip(block_specs, block_colors, strict=True)):
        x = int((bx + velocities[0] * local + index * 17) % max(1, WIDTH - bw))
        y = int((by + velocities[1] * local + index * 11) % max(1, HEIGHT - bh))
        shadow_x = x + 5 + index
        shadow_y = y + 6 + index
        _fill_rect(canvas, shadow_x, shadow_y, bw, bh, (12, 12, 16))
        _fill_rect(canvas, x, y, bw, bh, color)
        inner = tuple(min(255, component + 34) for component in color)
        _fill_rect(canvas, x + 6, y + 6, max(1, bw - 12), max(1, bh - 12), inner)
        blocks.append({"block_id": index, "x": x, "y": y, "w": bw, "h": bh, "color": list(color)})

    message = _SHOT_MESSAGES[shot_id]
    scale = 3
    period = _text_width(message, scale) + 56
    offset = (local * 5 + shot_id * 19) % period
    text_y = 18 + (shot_id % 2) * 8
    for copy_index in range(-1, 3):
        text_x = WIDTH - offset + copy_index * period
        _draw_text(canvas, message, text_x + 2, text_y + 3, scale, (8, 8, 12))
        _draw_text(canvas, message, text_x, text_y, scale, (245, 241, 220))

    frame_label = f"FRAME {frame_index:03d}"
    _draw_text(canvas, frame_label, 18, HEIGHT - 31, 2, (8, 8, 12))
    _draw_text(canvas, frame_label, 16, HEIGHT - 33, 2, (238, 236, 220))
    _fill_rect(canvas, 0, HEIGHT - 4, WIDTH, 4, (236, 232, 214))
    _fill_rect(canvas, 0, 0, WIDTH, 3, (16, 16, 20))
    truth = {
        "frame_index": frame_index,
        "shot_id": shot_id,
        "timestamp_ms": round(frame_index * 1000.0 / FPS, 6),
        "blocks": blocks,
        "text": {"value": message, "scroll_offset_px": offset, "scale": scale},
        "hard_cut": frame_index in {shot_item["start_frame"] for shot_item in SHOTS},
    }
    return canvas, truth


def _rgb_to_yuv420(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = rgb.astype(np.float32)
    red = source[:, :, 0]
    green = source[:, :, 1]
    blue = source[:, :, 2]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    u_full = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    v_full = 0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    u = u_full.reshape(HEIGHT // 2, 2, WIDTH // 2, 2).mean(axis=(1, 3))
    v = v_full.reshape(HEIGHT // 2, 2, WIDTH // 2, 2).mean(axis=(1, 3))
    return tuple(np.clip(np.floor(plane + 0.5), 0, 255).astype(np.uint8) for plane in (y, u, v))  # type: ignore[return-value]


def _degrade_yuv(frame: Y4MFrame, frame_index: int, seed: int) -> Y4MFrame:
    y = frame.y.astype(np.float32)
    u = frame.u.astype(np.float32)
    v = frame.v.astype(np.float32)
    smooth_y = y.reshape(HEIGHT // 2, 2, WIDTH // 2, 2).mean(axis=(1, 3))
    smooth_y = np.repeat(np.repeat(smooth_y, 2, axis=0), 2, axis=1)
    quant = np.floor((y - 128.0) / 6.0 + 0.5) * 6.0 + 128.0
    yy = np.arange(HEIGHT, dtype=np.int32)[:, None]
    xx = np.arange(WIDTH, dtype=np.int32)[None, :]
    pattern = ((xx * 17 + yy * 11 + frame_index * 5 + seed) % 9 - 4).astype(np.float32) * 0.42
    encoded_y = 0.82 * quant + 0.18 * smooth_y + pattern
    smooth_u = u.reshape(HEIGHT // 4, 2, WIDTH // 4, 2).mean(axis=(1, 3))
    smooth_u = np.repeat(np.repeat(smooth_u, 2, axis=0), 2, axis=1)
    smooth_v = v.reshape(HEIGHT // 4, 2, WIDTH // 4, 2).mean(axis=(1, 3))
    smooth_v = np.repeat(np.repeat(smooth_v, 2, axis=0), 2, axis=1)
    encoded_u = 0.75 * (np.floor((u - 128.0) / 10.0 + 0.5) * 10.0 + 128.0) + 0.25 * smooth_u
    encoded_v = 0.75 * (np.floor((v - 128.0) / 10.0 + 0.5) * 10.0 + 128.0) + 0.25 * smooth_v
    return Y4MFrame(
        np.clip(np.floor(encoded_y + 0.5), 0, 255).astype(np.uint8),
        np.clip(np.floor(encoded_u + 0.5), 0, 255).astype(np.uint8),
        np.clip(np.floor(encoded_v + 0.5), 0, 255).astype(np.uint8),
    )


def _source_frames(seed: int, truth_frames: list[dict]) -> Iterator[Y4MFrame]:
    for frame_index in range(FRAME_COUNT):
        rgb, truth = _render_frame(frame_index)
        truth_frames.append(truth)
        y, u, v = _rgb_to_yuv420(rgb)
        yield Y4MFrame(y, u, v)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.ascontiguousarray(samples, dtype="<i2").tobytes()
    byte_rate = SAMPLE_RATE * CHANNELS * 2
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, CHANNELS, SAMPLE_RATE, byte_rate, CHANNELS * 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    with atomic_binary_stream(path) as handle:
        handle.write(header)
        handle.write(pcm)


def _make_audio() -> tuple[np.ndarray, np.ndarray]:
    sample_index = np.arange(SAMPLES_PER_CHANNEL, dtype=np.float64)
    time = sample_index / SAMPLE_RATE
    left = np.zeros_like(time)
    right = np.zeros_like(time)
    frequencies = ((220.0, 224.0), (330.0, 336.0), (440.0, 446.0), (660.0, 672.0))
    for shot_id, (left_frequency, right_frequency) in enumerate(frequencies):
        start = int(SHOTS[shot_id]["start_frame"] * SAMPLE_RATE / FPS)
        end = int((SHOTS[shot_id]["end_frame"] + 1) * SAMPLE_RATE / FPS)
        local_time = time[start:end] - time[start]
        envelope = 0.72 + 0.18 * np.sin(2.0 * np.pi * 1.5 * local_time + shot_id)
        left[start:end] = envelope * (
            0.30 * np.sin(2.0 * np.pi * left_frequency * local_time)
            + 0.06 * np.sin(2.0 * np.pi * left_frequency * 2.0 * local_time + 0.2)
        )
        right[start:end] = envelope * (
            0.28 * np.sin(2.0 * np.pi * right_frequency * local_time + 0.35)
            + 0.05 * np.sin(2.0 * np.pi * right_frequency * 1.5 * local_time + 0.7)
        )
    interleaved = np.empty(TOTAL_AUDIO_SAMPLES, dtype=np.float64)
    interleaved[0::2] = left
    interleaved[1::2] = right
    pcm = np.clip(np.floor(interleaved * 32767.0 + 0.5), -32768, 32767).astype(np.int16)
    reference = (pcm.astype(np.float32) / 32768.0).reshape(-1, CHANNELS)
    return pcm.reshape(-1, CHANNELS), reference


def generate_assets(workspace: Path, seed: int = SEED) -> dict:
    workspace = workspace.resolve()
    assets = workspace / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    source_path = assets / "source.y4m"
    encoded_path = assets / "encoded.y4m"
    wav_path = assets / "audio.wav"
    reference_path = assets / "audio_reference.f32le"
    truth_path = assets / "truth.json"

    truth_frames: list[dict] = []
    write_y4m(source_path, WIDTH, HEIGHT, FPS, _source_frames(seed, truth_frames))
    write_y4m(
        encoded_path,
        WIDTH,
        HEIGHT,
        FPS,
        (_degrade_yuv(frame, index, seed) for index, frame in enumerate(iter_y4m(source_path))),
    )

    pcm, reference = _make_audio()
    _write_wav(wav_path, pcm)
    atomic_binary_stream_path = reference_path
    with atomic_binary_stream(atomic_binary_stream_path) as handle:
        handle.write(np.ascontiguousarray(reference, dtype="<f4").tobytes())

    source_hash = sha256_file(source_path)
    encoded_hash = sha256_file(encoded_path)
    wav_hash = sha256_file(wav_path)
    reference_hash = sha256_file(reference_path)
    preview_indices = [int(index * (SAMPLES_PER_CHANNEL - 1) / 63) for index in range(64)]
    for frame_truth in truth_frames:
        sample_index = int(frame_truth["frame_index"] * SAMPLE_RATE / FPS)
        frame_truth["audio_reference_sample"] = {
            "sample_index": sample_index,
            "left": round(float(reference[sample_index, 0]), 9),
            "right": round(float(reference[sample_index, 1]), 9),
        }

    truth_core = {
        "schema_version": TRUTH_SCHEMA,
        "seed": seed,
        "asset_id": CONTENT_ASSET_ID,
        "video": {
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "duration_seconds": DURATION_SECONDS,
            "pixel_format": "yuv420p8",
            "source_file": relative_posix(source_path, workspace),
            "encoded_file": relative_posix(encoded_path, workspace),
        },
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "sample_count_per_channel": SAMPLES_PER_CHANNEL,
            "total_samples": TOTAL_AUDIO_SAMPLES,
            "encoding": "pcm_s16le",
            "wav_file": relative_posix(wav_path, workspace),
            "reference_file": relative_posix(reference_path, workspace),
            "reference_format": "float32le_interleaved",
            "reference_samples_preview": [
                {
                    "sample_index": index,
                    "left": round(float(reference[index, 0]), 9),
                    "right": round(float(reference[index, 1]), 9),
                }
                for index in preview_indices
            ],
        },
        "shots": [dict(shot) for shot in SHOTS],
        "shot_count": len(SHOTS),
        "frame_count": FRAME_COUNT,
        "shots_boundaries": [int(shot["start_frame"]) for shot in SHOTS],
        "frames": truth_frames,
        "encoding_params_snapshot": dict(ENCODING_PARAMS),
    }
    content_hash = hash_object(
        {
            "asset_id": CONTENT_ASSET_ID,
            "source_video_sha256": source_hash,
            "encoded_video_sha256": encoded_hash,
            "audio_sha256": wav_hash,
            "reference_audio_sha256": reference_hash,
            "truth_core": truth_core,
        }
    )
    truth = {
        **truth_core,
        "content_hash": content_hash,
        "file_hashes": {
            "source_video_sha256": source_hash,
            "encoded_video_sha256": encoded_hash,
            "audio_wav_sha256": wav_hash,
            "audio_reference_sha256": reference_hash,
        },
    }
    atomic_write_canonical_json(truth_path, truth)
    truth["file_hashes"]["truth_sha256"] = sha256_file(truth_path)
    summary = {
        "workspace": str(workspace),
        "content_hash": content_hash,
        "frame_count": FRAME_COUNT,
        "shot_count": len(SHOTS),
        "audio_sample_count_per_channel": SAMPLES_PER_CHANNEL,
        "audio_total_samples": TOTAL_AUDIO_SAMPLES,
        "files": {
            "source_y4m": {"path": str(source_path), "bytes": source_path.stat().st_size, "sha256": source_hash},
            "encoded_y4m": {"path": str(encoded_path), "bytes": encoded_path.stat().st_size, "sha256": encoded_hash},
            "audio_wav": {"path": str(wav_path), "bytes": wav_path.stat().st_size, "sha256": wav_hash},
            "audio_reference": {
                "path": str(reference_path),
                "bytes": reference_path.stat().st_size,
                "sha256": reference_hash,
            },
            "truth_json": {"path": str(truth_path), "bytes": truth_path.stat().st_size, "sha256": truth["file_hashes"]["truth_sha256"]},
        },
    }
    return summary
