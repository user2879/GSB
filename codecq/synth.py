"""Deterministic synthetic source material generator.

Fixed seed -> 10s 640x360@30fps video (translating color block, gradient,
scrolling text, hard shot cuts) + 48kHz stereo audio + per-frame ground truth.
Everything is a pure function of the config; no wall-clock, no global RNG.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from . import colors, util

PALETTES = [
    {"top": (16, 32, 64), "bottom": (200, 120, 40), "block": (220, 40, 40), "text": (255, 255, 255)},
    {"top": (8, 80, 40), "bottom": (40, 40, 120), "block": (40, 200, 220), "text": (255, 255, 160)},
    {"top": (90, 20, 90), "bottom": (20, 20, 20), "block": (240, 200, 40), "text": (160, 255, 160)},
    {"top": (30, 30, 30), "bottom": (160, 160, 190), "block": (200, 60, 160), "text": (255, 200, 200)},
]

BLOCK_W = 96
BLOCK_H = 64


def frame_count(cfg) -> int:
    return int(round(cfg["video"]["duration"] * cfg["video"]["fps"]))


def sample_count(cfg) -> int:
    return int(round(cfg["video"]["duration"] * cfg["audio"]["rate"]))


def shot_of(frame: int, shots) -> int:
    idx = 0
    for i, start in enumerate(shots):
        if frame >= start:
            idx = i
        else:
            break
    return idx


def shot_ranges(cfg):
    shots = list(cfg["shots"])
    n = frame_count(cfg)
    return [
        {"index": i, "start": s, "end": shots[i + 1] if i + 1 < len(shots) else n}
        for i, s in enumerate(shots)
    ]


def _bounce(pos: int, lo: int, hi: int) -> int:
    span = hi - lo
    m = pos % (2 * span)
    return lo + (m if m <= span else 2 * span - m)


def block_rect(frame: int, cfg):
    """Deterministic translating (bouncing) color-block rectangle."""
    w, h = cfg["video"]["width"], cfg["video"]["height"]
    s = shot_of(frame, cfg["shots"])
    vx, vy = 5 + 2 * s, 3 + s
    bx = _bounce(40 + vx * frame, 0, w - BLOCK_W)
    by = _bounce(30 + vy * frame, 0, h - BLOCK_H)
    return bx, by, BLOCK_W, BLOCK_H


def gen_frame(frame: int, cfg) -> np.ndarray:
    """Render one RGB frame (H,W,3) uint8. Pure function of (frame, cfg)."""
    w = cfg["video"]["width"]
    h = cfg["video"]["height"]
    fps = cfg["video"]["fps"]
    s = shot_of(frame, cfg["shots"])
    pal = PALETTES[s % len(PALETTES)]

    top = np.array(pal["top"], dtype=np.float64)
    bottom = np.array(pal["bottom"], dtype=np.float64)
    t = np.linspace(0.0, 1.0, h).reshape(h, 1, 1)
    img = top * (1.0 - t) + bottom * t
    img = np.broadcast_to(img, (h, w, 3)).copy()

    x = np.arange(w, dtype=np.float64).reshape(1, w, 1)
    img += 18.0 * np.sin(2.0 * np.pi * (x / w + frame / (fps * 3.0)))

    bx, by, bw, bh = block_rect(frame, cfg)
    img[by : by + bh, bx : bx + bw] = np.array(pal["block"], dtype=np.float64)

    rng = np.random.default_rng((cfg["seed"], frame))
    img += rng.integers(-2, 3, size=(h, w, 1))

    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)

    pil = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    text = f"SHOT {s} FRAME {frame:03d} CODECQ"
    tw = int(draw.textlength(text, font=font))
    tx = w - ((frame * 4) % (w + tw + 1))
    draw.text((tx, h - 22), text, fill=pal["text"], font=font)
    return np.asarray(pil)


def gen_audio(cfg) -> np.ndarray:
    """Synthesize 48kHz stereo int16 PCM, shape (2, N). Pure function of cfg."""
    rate = cfg["audio"]["rate"]
    n = sample_count(cfg)
    t = np.arange(n, dtype=np.float64) / rate
    left = 0.35 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 660.0 * t + 0.5)
    right = 0.35 * np.sin(2 * np.pi * 330.0 * t + 0.3) + 0.15 * np.sin(2 * np.pi * 990.0 * t)
    fps = cfg["video"]["fps"]
    beep_len = int(0.05 * rate)
    beep_t = np.arange(beep_len, dtype=np.float64) / rate
    beep = 0.25 * np.sin(2 * np.pi * 1000.0 * beep_t)
    for s in cfg["shots"][1:]:
        start = int(round(s / fps * rate))
        left[start : start + beep_len] += beep
        right[start : start + beep_len] += beep
    pcm = np.stack([left, right])
    pcm = np.clip(pcm, -1.0, 1.0)
    return np.rint(pcm * 32767.0).astype(np.int16)


def ground_truth(cfg, audio: np.ndarray, frame_hashes) -> dict:
    n = frame_count(cfg)
    return {
        "seed": cfg["seed"],
        "video": {
            "width": cfg["video"]["width"],
            "height": cfg["video"]["height"],
            "fps": cfg["video"]["fps"],
            "frames": n,
            "duration": cfg["video"]["duration"],
            "content_sha256": util.sha256_hex("".join(frame_hashes).encode("ascii")),
        },
        "audio": {
            "rate": cfg["audio"]["rate"],
            "channels": 2,
            "samples": int(audio.shape[1]),
            "sha256": util.hash_array(audio),
        },
        "shots": shot_ranges(cfg),
        "blocks": [
            dict(zip(("x", "y", "w", "h"), block_rect(i, cfg)), frame=i) for i in range(n)
        ],
    }


def write_wav(path, pcm: np.ndarray, rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = np.ascontiguousarray(pcm.T)  # (N, 2)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(interleaved.tobytes())


def generate_source(cfg, out_dir):
    """Generate all source material deterministically.

    Returns (frames_rgb, audio_pcm, truth). Writes audio.wav and truth.yaml
    under out_dir. The caller decides how to persist the inspection video.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = frame_count(cfg)
    frames = [gen_frame(i, cfg) for i in range(n)]
    audio = gen_audio(cfg)
    frame_hashes = [util.hash_array(f) for f in frames]
    truth = ground_truth(cfg, audio, frame_hashes)

    write_wav(out_dir / "audio.wav", audio, cfg["audio"]["rate"])
    (out_dir / "truth.yaml").write_text(
        yaml.safe_dump(truth, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    return frames, audio, truth
