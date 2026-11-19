"""Deterministic synthetic media generation and segment manifests."""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from .canonical import hash_object, relative_posix, sha256_file, write_canonical
from .config import EncoderConfig, MediaConfig


# A compact 5x7 bitmap font keeps the synthetic frame content independent of
# external fonts or platform text-rendering libraries.
_FONT: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
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
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
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
    ":": ("00000", "00100", "00100", "00000", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"),
}


@dataclass(frozen=True)
class GeneratedMedia:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    source_video: Path
    source_audio: Path


def _glyph_rows(character: str) -> tuple[str, ...]:
    return _FONT.get(character, _FONT[" "])


def _draw_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    cursor = x
    for character in text:
        for row_number, row in enumerate(_glyph_rows(character)):
            for column_number, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = cursor + column_number * scale
                y0 = y + row_number * scale
                x1 = min(width, x0 + scale)
                y1 = min(height, y0 + scale)
                if x0 < width and y0 < height and x1 > 0 and y1 > 0:
                    image[max(0, y0) : y1, max(0, x0) : x1] = color
        cursor += 6 * scale


class SyntheticMediaGenerator:
    """Generate source chunks, PCM audio, a full source, and a byte-stable manifest."""

    def __init__(self, config: MediaConfig, encoder: EncoderConfig, root: str | Path):
        config.validate()
        self.config = config
        self.encoder = encoder
        self.root = Path(root)
        self.source_dir = self.root / "source"
        self.slice_dir = self.root / "slices"
        self.manifest_dir = self.root / "manifests"
        self._base_frame = self._build_base_frame()
        self._palette = self._build_palette()

    def generate(self) -> GeneratedMedia:
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.slice_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        source_audio = self.source_dir / "audio_48k_stereo.wav"
        full_audio = self._synthesize_audio(0, self.config.total_audio_samples)
        self._write_wav(source_audio, full_audio)
        source_audio_hash = hashlib.sha256(full_audio.tobytes()).hexdigest()

        source_frames_hash = hashlib.sha256()
        slices: list[dict[str, Any]] = []
        slice_video_paths: list[Path] = []
        for slice_index in range(self.config.slice_count):
            frame_count = self.config.frame_count(slice_index)
            audio_start = int(round(slice_index * self.config.slice_seconds * self.config.sample_rate))
            audio_count = self.config.audio_samples(slice_index)
            audio = full_audio[audio_start : audio_start + audio_count].copy()

            video_name = f"segment_{slice_index:05d}_source.ts"
            audio_name = f"segment_{slice_index:05d}_audio.wav"
            video_path = self.slice_dir / video_name
            audio_path = self.slice_dir / audio_name
            video_hash = self._write_source_video(
                video_path,
                slice_index * self.config.frames_per_slice,
                frame_count,
                source_frames_hash,
            )
            self._write_wav(audio_path, audio)
            audio_hash = hashlib.sha256(audio.tobytes()).hexdigest()

            content_descriptor = {
                "index": slice_index,
                "start_pts": slice_index * self.config.frames_per_slice,
                "start_sample": audio_start,
                "frame_count": frame_count,
                "audio_samples": audio_count,
                "video_sha256": video_hash,
                "audio_sha256": audio_hash,
            }
            slices.append(
                {
                    "index": slice_index,
                    "video_file": relative_posix(video_path, self.root),
                    "audio_file": relative_posix(audio_path, self.root),
                    "start_pts": slice_index * self.config.frames_per_slice,
                    "end_pts": slice_index * self.config.frames_per_slice + frame_count,
                    "pts_time_base": f"1/{self.config.fps}",
                    "start_ms": slice_index * int(round(self.config.slice_seconds * 1000)),
                    "duration_ms": int(round(frame_count * 1000 / self.config.fps)),
                    "start_sample": audio_start,
                    "frame_count": frame_count,
                    "audio_samples": audio_count,
                    "video_sha256": video_hash,
                    "audio_sha256": audio_hash,
                    "content_sha256": hash_object(content_descriptor),
                }
            )
            slice_video_paths.append(video_path)

        source_video = self.source_dir / "source_720p30.ts"
        self._assemble_source(source_video, slice_video_paths)

        source_descriptor = {
            "seed": self.config.seed,
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "frame_count": self.config.total_frames,
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "audio_samples": self.config.total_audio_samples,
            "video_sha256": source_frames_hash.hexdigest(),
            "audio_sha256": source_audio_hash,
        }
        manifest = {
            "schema_version": 1,
            "config": self.config.to_dict(),
            "source": {
                "video_file": relative_posix(source_video, self.root),
                "audio_file": relative_posix(source_audio, self.root),
                "video_sha256": source_frames_hash.hexdigest(),
                "audio_sha256": source_audio_hash,
                "content_sha256": hash_object(source_descriptor),
                "frame_count": self.config.total_frames,
                "audio_samples": self.config.total_audio_samples,
            },
            "slices": slices,
        }
        manifest_path = self.manifest_dir / "source_manifest.json"
        write_canonical(manifest_path, manifest)
        return GeneratedMedia(
            root=self.root,
            manifest_path=manifest_path,
            manifest=manifest,
            source_video=source_video,
            source_audio=source_audio,
        )

    def _build_base_frame(self) -> np.ndarray:
        width, height = self.config.width, self.config.height
        x = np.arange(width, dtype=np.int32)[None, :]
        y = np.arange(height, dtype=np.int32)[:, None]
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = (18 + (90 * x) // max(1, width - 1) + (12 * y) // max(1, height - 1)) % 256
        image[:, :, 1] = (22 + (24 * x) // max(1, width - 1) + (92 * y) // max(1, height - 1)) % 256
        image[:, :, 2] = (48 + (84 * (width - 1 - x)) // max(1, width - 1) + (38 * y) // max(1, height - 1)) % 256
        return image

    def _build_palette(self) -> tuple[tuple[int, int, int], ...]:
        rng = np.random.default_rng(self.config.seed)
        colors = rng.integers(64, 240, size=(12, 3), dtype=np.int32)
        return tuple(tuple(int(channel) for channel in color) for color in colors)

    def _make_frame(self, global_frame: int) -> np.ndarray:
        width, height = self.config.width, self.config.height
        image = self._base_frame.copy()

        block_width = max(48, width // 7)
        block_height = max(36, height // 9)
        positions = (
            ((global_frame * 7 + 13) % max(1, width - block_width), (height // 8 + global_frame * 2) % max(1, height - block_height)),
            ((global_frame * 5 + width // 3) % max(1, width - block_width), (height // 2 + global_frame * 3) % max(1, height - block_height)),
            ((global_frame * 11 + width // 2) % max(1, width - block_width), (height // 5 + global_frame * 5) % max(1, height - block_height)),
            ((global_frame * 3 + width // 5) % max(1, width - block_width), (height * 2 // 3 + global_frame) % max(1, height - block_height)),
        )
        for index, (x0, y0) in enumerate(positions):
            color = self._palette[index]
            image[y0 : y0 + block_height, x0 : x0 + block_width] = color

        # A frame-index bar makes dropped/reordered frames obvious to a human.
        bar_width = max(1, width // (self.config.total_frames + 8))
        filled = int(round(width * (global_frame + 1) / self.config.total_frames))
        image[height - 14 : height - 4, :] = (24, 24, 24)
        image[height - 14 : height - 4, :filled] = (240, 210, 40)
        image[height - 14 : height - 4, filled : min(width, filled + bar_width)] = (255, 255, 255)

        slice_index = global_frame // self.config.frames_per_slice
        label = f"HLS SEQ {slice_index:04d} FRAME {global_frame:06d}"
        _draw_text(image, label, 24, 20, 3, (250, 250, 250))
        _draw_text(image, f"PTS {global_frame:06d}", 24, 56, 2, (80, 255, 190))

        scale = 3 if width >= 900 else 2
        scrolling = f"SYNTHETIC LIVE FRAME {global_frame:06d} SEQUENCE CHECK"
        text_width = len(scrolling) * 6 * scale
        scroll_x = int((global_frame * 8) % (width + text_width)) - text_width
        _draw_text(image, scrolling, scroll_x, height // 2 - 10, scale, (255, 245, 140))

        # Ten binary bars encode the low ten bits of the frame index.
        for bit in range(10):
            active = (global_frame >> bit) & 1
            x0 = width - 30 - bit * 10
            image[height // 2 - 35 : height // 2 + 35, x0 - 6 : x0] = (240, 240, 240) if active else (20, 20, 20)
        return image

    def _synthesize_audio(self, start_sample: int, sample_count: int) -> np.ndarray:
        sample_rate = self.config.sample_rate
        indices = np.arange(start_sample, start_sample + sample_count, dtype=np.float64)
        t = indices / sample_rate
        base_frequency = 220.0 + (self.config.seed % 7) * 10.0
        left = 0.22 * np.sin(2.0 * np.pi * base_frequency * t)
        left += 0.08 * np.sin(2.0 * np.pi * (2.0 * base_frequency) * t + 0.25)
        right = 0.20 * np.sin(2.0 * np.pi * (base_frequency * 1.5) * t + 0.5)
        right += 0.07 * np.sin(2.0 * np.pi * (base_frequency * 3.0) * t + 0.75)
        envelope = 0.72 + 0.28 * np.sin(2.0 * np.pi * 0.5 * t) ** 2
        pulse_phase = np.mod(indices, sample_rate // 2)
        pulse = np.where(pulse_phase < 48, 1.0 - pulse_phase / 48.0, 0.0)
        left = (left * envelope + 0.045 * pulse) * 32760.0
        right = (right * envelope - 0.045 * pulse) * 32760.0
        stereo = np.stack([left, right], axis=1)
        return np.clip(np.rint(stereo), -32768, 32767).astype(np.int16)

    def _write_wav(self, path: Path, samples: np.ndarray) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(self.config.channels)
            handle.setsampwidth(2)
            handle.setframerate(self.config.sample_rate)
            handle.writeframes(samples.astype("<i2", copy=False).tobytes())

    def _x264_options(self) -> dict[str, str]:
        return {
            "preset": self.encoder.h264_preset,
            "crf": str(self.encoder.source_crf),
            "g": str(max(1, self.config.frames_per_slice)),
            "keyint_min": str(max(1, self.config.frames_per_slice)),
            "scenecut": "0",
            "bf": "0",
            "threads": str(self.encoder.threads),
            "flags": "+bitexact",
            "x264-params": f"deterministic=1:threads={self.encoder.threads}",
        }

    def _write_source_video(
        self,
        path: Path,
        start_frame: int,
        frame_count: int,
        aggregate_hash: hashlib._Hash | None = None,
    ) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        container = av.open(str(path), mode="w", format="mpegts", options={"fflags": "+bitexact"})
        try:
            stream = container.add_stream("libx264", rate=self.config.fps)
            stream.width = self.config.width
            stream.height = self.config.height
            stream.pix_fmt = "yuv420p"
            stream.time_base = self.config.video_time_base
            stream.codec_context.max_b_frames = 0
            stream.options = self._x264_options()
            for local_index in range(frame_count):
                raw = self._make_frame(start_frame + local_index)
                payload = raw.tobytes()
                digest.update(payload)
                if aggregate_hash is not None:
                    aggregate_hash.update(payload)
                frame = av.VideoFrame.from_ndarray(raw, format="rgb24").reformat(format="yuv420p")
                frame.pts = local_index
                frame.time_base = self.config.video_time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        finally:
            container.close()
        return digest.hexdigest()

    def _assemble_source(
        self,
        source_path: Path,
        slice_video_paths: list[Path],
    ) -> None:
        container = av.open(str(source_path), mode="w", format="mpegts", options={"fflags": "+bitexact"})
        video_out = None
        try:
            for slice_index, slice_path in enumerate(slice_video_paths):
                input_container = av.open(str(slice_path), mode="r")
                try:
                    input_stream = input_container.streams.video[0]
                    if video_out is None:
                        video_out = container.add_stream_from_template(input_stream)
                    offset = int(
                        round(
                            Fraction(slice_index * self.config.frames_per_slice, self.config.fps)
                            / Fraction(input_stream.time_base)
                        )
                    )
                    for packet in input_container.demux(input_stream):
                        if packet.dts is None:
                            continue
                        packet.pts += offset
                        packet.dts += offset
                        packet.stream = video_out
                        container.mux(packet)
                finally:
                    input_container.close()

        finally:
            container.close()
