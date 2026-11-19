"""Deterministic synthetic forensic source generation and manifest loading."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .avi import AVIVideoReader, DeterministicAVIWriter
from .config import AnalysisWindow, PipelineConfig, SamplingSpec, ScopedRect, WatermarkMask
from .errors import frame_format
from .util import atomic_write_json, canonical_bytes, read_json, sha256_file, sha256_json


SCENE_PERCENT_BOUNDARIES = (0, 18, 40, 66, 90)
SCENE_PALETTES = (
    ((34, 54, 92), (72, 132, 218), (64, 210, 242)),
    ((42, 88, 42), (158, 202, 82), (232, 186, 72)),
    ((84, 38, 118), (212, 92, 174), (64, 228, 198)),
    ((118, 72, 26), (242, 166, 58), (80, 174, 232)),
    ((28, 46, 64), (56, 132, 184), (226, 104, 92)),
)


def _seed_mix(seed: int, scene_index: int, channel: int) -> int:
    value = (
        (seed & 0xFFFFFFFF)
        + 0x9E3779B9 * (scene_index + 1)
        + 0x85EBCA6B * (channel + 1)
    ) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


@dataclass(frozen=True)
class SourceSpec:
    width: int = 640
    height: int = 360
    fps: int = 30
    duration_seconds: int = 60
    seed: int = 20261119
    gop_size: int = 30

    @property
    def frame_count(self) -> int:
        return self.fps * self.duration_seconds

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise frame_format(
                "source dimensions must be positive",
                width=self.width,
                height=self.height,
            )
        if self.fps <= 0 or self.duration_seconds <= 0:
            raise frame_format(
                "source timeline must be positive",
                fps=self.fps,
                duration_seconds=self.duration_seconds,
            )
        if self.gop_size <= 0:
            raise frame_format("gop_size must be positive", gop_size=self.gop_size)
        if self.width * self.height * 3 * self.frame_count > 0xFFFFFFFF:
            raise frame_format(
                "the default uncompressed AVI would exceed the RIFF limit",
                width=self.width,
                height=self.height,
                frame_count=self.frame_count,
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "gop_size": self.gop_size,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceSpec":
        return cls(
            width=int(value["width"]),
            height=int(value["height"]),
            fps=int(value["fps"]),
            duration_seconds=int(value["duration_seconds"]),
            seed=int(value["seed"]),
            gop_size=int(value["gop_size"]),
        )


@dataclass
class SourceBundle:
    directory: Path
    manifest: dict[str, Any]
    spec: SourceSpec
    video_path: Path
    truth_path: Path
    manifest_digest: str
    content_sha256: str
    video_sha256: str
    truth_sha256: str
    shot_boundaries: frozenset[int]
    shots: tuple[dict[str, Any], ...]

    @property
    def frame_count(self) -> int:
        return self.spec.frame_count

    @property
    def gop_count(self) -> int:
        return (self.frame_count + self.spec.gop_size - 1) // self.spec.gop_size

    @property
    def gop_boundaries(self) -> tuple[int, ...]:
        return tuple(range(0, self.frame_count, self.spec.gop_size))

    def open_reader(self) -> AVIVideoReader:
        return AVIVideoReader(self.video_path)

    def default_pipeline_config(self) -> PipelineConfig:
        masks = tuple(
            ScopedRect(
                x=int(shot["watermark"]["x"]),
                y=int(shot["watermark"]["y"]),
                width=int(shot["watermark"]["width"]),
                height=int(shot["watermark"]["height"]),
                frame_start=int(shot["start_frame"]),
                frame_end=int(shot["end_frame"]),
            )
            for shot in self.shots
        )
        return PipelineConfig(
            gop_size=self.spec.gop_size,
            sampling=SamplingSpec(
                (ScopedRect(0, 0, self.spec.width, self.spec.height, 0, self.frame_count),)
            ),
            watermark_mask=WatermarkMask(masks),
            analysis_window=AnalysisWindow(0, self.frame_count),
        )


def _scene_boundaries(frame_count: int) -> list[int]:
    values = {0}
    for percent in SCENE_PERCENT_BOUNDARIES[1:]:
        values.add((frame_count * percent) // 100)
    return sorted(value for value in values if 0 <= value < frame_count)


def _watermark_rect(
    seed: int, width: int, height: int, scene_index: int
) -> dict[str, int]:
    mix = _seed_mix(seed, scene_index, 0)
    wm_width = min(width, 148, max(8, width // 5 + int(mix % 7) * 2))
    wm_height = min(height, 36, max(6, height // 9 + int((mix >> 5) % 5)))
    x_factor = (82, 8, 72, 28, 84)[scene_index % 5] + int((mix >> 8) % 9) - 4
    y_factor = (6, 82, 5, 86, 80)[scene_index % 5] + int((mix >> 13) % 9) - 4
    x_factor = max(3, min(92, x_factor))
    y_factor = max(3, min(92, y_factor))
    x = ((width - wm_width) * x_factor) // 100
    y = ((height - wm_height) * y_factor) // 100
    return {"x": x, "y": y, "width": wm_width, "height": wm_height}


def _shots(spec: SourceSpec) -> list[dict[str, Any]]:
    starts = _scene_boundaries(spec.frame_count)
    shots: list[dict[str, Any]] = []
    for scene_index, start in enumerate(starts):
        end = starts[scene_index + 1] if scene_index + 1 < len(starts) else spec.frame_count
        shots.append(
            {
                "scene_index": scene_index,
                "start_frame": start,
                "end_frame": end,
                "watermark": _watermark_rect(
                    spec.seed, spec.width, spec.height, scene_index
                ),
            }
        )
    return shots


def _bounce(value: int, span: int) -> int:
    if span <= 0:
        return 0
    period = 2 * span
    position = value % period
    return position if position <= span else period - position


def _block_definitions(
    seed: int, width: int, height: int, scene_index: int
) -> list[dict[str, int]]:
    definitions: list[dict[str, int]] = []
    palette = SCENE_PALETTES[scene_index % len(SCENE_PALETTES)]
    for block_index in range(3):
        mix = _seed_mix(seed, scene_index, block_index + 1)
        block_width = max(
            4,
            min(
                width,
                width // 3,
                40 + scene_index * 9 + block_index * 14 + int(mix % 11),
            ),
        )
        block_height = max(
            4,
            min(
                height,
                height // 3,
                26 + block_index * 13 + scene_index * 4 + int((mix >> 4) % 9),
            ),
        )
        amplitude = max(
            8,
            min(
                height // 5,
                18 + block_index * 9 + scene_index * 3 + int((mix >> 9) % 11),
            ),
        )
        y_period = 70 + scene_index * 11 + block_index * 17 + int((mix >> 14) % 13)
        definitions.append(
            {
                "width": block_width,
                "height": block_height,
                "x_base": (
                    scene_index * 113 + block_index * 197 + 31 + int(mix % width)
                )
                % width,
                "y_base": max(2, ((height - block_height) * (10 + block_index * 28)) // 100),
                "x_speed": 1 + (scene_index + block_index + int((mix >> 18) % 4)) % 5,
                "y_amplitude": amplitude,
                "y_period": y_period,
                "y_phase": (scene_index * 23 + block_index * 41 + int(mix)) % y_period,
                "color": palette[
                    (block_index + 1 + int((mix >> 24) % len(palette))) % len(palette)
                ],
            }
        )
    return definitions


def _occlusion_definition(
    seed: int, width: int, height: int, scene_index: int
) -> dict[str, int]:
    mix = _seed_mix(seed, scene_index, 7)
    return {
        "width": max(24, min(width // 3, 70 + scene_index * 8 + int(mix % 9))),
        "height": max(18, min(height // 3, 38 + scene_index * 5 + int((mix >> 6) % 9))),
        "x_base": (scene_index * 71 + 17 + int(mix % width)) % width,
        "y_base": max(0, ((height - 60) * (18 + scene_index * 13)) // 100),
        "x_speed": 1 + (scene_index + int((mix >> 12) % 3)) % 3,
    }


def _scene_background(spec: SourceSpec, scene_index: int) -> np.ndarray:
    height, width = spec.height, spec.width
    palette = SCENE_PALETTES[scene_index % len(SCENE_PALETTES)]
    mix = _seed_mix(spec.seed, scene_index, 11)
    base = tuple(
        max(0, min(255, channel + int((mix >> (index * 5)) & 0x1F) - 15))
        for index, channel in enumerate(palette[0])
    )
    range_b = 12 + scene_index + int(mix % 9)
    range_g = 15 + scene_index * 2 + int((mix >> 6) % 9)
    range_r = 5 + scene_index + int((mix >> 11) % 9)
    checker_strength = 5 + int((mix >> 16) % 9)
    diagonal_modulus = 83 + int((mix >> 20) % 29)
    diagonal_width = 1 + int((mix >> 25) % 3)
    yy, xx = np.indices((height, width), dtype=np.int32)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = np.clip(
        base[0] + (xx * range_b) // max(1, width - 1) + (yy * 7) // max(1, height - 1),
        0,
        255,
    )
    frame[..., 1] = np.clip(
        base[1] + (yy * range_g) // max(1, height - 1),
        0,
        255,
    )
    frame[..., 2] = np.clip(
        base[2] + ((xx + yy) * range_r) // max(1, width + height - 2),
        0,
        255,
    )
    checker = (((xx // 32) + (yy // 32)) & 1).astype(np.uint8) * checker_strength
    frame = np.clip(frame.astype(np.uint16) + checker[..., None], 0, 255).astype(np.uint8)
    diagonal_offset = mix % diagonal_modulus
    diagonal = (
        (xx + 2 * yy + scene_index * 13 + diagonal_offset) % diagonal_modulus
    ) < diagonal_width
    frame[diagonal] = np.clip(frame[diagonal].astype(np.uint16) + 7, 0, 255).astype(np.uint8)
    return frame


def _fill_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    *,
    alpha: int = 255,
) -> None:
    frame_height, frame_width = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_width, x + width)
    y1 = min(frame_height, y + height)
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    if alpha >= 255:
        region[:] = color
        return
    color_array = np.asarray(color, dtype=np.uint16)
    blended = (region.astype(np.uint16) * (255 - alpha) + color_array * alpha) // 255
    region[:] = blended.astype(np.uint8)


def _draw_watermark(frame: np.ndarray, rect: dict[str, int], color: tuple[int, int, int]) -> None:
    x, y = rect["x"], rect["y"]
    width, height = rect["width"], rect["height"]
    _fill_rect(frame, x, y, width, height, color, alpha=58)
    _fill_rect(frame, x, y, width, 2, color, alpha=220)
    _fill_rect(frame, x, y + height - 2, width, 2, color, alpha=220)
    _fill_rect(frame, x, y, 2, height, color, alpha=220)
    _fill_rect(frame, x + width - 2, y, 2, height, color, alpha=220)
    region = frame[y + 3 : y + height - 3, x + 3 : x + width - 3]
    if region.size:
        region[::5, ::8] = color
        region[2::5, 3::8] = np.asarray(color, dtype=np.uint8)


def _iter_generated_frames(
    spec: SourceSpec, shots: list[dict[str, Any]], writer: DeterministicAVIWriter
) -> Iterator[tuple[int, np.ndarray, dict[str, Any]]]:
    scene_map: list[tuple[int, int]] = []
    for shot in shots:
        scene_map.extend([(int(shot["scene_index"]), int(shot["start_frame"]))] * (
            int(shot["end_frame"]) - int(shot["start_frame"])
        ))
    backgrounds = [_scene_background(spec, index) for index in range(len(shots))]
    block_defs = [
        _block_definitions(spec.seed, spec.width, spec.height, index)
        for index in range(len(shots))
    ]
    occlusion_defs = [
        _occlusion_definition(spec.seed, spec.width, spec.height, index)
        for index in range(len(shots))
    ]
    shot_boundaries = {int(shot["start_frame"]) for shot in shots}
    gop_boundaries = set(range(0, spec.frame_count, spec.gop_size))
    for frame_index in range(spec.frame_count):
        scene_index, shot_start = scene_map[frame_index]
        shot = shots[scene_index]
        frame = backgrounds[scene_index].copy()
        local_time = frame_index - shot_start
        blocks_truth: list[dict[str, Any]] = []
        for definition in block_defs[scene_index]:
            block_width = int(definition["width"])
            block_height = int(definition["height"])
            x = _bounce(
                int(definition["x_base"]) + int(definition["x_speed"]) * local_time,
                max(1, spec.width - block_width),
            )
            amplitude = int(definition["y_amplitude"])
            y = int(definition["y_base"]) + _bounce(
                local_time + int(definition["y_phase"]), amplitude
            )
            y = max(0, min(spec.height - block_height, y))
            color = tuple(int(channel) for channel in definition["color"])
            _fill_rect(frame, x, y, block_width, block_height, color, alpha=228)
            blocks_truth.append(
                {
                    "x": x,
                    "y": y,
                    "width": block_width,
                    "height": block_height,
                    "color_bgr": list(color),
                }
            )
        occlusion_def = occlusion_defs[scene_index]
        occlusion_width = int(occlusion_def["width"])
        occlusion_height = int(occlusion_def["height"])
        occlusion_x = _bounce(
            int(occlusion_def["x_base"])
            + int(occlusion_def["x_speed"]) * local_time,
            max(1, spec.width - occlusion_width),
        )
        occlusion_y = int(occlusion_def["y_base"])
        _fill_rect(
            frame,
            occlusion_x,
            occlusion_y,
            occlusion_width,
            occlusion_height,
            (18, 20, 24),
            alpha=172,
        )
        watermark = dict(shot["watermark"])
        _draw_watermark(frame, watermark, (188, 240, 255))
        truth = {
            "frame": frame_index,
            "scene_index": scene_index,
            "shot_boundary": frame_index in shot_boundaries,
            "gop_index": frame_index // spec.gop_size,
            "gop_boundary": frame_index in gop_boundaries,
            "blocks": blocks_truth,
            "occlusion": {
                "x": occlusion_x,
                "y": occlusion_y,
                "width": occlusion_width,
                "height": occlusion_height,
            },
            "watermark": watermark,
        }
        writer.add_frame(frame)
        yield frame_index, frame, truth


def _write_truth(path: Path, records: Iterator[dict[str, Any]]) -> tuple[str, int]:
    temporary = path.with_name(f".{path.name}.pending")
    digest = hashlib.sha256()
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    try:
        with temporary.open("wb") as handle:
            for record in records:
                line = canonical_bytes(record) + b"\n"
                handle.write(line)
                digest.update(line)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest(), count


def generate_source(
    source_dir: Path,
    spec: SourceSpec | None = None,
    *,
    overwrite: bool = False,
) -> SourceBundle:
    spec = spec or SourceSpec()
    spec.validate()
    source_dir = Path(source_dir)
    manifest_path = source_dir / "source_manifest.json"
    if manifest_path.exists() and not overwrite:
        return load_source(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    video_path = source_dir / "source.avi"
    truth_path = source_dir / "ground_truth.jsonl"
    shots = _shots(spec)
    pixel_digest = hashlib.sha256()
    writer = DeterministicAVIWriter(
        video_path, spec.width, spec.height, spec.fps, spec.frame_count
    )

    def records() -> Iterator[dict[str, Any]]:
        for _frame_index, frame, truth in _iter_generated_frames(spec, shots, writer):
            pixel_digest.update(frame.tobytes(order="C"))
            yield truth

    try:
        truth_sha256, truth_frames = _write_truth(truth_path, records())
        if truth_frames != spec.frame_count:
            raise frame_format(
                "truth frame count differs from the source timeline",
                expected=spec.frame_count,
                actual=truth_frames,
            )
        video_sha256, video_bytes = writer.close()
    except BaseException:
        writer.abort()
        raise
    manifest_without_hash = {
        "format": "forensic-synthetic-avi-v1",
        "generator": {"name": "forensic_workbench", "version": "1.0.0"},
        "spec": spec.to_dict(),
        "video": {
            "file": video_path.name,
            "sha256": video_sha256,
            "bytes": video_bytes,
            "frame_bytes": spec.width * spec.height * 3,
        },
        "truth": {
            "file": truth_path.name,
            "sha256": truth_sha256,
            "frame_count": truth_frames,
        },
        "shots": shots,
        "shot_boundaries": [int(shot["start_frame"]) for shot in shots],
        "gop": {
            "size": spec.gop_size,
            "boundaries": list(range(0, spec.frame_count, spec.gop_size)),
        },
        "pixel_content_sha256": pixel_digest.hexdigest(),
    }
    manifest = dict(manifest_without_hash)
    manifest["manifest_sha256"] = sha256_json(manifest_without_hash)
    atomic_write_json(manifest_path, manifest)
    return load_source(source_dir, verify_hashes=False)


def load_source(source_dir: Path, *, verify_hashes: bool = False) -> SourceBundle:
    source_dir = Path(source_dir)
    manifest_path = source_dir / "source_manifest.json"
    if not manifest_path.exists():
        raise frame_format(
            "source manifest is missing; generate the synthetic source first",
            path=str(manifest_path),
        )
    try:
        manifest = read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise frame_format(
            "source manifest cannot be read",
            path=str(manifest_path),
            reason=str(exc),
        ) from exc
    if not isinstance(manifest, dict):
        raise frame_format("source manifest root must be an object", path=str(manifest_path))
    declared_manifest_hash = manifest.get("manifest_sha256")
    manifest_without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    actual_manifest_hash = sha256_json(manifest_without_hash)
    if declared_manifest_hash != actual_manifest_hash:
        raise frame_format(
            "source manifest hash mismatch",
            path=str(manifest_path),
            expected=declared_manifest_hash,
            actual=actual_manifest_hash,
        )
    try:
        spec = SourceSpec.from_dict(manifest["spec"])
        video_info = manifest["video"]
        truth_info = manifest["truth"]
        video_path = source_dir / str(video_info["file"])
        truth_path = source_dir / str(truth_info["file"])
        shots = tuple(manifest["shots"])
        shot_boundaries = frozenset(int(value) for value in manifest["shot_boundaries"])
    except (KeyError, TypeError, ValueError) as exc:
        raise frame_format(
            "source manifest has an invalid structure",
            path=str(manifest_path),
            reason=str(exc),
        ) from exc
    for path, label in ((video_path, "video"), (truth_path, "truth")):
        if not path.exists():
            raise frame_format(
                f"source {label} file is missing",
                path=str(path),
                manifest=str(manifest_path),
            )
    if verify_hashes:
        actual_video_hash = sha256_file(video_path)
        if actual_video_hash != video_info.get("sha256"):
            raise frame_format(
                "source video hash mismatch",
                path=str(video_path),
                expected=video_info.get("sha256"),
                actual=actual_video_hash,
            )
        actual_truth_hash = sha256_file(truth_path)
        if actual_truth_hash != truth_info.get("sha256"):
            raise frame_format(
                "source truth hash mismatch",
                path=str(truth_path),
                expected=truth_info.get("sha256"),
                actual=actual_truth_hash,
            )
    with AVIVideoReader(video_path) as reader:
        reader.validate_metadata(
            width=spec.width,
            height=spec.height,
            fps=spec.fps,
            frame_count=spec.frame_count,
        )
    return SourceBundle(
        directory=source_dir,
        manifest=manifest,
        spec=spec,
        video_path=video_path,
        truth_path=truth_path,
        manifest_digest=str(declared_manifest_hash),
        content_sha256=str(manifest["pixel_content_sha256"]),
        video_sha256=str(video_info["sha256"]),
        truth_sha256=str(truth_info["sha256"]),
        shot_boundaries=shot_boundaries,
        shots=shots,
    )


def ensure_source(
    source_dir: Path,
    spec: SourceSpec | None = None,
    *,
    verify_hashes: bool = False,
) -> SourceBundle:
    manifest_path = Path(source_dir) / "source_manifest.json"
    if not manifest_path.exists():
        return generate_source(source_dir, spec)
    return load_source(source_dir, verify_hashes=verify_hashes)
