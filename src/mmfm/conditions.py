"""Deterministic multimodal condition loading and encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import content_digest, round_div, sha256_bytes
from .config import RenderConfig, Vocabulary, load_vocabulary, require_mapping, validate_prompt
from .errors import (
    E_CONDITION_REF,
    E_CONFIG_INVALID,
    E_DIMENSION,
    E_FRAME_COUNT,
    E_MOTION_DIMENSION,
    E_MOTION_INVALID,
    fail,
)
from .fixed import QuantProfile, quantize_milli_vector
from .png import decode_png_rgb


@dataclass(frozen=True)
class ConditionAsset:
    asset_id: str
    kind: str
    path: str
    sha256: str
    data: bytes


@dataclass(frozen=True)
class FrameInput:
    frame: int
    text_id: str
    prompt: dict[str, list[str]]
    image_id: str
    image_path: str
    image_sha256: str
    image_png: bytes
    motion_id: str
    motion_path: str
    motion_sha256: str
    motion_vectors: tuple[tuple[tuple[int, int], ...], ...]
    guidance_permille: int
    seed_offset: int

    def serialisable(self) -> dict[str, Any]:
        import base64

        return {
            "frame": self.frame,
            "text_id": self.text_id,
            "prompt": self.prompt,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "image_png_b64": base64.b64encode(self.image_png).decode("ascii"),
            "motion_id": self.motion_id,
            "motion_path": self.motion_path,
            "motion_sha256": self.motion_sha256,
            "motion_vectors": [
                [[int(dx), int(dy)] for dx, dy in row] for row in self.motion_vectors
            ],
            "guidance_permille": self.guidance_permille,
            "seed_offset": self.seed_offset,
        }

    @classmethod
    def from_serialised(cls, value: dict[str, Any]) -> "FrameInput":
        import base64

        return cls(
            frame=int(value["frame"]),
            text_id=str(value["text_id"]),
            prompt={"entities": list(value["prompt"]["entities"]), "attributes": list(value["prompt"]["attributes"])},
            image_id=str(value["image_id"]),
            image_path=str(value["image_path"]),
            image_sha256=str(value["image_sha256"]),
            image_png=base64.b64decode(value["image_png_b64"]),
            motion_id=str(value["motion_id"]),
            motion_path=str(value["motion_path"]),
            motion_sha256=str(value["motion_sha256"]),
            motion_vectors=tuple(
                tuple((int(pair[0]), int(pair[1])) for pair in row)
                for row in value["motion_vectors"]
            ),
            guidance_permille=int(value["guidance_permille"]),
            seed_offset=int(value["seed_offset"]),
        )


@dataclass
class ResolvedConditions:
    manifest: dict[str, Any]
    manifest_path: str
    vocab: Vocabulary
    frames: list[FrameInput]

    def serialisable(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "manifest_path": self.manifest_path,
            "vocab": {
                "version": self.vocab.version,
                "digest": self.vocab.digest,
                "terms": [
                    {
                        "term": term.term,
                        "kind": term.kind,
                        "index": term.index,
                        "weight": term.weight,
                    }
                    for term in self.vocab.terms
                ],
            },
            "frames": [frame.serialisable() for frame in self.frames],
        }

    @classmethod
    def from_serialised(cls, value: dict[str, Any]) -> "ResolvedConditions":
        from .config import VocabTerm

        vocab_raw = value["vocab"]
        vocab = Vocabulary(
            version=str(vocab_raw["version"]),
            terms=tuple(
                VocabTerm(
                    term=str(item["term"]),
                    kind=str(item["kind"]),
                    index=int(item["index"]),
                    weight=int(item["weight"]),
                )
                for item in vocab_raw["terms"]
            ),
            digest=str(vocab_raw["digest"]),
        )
        return cls(
            manifest=dict(value["manifest"]),
            manifest_path=str(value["manifest_path"]),
            vocab=vocab,
            frames=[FrameInput.from_serialised(item) for item in value["frames"]],
        )


def _load_asset_ref(
    raw: Any,
    *,
    kind: str,
    asset_id: str,
    base_dir: Path,
) -> tuple[str, bytes]:
    if isinstance(raw, str):
        relative = raw
    elif isinstance(raw, dict):
        relative = raw.get("path")
    else:
        fail(E_CONFIG_INVALID, "condition asset entry must be a path string or object", asset_id=asset_id)
    if not isinstance(relative, str) or not relative:
        fail(E_CONFIG_INVALID, "condition asset path is missing", asset_id=asset_id)
    path = Path(relative)
    if not path.is_absolute():
        path = base_dir / path
    try:
        data = path.read_bytes()
    except OSError as exc:
        fail(
            E_CONDITION_REF,
            "condition asset cannot be read",
            asset_id=asset_id,
            kind=kind,
            path=str(path),
            error=str(exc),
        )
    return str(path), data


def _validate_motion(
    data: bytes,
    *,
    config: RenderConfig,
    asset_id: str,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(E_MOTION_INVALID, "motion asset is not valid JSON", asset_id=asset_id, error=str(exc))
    if not isinstance(raw, dict):
        fail(E_MOTION_INVALID, "motion asset must be an object", asset_id=asset_id)
    if raw.get("width") != config.image.width or raw.get("height") != config.image.height:
        fail(
            E_MOTION_DIMENSION,
            "motion asset resolution differs from configured image resolution",
            asset_id=asset_id,
            expected=[config.image.width, config.image.height],
            actual=[raw.get("width"), raw.get("height")],
        )
    if raw.get("block_size") != config.motion.block_size:
        fail(
            E_MOTION_DIMENSION,
            "motion block size differs from configuration",
            asset_id=asset_id,
            expected=config.motion.block_size,
            actual=raw.get("block_size"),
        )
    vectors = raw.get("vectors")
    expected_rows, expected_cols = config.motion.grid_shape(config.image)
    if not isinstance(vectors, list) or len(vectors) != expected_rows:
        fail(
            E_MOTION_DIMENSION,
            "motion vector row count is incompatible with the configured grid",
            asset_id=asset_id,
            expected=expected_rows,
            actual=len(vectors) if isinstance(vectors, list) else None,
        )
    result: list[tuple[tuple[int, int], ...]] = []
    for row_index, row in enumerate(vectors):
        if not isinstance(row, list) or len(row) != expected_cols:
            fail(
                E_MOTION_DIMENSION,
                "motion vector column count is incompatible with the configured grid",
                asset_id=asset_id,
                row=row_index,
                expected=expected_cols,
                actual=len(row) if isinstance(row, list) else None,
            )
        parsed_row: list[tuple[int, int]] = []
        for col_index, vector in enumerate(row):
            if (
                not isinstance(vector, list)
                or len(vector) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in vector)
            ):
                fail(
                    E_MOTION_INVALID,
                    "motion vector must be a two-integer array",
                    asset_id=asset_id,
                    row=row_index,
                    column=col_index,
                )
            parsed_row.append((int(vector[0]), int(vector[1])))
        result.append(tuple(parsed_row))
    return tuple(result)


def resolve_conditions(
    config: RenderConfig,
    manifest_path: str | Path,
    *,
    vocabulary: Vocabulary | None = None,
) -> ResolvedConditions:
    path = Path(manifest_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        fail(E_CONDITION_REF, "conditions manifest cannot be read", path=str(path), error=str(exc))
    except json.JSONDecodeError as exc:
        fail(E_CONFIG_INVALID, "conditions manifest is invalid JSON", path=str(path), error=exc.msg)
    manifest = require_mapping(raw, "conditions manifest")
    if manifest.get("schema") not in {None, "mmfm-conditions-v1"}:
        fail(E_CONFIG_INVALID, "unsupported conditions manifest schema", schema=manifest.get("schema"))
    vocab = vocabulary or load_vocabulary(config.vocab_path, config.condition_dim)
    texts = require_mapping(manifest.get("texts"), "texts")
    images = require_mapping(manifest.get("images"), "images")
    motions = require_mapping(manifest.get("motions"), "motions")
    frames_raw = manifest.get("frames")
    if not isinstance(frames_raw, list):
        fail(E_CONFIG_INVALID, "conditions manifest frames must be a list")
    if len(frames_raw) != config.frame_count:
        fail(
            E_FRAME_COUNT,
            "conditions frame count differs from configured frame_count",
            expected=config.frame_count,
            actual=len(frames_raw),
        )
    base_dir = path.parent
    image_cache: dict[str, tuple[str, bytes]] = {}
    motion_cache: dict[str, tuple[str, bytes, tuple[tuple[tuple[int, int], ...], ...]]] = {}
    frame_inputs: list[FrameInput] = []
    seen_frames: set[int] = set()
    for item in frames_raw:
        frame_raw = require_mapping(item, "frame")
        frame_number = frame_raw.get("frame")
        if isinstance(frame_number, bool) or not isinstance(frame_number, int) or frame_number <= 0:
            fail(E_FRAME_COUNT, "frame numbers must be positive integers", value=frame_number)
        if frame_number in seen_frames:
            fail(E_FRAME_COUNT, "duplicate frame number", frame=frame_number)
        seen_frames.add(frame_number)
        text_id = frame_raw.get("text")
        image_id = frame_raw.get("image")
        motion_id = frame_raw.get("motion")
        if text_id not in texts:
            fail(E_CONDITION_REF, "frame references an unknown text condition", frame=frame_number, text=text_id)
        if image_id not in images:
            fail(E_CONDITION_REF, "frame references an unknown image condition", frame=frame_number, image=image_id)
        if motion_id not in motions:
            fail(E_CONDITION_REF, "frame references an unknown motion condition", frame=frame_number, motion=motion_id)
        prompt = validate_prompt(texts[text_id], vocab)
        if image_id not in image_cache:
            image_cache[image_id] = _load_asset_ref(images[image_id], kind="image", asset_id=image_id, base_dir=base_dir)
        image_path, image_bytes = image_cache[image_id]
        decode_png_rgb(
            image_bytes,
            width=config.image.width,
            height=config.image.height,
            mode=config.image.mode,
        )
        if motion_id not in motion_cache:
            motion_path, motion_bytes = _load_asset_ref(
                motions[motion_id], kind="motion", asset_id=motion_id, base_dir=base_dir
            )
            vectors = _validate_motion(motion_bytes, config=config, asset_id=motion_id)
            motion_cache[motion_id] = (motion_path, motion_bytes, vectors)
        motion_path, motion_bytes, vectors = motion_cache[motion_id]
        guidance = frame_raw.get("guidance_permille", config.guidance_permille_default)
        if isinstance(guidance, bool) or not isinstance(guidance, int) or guidance < 0:
            fail(
                E_CONFIG_INVALID,
                "frame guidance_permille must be a non-negative integer",
                frame=frame_number,
            )
        seed_offset = frame_raw.get("seed_offset", frame_number * 1009)
        if isinstance(seed_offset, bool) or not isinstance(seed_offset, int) or seed_offset < 0:
            fail(E_CONFIG_INVALID, "frame seed_offset must be a non-negative integer", frame=frame_number)
        frame_inputs.append(
            FrameInput(
                frame=frame_number,
                text_id=str(text_id),
                prompt=prompt,
                image_id=str(image_id),
                image_path=image_path,
                image_sha256=sha256_bytes(image_bytes),
                image_png=image_bytes,
                motion_id=str(motion_id),
                motion_path=motion_path,
                motion_sha256=sha256_bytes(motion_bytes),
                motion_vectors=vectors,
                guidance_permille=guidance,
                seed_offset=seed_offset,
            )
        )
    frame_inputs.sort(key=lambda frame: frame.frame)
    return ResolvedConditions(
        manifest=manifest,
        manifest_path=str(path),
        vocab=vocab,
        frames=frame_inputs,
    )


def encode_text(
    prompt: dict[str, list[str]],
    vocabulary: Vocabulary,
    profile: QuantProfile,
    condition_dim: int,
) -> dict[str, Any]:
    selected = set(prompt["entities"]) | set(prompt["attributes"])
    counts = [0] * condition_dim
    hits: list[dict[str, Any]] = []
    total_weight = 0
    for term in vocabulary.terms:
        if term.term not in selected:
            continue
        counts[term.index] += term.weight
        total_weight += term.weight
        hits.append(
            {
                "term": term.term,
                "kind": term.kind,
                "index": term.index,
                "weight": term.weight,
            }
        )
    if total_weight <= 0:
        fail(E_CONFIG_INVALID, "prompt has no weighted controlled-vocabulary terms")
    milli = [round_div(value * 1000, total_weight) for value in counts]
    if len(milli) != condition_dim:
        fail(E_DIMENSION, "text condition vector dimension differs from configuration", expected=condition_dim, actual=len(milli))
    quantized = quantize_milli_vector(milli, profile)
    return {
        "condition_type": "text",
        "hits": hits,
        "hit_positions": [hit["index"] for hit in hits],
        "total_weight": total_weight,
        "vector_milli": milli,
        "quantization": quantized,
        "vector": quantized["values"],
        "digest": content_digest(
            {
                "type": "text",
                "prompt": prompt,
                "vocab": vocabulary.digest,
                "vector": quantized["values"],
                "profile": profile.name,
            }
        ),
    }


def encode_image(
    png_bytes: bytes,
    *,
    config: RenderConfig,
    profile: QuantProfile,
    asset_id: str,
) -> dict[str, Any]:
    pixels = decode_png_rgb(
        png_bytes,
        width=config.image.width,
        height=config.image.height,
        mode=config.image.mode,
    )
    height, width, channels = pixels.shape
    if channels != 3:
        fail(E_DIMENSION, "image encoder requires RGB channels", channels=channels)
    tile = config.image.tile_size
    feature_sums = [0] * config.condition_dim
    feature_counts = [0] * config.condition_dim
    tiles: list[dict[str, Any]] = []
    tile_index = 0
    for y0 in range(0, height, tile):
        y1 = min(height, y0 + tile)
        for x0 in range(0, width, tile):
            x1 = min(width, x0 + tile)
            block = pixels[y0:y1, x0:x1, :].astype(np.int64, copy=False)
            count = block.shape[0] * block.shape[1]
            means = [
                round_div(int(block[:, :, channel].sum(dtype=np.int64)), count)
                for channel in range(3)
            ]
            if block.shape[1] > 1:
                grad_x = round_div(int(np.abs(np.diff(block, axis=1)).sum(dtype=np.int64)), block.shape[0] * (block.shape[1] - 1))
            else:
                grad_x = 0
            if block.shape[0] > 1:
                grad_y = round_div(int(np.abs(np.diff(block, axis=0)).sum(dtype=np.int64)), block.shape[1] * (block.shape[0] - 1))
            else:
                grad_y = 0
            features = [means[0], means[1], means[2], grad_x, grad_y]
            feature_milli = [round_div(value * 1000, 255) for value in features]
            for feature_index, value in enumerate(feature_milli):
                dimension = (tile_index * 5 + feature_index) % config.condition_dim
                feature_sums[dimension] += value
                feature_counts[dimension] += 1
            tiles.append(
                {
                    "index": tile_index,
                    "x": x0,
                    "y": y0,
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "mean_rgb": means,
                    "gradient_abs_mean_xy": [grad_x, grad_y],
                }
            )
            tile_index += 1
    milli = [
        round_div(feature_sums[index], feature_counts[index]) if feature_counts[index] else 0
        for index in range(config.condition_dim)
    ]
    quantized = quantize_milli_vector(milli, profile)
    return {
        "condition_type": "image",
        "asset_id": asset_id,
        "tile_size": tile,
        "tiles": tiles,
        "vector_milli": milli,
        "quantization": quantized,
        "vector": quantized["values"],
        "digest": content_digest(
            {
                "type": "image",
                "asset_sha256": sha256_bytes(png_bytes),
                "tile_size": tile,
                "vector": quantized["values"],
                "profile": profile.name,
            }
        ),
    }


def encode_motion(
    vectors: tuple[tuple[tuple[int, int], ...], ...],
    *,
    config: RenderConfig,
    profile: QuantProfile,
    asset_id: str,
) -> dict[str, Any]:
    sums = [0] * config.condition_dim
    counts = [0] * config.condition_dim
    flat_count = 0
    for row in vectors:
        for dx, dy in row:
            dx_dim = (flat_count * 2) % config.condition_dim
            dy_dim = (flat_count * 2 + 1) % config.condition_dim
            sums[dx_dim] += dx * 1000
            sums[dy_dim] += dy * 1000
            counts[dx_dim] += 1
            counts[dy_dim] += 1
            flat_count += 1
    milli = [
        round_div(sums[index], counts[index]) if counts[index] else 0
        for index in range(config.condition_dim)
    ]
    quantized = quantize_milli_vector(milli, profile)
    return {
        "condition_type": "motion",
        "asset_id": asset_id,
        "block_rows": len(vectors),
        "block_columns": len(vectors[0]) if vectors else 0,
        "vectors": [[[dx, dy] for dx, dy in row] for row in vectors],
        "vector_milli": milli,
        "quantization": quantized,
        "vector": quantized["values"],
        "digest": content_digest(
            {
                "type": "motion",
                "vectors": vectors,
                "vector": quantized["values"],
                "profile": profile.name,
            }
        ),
    }
