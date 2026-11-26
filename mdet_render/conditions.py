"""Condition parsing and deterministic fixed-point encoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import BusinessError
from .fixed import FixedPoint
from .png import RGBImage, load_png
from .util import digest_obj, quantize_array, read_json_file, round_div, sha256_hex


@dataclass(frozen=True)
class Vocabulary:
    terms: tuple[str, ...]
    entities: frozenset[str]
    attributes: frozenset[str]
    index: dict[str, int]
    digest: str

    @property
    def dimension(self) -> int:
        return len(self.terms)


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    entities: tuple[str, ...]
    attributes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"prompt_id": self.prompt_id, "entities": list(self.entities),
                "attributes": list(self.attributes)}


@dataclass(frozen=True)
class MotionTable:
    block_rows: int
    block_cols: int
    vector_dim: int
    seed: int
    values: np.ndarray
    digest: str
    source_path: Path


@dataclass(frozen=True)
class EncodedCondition:
    condition_type: str
    values: np.ndarray
    digest: str
    details: dict[str, Any]


def _string_list(value: Any, field_name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise BusinessError("E_VOCAB", f"{field_name} must be an array of non-empty strings", {"field": field_name})
    if not allow_empty and not value:
        raise BusinessError("E_VOCAB", f"{field_name} must not be empty", {"field": field_name})
    return tuple(value)


def load_vocabulary(path: str | Path, expected_dim: int) -> Vocabulary:
    path = Path(path)
    raw = read_json_file(path, code="E_VOCAB_JSON")
    if not isinstance(raw, dict):
        raise BusinessError("E_VOCAB", "vocabulary must be a JSON object")
    terms = _string_list(raw.get("terms"), "terms", allow_empty=False)
    entities = _string_list(raw.get("entities"), "entities", allow_empty=False)
    attributes = _string_list(raw.get("attributes"), "attributes", allow_empty=False)
    if len(terms) != len(set(terms)):
        raise BusinessError("E_VOCAB", "vocabulary terms must be unique")
    if len(entities) != len(set(entities)) or len(attributes) != len(set(attributes)):
        raise BusinessError("E_VOCAB", "entity and attribute vocabularies must be unique")
    term_set = set(terms)
    if not set(entities).issubset(term_set):
        raise BusinessError("E_VOCAB", "entity vocabulary contains unknown terms",
                            {"unknown": sorted(set(entities) - term_set)})
    if not set(attributes).issubset(term_set):
        raise BusinessError("E_VOCAB", "attribute vocabulary contains unknown terms",
                            {"unknown": sorted(set(attributes) - term_set)})
    if set(entities) & set(attributes):
        raise BusinessError("E_VOCAB", "entity and attribute sets must be disjoint")
    if len(terms) != expected_dim:
        raise BusinessError("E_VECTOR_DIM", "vocabulary dimension does not match condition_dim",
                            {"vocabulary_dim": len(terms), "condition_dim": expected_dim})
    digest = digest_obj({"terms": list(terms), "entities": list(entities), "attributes": list(attributes)})
    return Vocabulary(terms=terms, entities=frozenset(entities), attributes=frozenset(attributes),
                      index={term: index for index, term in enumerate(terms)}, digest=digest)


def parse_prompt(raw: Any, vocabulary: Vocabulary, *, prompt_id: str | None = None,
                 source: str = "<inline>") -> PromptSpec:
    if not isinstance(raw, dict):
        raise BusinessError("E_PROMPT_SCHEMA", "prompt must be a JSON object", {"source": source})
    entities = _string_list(raw.get("entities"), "entities")
    attributes = _string_list(raw.get("attributes"), "attributes")
    if not entities and not attributes:
        raise BusinessError("E_PROMPT_SCHEMA", "prompt must contain at least one entity or attribute",
                            {"source": source})
    for term in entities:
        if term not in vocabulary.index:
            raise BusinessError("E_PROMPT_TERM", "prompt entity is not in the controlled vocabulary",
                                {"term": term, "source": source})
        if term not in vocabulary.entities:
            raise BusinessError("E_PROMPT_CATEGORY", "prompt term is not registered as an entity",
                                {"term": term, "source": source})
    for term in attributes:
        if term not in vocabulary.index:
            raise BusinessError("E_PROMPT_TERM", "prompt attribute is not in the controlled vocabulary",
                                {"term": term, "source": source})
        if term not in vocabulary.attributes:
            raise BusinessError("E_PROMPT_CATEGORY", "prompt term is not registered as an attribute",
                                {"term": term, "source": source})
    pid = str(raw.get("prompt_id", prompt_id or "prompt"))
    if not pid:
        raise BusinessError("E_PROMPT_SCHEMA", "prompt_id must not be empty")
    return PromptSpec(prompt_id=pid, entities=entities, attributes=attributes)


def load_prompt(path: str | Path, vocabulary: Vocabulary) -> PromptSpec:
    path = Path(path)
    raw = read_json_file(path, code="E_PROMPT_JSON")
    return parse_prompt(raw, vocabulary, prompt_id=path.stem, source=str(path))


def encode_text(prompt: PromptSpec, vocabulary: Vocabulary, fixed: FixedPoint) -> EncodedCondition:
    counts = np.zeros(vocabulary.dimension, dtype=np.int64)
    hit_positions: list[int] = []
    for term in (*prompt.entities, *prompt.attributes):
        position = vocabulary.index[term]
        counts[position] += 1
        if position not in hit_positions:
            hit_positions.append(position)
    total = int(counts.sum())
    if total <= 0:
        raise BusinessError("E_PROMPT_SCHEMA", "prompt has no words after validation")
    values = np.array([fixed.from_ratio(int(count), total) for count in counts], dtype=np.int64)
    values, _ = quantize_array(values, fixed.quant_level)
    values = np.clip(values, fixed.low, fixed.high)
    digest = sha256_hex(b"text-bow-v1|" + values.astype("<i8").tobytes())
    return EncodedCondition(
        condition_type="text",
        values=values,
        digest=digest,
        details={
            "vocabulary_digest": vocabulary.digest,
            "hit_positions": sorted(hit_positions),
            "hit_terms": [vocabulary.terms[index] for index in sorted(hit_positions)],
            "normalization": "term_count/total_terms",
            "prompt": prompt.as_dict(),
        },
    )


def encode_image(image: RGBImage, grid_rows: int, grid_cols: int, fixed: FixedPoint) -> EncodedCondition:
    height, width = image.pixels.shape[:2]
    if height % grid_rows or width % grid_cols:
        raise BusinessError("E_VECTOR_DIM", "image dimensions are not divisible by the block grid")
    cell_h = height // grid_rows
    cell_w = width // grid_cols
    pixels = image.pixels.astype(np.int64)
    values: list[int] = []
    cells: list[dict[str, Any]] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0, x0 = row * cell_h, col * cell_w
            cell = pixels[y0:y0 + cell_h, x0:x0 + cell_w]
            means = [round_div(int(cell[..., channel].sum()), cell_h * cell_w) for channel in range(3)]
            if cell_h > 1 and cell_w > 1:
                dx = np.abs(np.diff(cell, axis=1)).sum(axis=(0, 1, 2)) if cell_w > 1 else np.int64(0)
                dy = np.abs(np.diff(cell, axis=0)).sum(axis=(0, 1, 2)) if cell_h > 1 else np.int64(0)
                dx_den = cell_h * max(1, cell_w - 1) * 3
                dy_den = cell_w * max(1, cell_h - 1) * 3
                grad_x = round_div(int(dx), dx_den)
                grad_y = round_div(int(dy), dy_den)
            else:
                grad_x = grad_y = 0
            stats = [*means, grad_x, grad_y]
            quantized = [fixed.from_ratio(stat, 255) for stat in stats]
            values.extend(quantized)
            cells.append({"row": row, "col": col, "mean_rgb": means,
                          "gradient_xy": [grad_x, grad_y], "fixed": quantized})
    array = np.array(values, dtype=np.int64)
    array, _ = quantize_array(array, fixed.quant_level)
    array = np.clip(array, fixed.low, fixed.high)
    digest = sha256_hex(b"image-block-stats-v1|" + array.astype("<i8").tobytes())
    return EncodedCondition(
        condition_type="image",
        values=array,
        digest=digest,
        details={
            "image_digest": image.digest,
            "width": width,
            "height": height,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "statistics": "block mean RGB followed by block mean absolute gradients",
            "cells": cells,
        },
    )


def load_motion_table(path: str | Path, expected_rows: int, expected_cols: int,
                      expected_dim: int) -> MotionTable:
    path = Path(path)
    raw = read_json_file(path, code="E_MOTION_JSON")
    if not isinstance(raw, dict):
        raise BusinessError("E_MOTION", "motion table must be a JSON object")
    rows = raw.get("block_rows")
    cols = raw.get("block_cols")
    vector_dim = raw.get("vector_dim", 2)
    seed = raw.get("seed")
    blocks = raw.get("blocks")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise BusinessError("E_MOTION_ROWS", "motion table block_rows must be a positive integer")
    if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
        raise BusinessError("E_MOTION_ROWS", "motion table block_cols must be a positive integer")
    if rows != expected_rows or cols != expected_cols:
        raise BusinessError("E_MOTION_ROWS", "motion table block dimensions are incompatible with config",
                            {"table": [rows, cols], "config": [expected_rows, expected_cols]})
    if vector_dim != 2:
        raise BusinessError("E_MOTION_DIM", "motion vectors must have dimension 2", {"vector_dim": vector_dim})
    if rows * cols * vector_dim != expected_dim:
        raise BusinessError("E_MOTION_DIM", "motion vector table dimension is incompatible with condition_dim",
                            {"motion_dim": rows * cols * vector_dim, "condition_dim": expected_dim})
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise BusinessError("E_MOTION", "motion table seed must be an integer")
    if not isinstance(blocks, list) or len(blocks) != rows:
        raise BusinessError("E_MOTION_ROWS", "motion table row count does not match block_rows",
                            {"expected": rows, "actual": len(blocks) if isinstance(blocks, list) else None})
    values: list[int] = []
    for row_index, row in enumerate(blocks):
        if not isinstance(row, list) or len(row) != cols:
            raise BusinessError("E_MOTION_ROWS", "motion table block row has an invalid column count",
                                {"row": row_index, "expected": cols})
        for col_index, vector in enumerate(row):
            if (not isinstance(vector, list) or len(vector) != 2
                    or any(isinstance(component, bool) or not isinstance(component, int)
                           or not (-1024 <= component <= 1024) for component in vector)):
                raise BusinessError("E_MOTION_DIM", "motion vector must contain two integers in -1024..1024",
                                    {"row": row_index, "col": col_index, "value": vector})
            values.extend(int(component) for component in vector)
    array = np.array(values, dtype=np.int64)
    digest = sha256_hex(b"motion-table-v1|" + repr((rows, cols, seed)).encode("ascii") + b"|"
                        + array.astype("<i8").tobytes())
    return MotionTable(block_rows=rows, block_cols=cols, vector_dim=2, seed=seed,
                       values=array.reshape(rows, cols, 2), digest=digest, source_path=path)


def encode_motion(table: MotionTable, fixed: FixedPoint, *, frame: int) -> EncodedCondition:
    vector = []
    for row in range(table.block_rows):
        for col in range(table.block_cols):
            for component in table.values[row, col]:
                vector.append(fixed.from_ratio(int(component), 255))
    array = np.array(vector, dtype=np.int64)
    array, residual = quantize_array(array, fixed.quant_level)
    array = np.clip(array, fixed.low, fixed.high)
    frame_mix = (frame * 17) % max(1, table.block_rows * table.block_cols)
    # A frame-independent motion table is intentionally reused, but the frame
    # binding is recorded so per-frame cache entries and invalidation are exact.
    digest = sha256_hex(b"motion-fixed-v1|" + repr((frame, frame_mix)).encode("ascii") + b"|"
                        + array.astype("<i8").tobytes())
    return EncodedCondition(
        condition_type="motion",
        values=array,
        digest=digest,
        details={
            "motion_table_digest": table.digest,
            "seed": table.seed,
            "block_rows": table.block_rows,
            "block_cols": table.block_cols,
            "frame_binding": frame,
            "quantization_residual": int(residual.sum()),
        },
    )


def load_image_condition(path: str | Path, width: int, height: int, pixel_mode: str) -> RGBImage:
    info, image = load_png(path)
    if info.width != width or info.height != height:
        raise BusinessError("E_PNG_RESOLUTION", "PNG resolution does not match configuration",
                            {"path": str(path), "expected": [width, height],
                             "actual": [info.width, info.height]})
    if info.pixel_mode != pixel_mode:
        raise BusinessError("E_PNG_PIXEL_MODE", "PNG pixel mode does not match configuration",
                            {"path": str(path), "expected": pixel_mode, "actual": info.pixel_mode})
    return image


def condition_hits(values: np.ndarray, limit: int = 12) -> list[int]:
    nonzero = np.flatnonzero(np.asarray(values) != 0)
    return [int(index) for index in nonzero[:limit]]


def vector_from_payload(payload: dict[str, Any]) -> np.ndarray:
    try:
        return np.asarray(payload["values"], dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        raise BusinessError("E_CACHE_ARTIFACT", "condition vector payload is invalid", {"error": str(exc)}) from exc
