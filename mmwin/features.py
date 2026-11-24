from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

from .canonical import content_hash, stage_key
from .config import WindowConfig
from .events import Event
from .png import read_png


def round_fixed(value: float) -> int:
    """Deterministic half-up rounding, including for negative values."""
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(-value + 0.5))


def _hash_stream(seed: str) -> Iterable[bytes]:
    counter = 0
    while True:
        digest = hashlib.sha256(f"{seed}:{counter}".encode("utf-8")).digest()
        for offset in range(0, len(digest), 4):
            yield digest[offset : offset + 4]
        counter += 1


def deterministic_ints(seed: str, count: int, low: int, high: int) -> list[int]:
    if high <= low:
        raise ValueError("high must exceed low")
    span = high - low + 1
    values: list[int] = []
    for chunk in _hash_stream(seed):
        values.append(low + int.from_bytes(chunk, "big") % span)
        if len(values) == count:
            return values
    raise AssertionError("unreachable")


def projection_matrix(config: WindowConfig) -> list[list[int]]:
    raw_dim = config.visual_raw_dim
    out_dim = config.visual_feature_dim
    seed = stage_key(
        "visual-projection-v1",
        config.semantic_hash,
        config.quantization_level,
        config.fractional_bits,
        raw_dim,
        out_dim,
    )
    limit = max(1, config.quant_scale // 2)
    values = deterministic_ints(seed, raw_dim * out_dim, -limit, limit)
    return [values[row * out_dim : (row + 1) * out_dim] for row in range(raw_dim)]


def frame_feature(frame: Any, config: WindowConfig, matrix: list[list[int]]) -> dict[str, Any]:
    import numpy as np

    image = np.asarray(frame)
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
        raise ValueError("unsupported frame shape")
    rgb = image[:, :, :3].astype(np.float64) / 255.0
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    gradient_y, gradient_x = np.gradient(gray)
    magnitude = np.hypot(gradient_x, gradient_y) / math.sqrt(2.0)
    height, width = gray.shape
    row_edges = [int(round(i * height / config.visual_block_rows)) for i in range(config.visual_block_rows + 1)]
    col_edges = [int(round(i * width / config.visual_block_cols)) for i in range(config.visual_block_cols + 1)]
    raw: list[float] = []
    for row in range(config.visual_block_rows):
        y0, y1 = row_edges[row], row_edges[row + 1]
        for col in range(config.visual_block_cols):
            x0, x1 = col_edges[col], col_edges[col + 1]
            colour = rgb[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
            grad = magnitude[y0:y1, x0:x1]
            raw.extend(
                [
                    float(colour[0]),
                    float(colour[1]),
                    float(colour[2]),
                    float(grad.mean()),
                    float((grad > 0.12).mean()),
                ]
            )
    scale = config.quant_scale
    feature_q = [round_fixed(value * scale) for value in raw]
    projected: list[int] = []
    for out_index in range(config.visual_feature_dim):
        accumulator = 0
        for raw_index, value_q in enumerate(feature_q):
            accumulator += value_q * matrix[raw_index][out_index]
        projected.append(round_fixed(accumulator / scale))
    return {
        "raw_q": feature_q,
        "vector_q": projected,
        "raw_dim": len(raw),
        "output_dim": len(projected),
    }


def visual_stage_key(
    config: WindowConfig,
    start_ms: int,
    end_ms: int,
    frame_events: Iterable[Event],
    selected_ids: Iterable[str],
) -> str:
    refs = [
        {
            "event_id": event.event_id,
            "timestamp_ms": event.timestamp_ms,
            "fingerprint": event.fingerprint,
            "frame_index": event.data["frame_index"],
        }
        for event in sorted(frame_events, key=lambda item: item.event_id)
    ]
    return stage_key(
        "visual",
        config.stage_config_hash("visual"),
        start_ms,
        end_ms,
        refs,
        sorted(selected_ids),
    )


def select_frames(
    config: WindowConfig, start_ms: int, end_ms: int, frame_events: Iterable[Event]
) -> list[Event]:
    center = (start_ms + end_ms) / 2.0
    candidates = sorted(
        frame_events,
        key=lambda item: (
            abs(item.timestamp_ms - center),
            item.timestamp_ms,
            item.data["frame_index"],
            item.event_id,
        ),
    )
    selected = candidates[: config.visual_frames_per_window]
    return sorted(
        selected,
        key=lambda item: (item.data["frame_index"], item.timestamp_ms, item.event_id),
    )


def compute_visual(
    config: WindowConfig,
    sample_dir: Path,
    start_ms: int,
    end_ms: int,
    frame_events: Iterable[Event],
    matrix: list[list[int]],
) -> dict[str, Any]:
    frames = list(frame_events)
    selected = select_frames(config, start_ms, end_ms, frames)
    if not selected:
        return {
            "empty_visual": True,
            "selected_frame_ids": [],
            "selected_frame_indices": [],
            "vector_q": [0] * config.visual_feature_dim,
            "frame_features": [],
        }
    per_frame: list[dict[str, Any]] = []
    for event in selected:
        image = read_png(sample_dir / event.data["path"])
        feature = frame_feature(image, config, matrix)
        per_frame.append(
            {
                "event_id": event.event_id,
                "frame_index": event.data["frame_index"],
                **feature,
            }
        )
    vector: list[int] = []
    for index in range(config.visual_feature_dim):
        vector.append(
            round_fixed(sum(item["vector_q"][index] for item in per_frame) / len(per_frame))
        )
    return {
        "empty_visual": False,
        "selected_frame_ids": [event.event_id for event in selected],
        "selected_frame_indices": [event.data["frame_index"] for event in selected],
        "vector_q": vector,
        "frame_features": per_frame,
    }


def _event_bucket(event: Event, dimension: int) -> int:
    digest = hashlib.sha256(f"embedding:{event.fingerprint}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dimension


def _text_polarity(event: Event) -> float:
    explicit = event.data.get("polarity")
    if explicit is not None:
        return max(-1.0, min(1.0, float(explicit)))
    value = event.data["value"]
    if isinstance(value, bool):
        return 1.0 if value else -1.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.tanh(float(value))
    digest = hashlib.sha256(f"value:{value}".encode("utf-8")).digest()
    return 1.0 if digest[0] & 1 else -1.0


def evidence_vectors(
    config: WindowConfig,
    events: Iterable[Event],
) -> tuple[list[int], list[int], dict[str, Any]]:
    text_vector = [0] * config.text_feature_dim
    sensor_vector = [0] * config.text_feature_dim
    scale = config.quant_scale
    ordered = sorted(events, key=lambda item: (item.timestamp_ms, item.event_id))
    for event in ordered:
        if event.kind == "text":
            confidence = max(0.0, min(1.0, float(event.data.get("confidence", 1.0))))
            magnitude = _text_polarity(event) * confidence
            text_vector[_event_bucket(event, config.text_feature_dim)] += round_fixed(
                magnitude * scale
            )
        elif event.kind == "sensor":
            channel = config.sensor_channels[event.data["channel"]]
            span = channel.maximum - channel.minimum
            normalised = (float(event.data["value"]) - channel.minimum) / span
            normalised = max(-1.0, min(1.0, normalised * 2.0 - 1.0))
            sensor_vector[_event_bucket(event, config.text_feature_dim)] += round_fixed(
                normalised * scale
            )
    meta = {
        "text_event_count": sum(1 for event in ordered if event.kind == "text"),
        "sensor_event_count": sum(1 for event in ordered if event.kind == "sensor"),
        "text_dim": config.text_feature_dim,
        "sensor_dim": config.text_feature_dim,
    }
    return text_vector, sensor_vector, meta


def fusion_stage_key(
    config: WindowConfig,
    event_refs: list[dict[str, Any]],
    visual_key: str | None = None,
) -> str:
    return stage_key(
        "fusion",
        config.stage_config_hash("fusion"),
        event_refs,
        visual_key,
    )


def fuse_payload(
    config: WindowConfig,
    text_vector: list[int],
    sensor_vector: list[int],
    visual_vector: list[int],
) -> dict[str, Any]:
    text_q = sum(text_vector)
    sensor_q = sum(sensor_vector)
    visual_q = sum(visual_vector)
    total_q = text_q + sensor_q + visual_q
    # The softmax logits are already integer fixed-point values.  The second
    # class is the combined evidence score; the first class is the neutral 0.
    positive_logit_q = round_fixed(total_q / config.fusion_temperature)
    logits_q = [0, positive_logit_q]
    maximum = max(logits_q)
    negative_exp = math.exp((logits_q[0] - maximum) / config.quant_scale)
    positive_exp = math.exp((logits_q[1] - maximum) / config.quant_scale)
    confidence_q = round_fixed(
        config.quant_scale * positive_exp / (negative_exp + positive_exp)
    )
    payload = {
        "confidence_q": confidence_q,
        "logits_q": logits_q,
        "temperature_q": round_fixed(config.fusion_temperature * config.quant_scale),
        "contributions_q": {
            "text": text_q,
            "sensor": sensor_q,
            "visual": visual_q,
            "total": total_q,
        },
        "text_vector_q": text_vector,
        "sensor_vector_q": sensor_vector,
        "visual_vector_q": visual_vector,
    }
    return payload


def fuse(
    config: WindowConfig,
    text_vector: list[int],
    sensor_vector: list[int],
    visual_vector: list[int],
    event_refs: list[dict[str, Any]],
    visual_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    payload = fuse_payload(config, text_vector, sensor_vector, visual_vector)
    key = fusion_stage_key(
        config, event_refs, visual_key
    )
    return payload, key
