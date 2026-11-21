"""Load and validate samples, descriptions, dual label lists and config.

Produces per-source failures with business codes; the offending source data
is dropped for that sample while other sources keep flowing downstream.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import (
    E_BOX_OUT_OF_RANGE,
    E_LABEL_ENUM,
    E_ORDER_GAP,
    E_SAMPLE_INVALID,
    failure,
)

LABEL_SOURCES = (("label_a", "labels_a.jsonl"), ("label_b", "labels_b.jsonl"))


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _load_images(images_dir, config, samples, failures):
    width, height = config.image_size
    if not images_dir.is_dir():
        return
    for png in sorted(images_dir.glob("*.png")):
        sample_id = png.stem
        entry = samples.setdefault(sample_id, {})
        try:
            with Image.open(png) as img:
                rgb = img.convert("RGB")
                if rgb.size != (width, height):
                    raise ValueError(f"image size {rgb.size} != {(width, height)}")
                entry["image"] = np.asarray(rgb, dtype=np.uint8).copy()
        except Exception as exc:  # truncated header / decode failure / size mismatch
            failures.append(failure(sample_id, "image", E_SAMPLE_INVALID, exc))
            entry["image"] = None


def _load_descriptions(desc_dir, config, samples, failures):
    if not desc_dir.is_dir():
        return
    for path in sorted(desc_dir.glob("*.json")):
        sample_id = path.stem
        try:
            desc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(failure(sample_id, "description", E_SAMPLE_INVALID, exc))
            continue
        if desc.get("sample_id") != sample_id:
            failures.append(
                failure(sample_id, "description", E_SAMPLE_INVALID, "sample_id mismatch")
            )
            continue
        valid = []
        for index, assertion in enumerate(desc.get("assertions", [])):
            category = assertion.get("category")
            color = assertion.get("color")
            if category not in config.categories or color not in config.colors:
                failures.append(
                    failure(
                        sample_id,
                        "description",
                        E_LABEL_ENUM,
                        f"assertion {index}: {category}/{color} not in enum",
                    )
                )
                continue
            count = assertion.get("count")
            if not _is_int(count) or count < 0:
                failures.append(
                    failure(
                        sample_id,
                        "description",
                        E_SAMPLE_INVALID,
                        f"assertion {index}: count must be a non-negative integer",
                    )
                )
                continue
            valid.append(
                {
                    "index": index,
                    "category": category,
                    "color": color,
                    "position": assertion.get("position", ""),
                    "count": count,
                }
            )
        samples.setdefault(sample_id, {})["description"] = valid


def _load_labels(data_dir, config, samples, failures):
    width, height = config.image_size
    for source, filename in LABEL_SOURCES:
        path = data_dir / filename
        if not path.exists():
            continue
        per_sample = {}
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception as exc:
                failures.append(failure("?", source, E_SAMPLE_INVALID, f"line {line_no}: {exc}"))
                continue
            sample_id = entry.get("sample_id", "?")
            category = entry.get("category")
            color = entry.get("color")
            if category not in config.categories or color not in config.colors:
                failures.append(
                    failure(sample_id, source, E_LABEL_ENUM, f"{category}/{color} not in enum")
                )
                continue
            bbox = entry.get("bbox")
            bbox_ok = (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(_is_int(v) for v in bbox)
                and 0 <= bbox[0] <= bbox[2] <= width
                and 0 <= bbox[1] <= bbox[3] <= height
            )
            if not bbox_ok:
                failures.append(
                    failure(sample_id, source, E_BOX_OUT_OF_RANGE, f"bbox {bbox} out of range")
                )
                continue
            order = entry.get("order")
            if not _is_int(order):
                failures.append(failure(sample_id, source, E_SAMPLE_INVALID, "order not int"))
                continue
            per_sample.setdefault(sample_id, []).append(
                {"category": category, "color": color, "bbox": list(bbox), "order": order}
            )
        for sample_id, entries in per_sample.items():
            entries.sort(key=lambda e: (e["order"], e["category"], e["color"]))
            kept = entries
            for i in range(1, len(entries)):
                if entries[i]["order"] != entries[i - 1]["order"] + 1:
                    failures.append(
                        failure(
                            sample_id,
                            source,
                            E_ORDER_GAP,
                            f"order {entries[i - 1]['order']} -> {entries[i]['order']}",
                        )
                    )
                    kept = entries[:i]
                    break
            samples.setdefault(sample_id, {})[source] = kept


def load_inputs(data_dir, config):
    """Return (samples, failures).

    samples: {sample_id: {"image": ndarray|None, "description": [...],
                          "label_a": [...], "label_b": [...]}}
    """
    data_dir = Path(data_dir)
    samples = {}
    failures = []
    _load_images(data_dir / "images", config, samples, failures)
    _load_descriptions(data_dir / "descriptions", config, samples, failures)
    _load_labels(data_dir, config, samples, failures)
    for sample_id in samples:
        samples[sample_id].setdefault("image", None)
        samples[sample_id].setdefault("description", [])
        samples[sample_id].setdefault("label_a", [])
        samples[sample_id].setdefault("label_b", [])
    return samples, failures
