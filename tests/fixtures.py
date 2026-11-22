"""Shared deterministic fixtures for the test suite."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from mm_pipeline.config import DEFAULT_CONFIG


def make_png(seed: int = 1, size: tuple[int, int] = (256, 256)) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def write_sample(
    root: Path,
    sample_id: str,
    *,
    image_bytes: bytes | None = None,
    assertions: dict | None = None,
    relations: dict | None = None,
) -> Path:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    if image_bytes is None:
        # Stable, PYTHONHASHSEED-independent seed derived from the sample id.
        stable_seed = sum((index + 1) * ord(ch) for index, ch in enumerate(sample_id))
        image_bytes = make_png(seed=stable_seed)
    (sample_dir / "image.png").write_bytes(image_bytes)
    if assertions is None:
        assertions = {
            "sample_id": sample_id,
            "entities": ["ent1", "ent2"],
            "assertions": [
                {"assertion_id": "a0", "entity": "ent1", "attribute": "density", "value": 0.42},
                {"assertion_id": "a1", "entity": "ent2", "attribute": "coverage", "value": 0.7},
            ],
        }
    if relations is None:
        relations = {
            "sample_id": sample_id,
            "relations": [
                {"relation_id": "r0", "relation_type": "link", "assertion_id": "a0", "region": 0, "weight": 1.0},
                {"relation_id": "r1", "relation_type": "conflict", "assertion_id": "a1", "region": 1, "weight": 0.8},
                {"relation_id": "r2", "relation_type": "supplement", "assertion_id": "a0", "region": 2, "weight": 1.1},
            ],
        }
    (sample_dir / "assertions.json").write_text(json.dumps(assertions), encoding="utf-8")
    (sample_dir / "relations.json").write_text(json.dumps(relations), encoding="utf-8")
    return sample_dir


def write_config(root: Path, overrides: dict | None = None) -> Path:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "grid_cols": 2,
            "grid_rows": 2,
            "visual_dim": 8,
            "text_dim": 6,
            "max_workers": 2,
        }
    )
    if overrides:
        cfg.update(overrides)
    path = root / "flow_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    return path


def build_corpus(tmp: Path, count: int = 3) -> tuple[Path, Path]:
    root = tmp / "samples"
    for i in range(count):
        write_sample(root, f"sample_{i:03d}")
    config = write_config(tmp)
    return root, config
