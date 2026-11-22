"""Generate the fully local, fixed-seed deterministic sample corpus.

Run with:  python -m scripts.generate_samples --out samples/base --count 6
No network access, model weights or external datasets are involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

WIDTH = HEIGHT = 256
ATTRIBUTES = ("density", "intensity", "coverage", "hue_bias", "edge_ratio")
ENTITIES = ("shape_a", "shape_b", "shape_c")
RELATION_TYPES = ("link", "conflict", "supplement")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def render_image(seed: int) -> bytes:
    rng = _rng(seed)
    base = rng.integers(0, 40, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    # Two filled rectangles and one ellipse give stable geometric primitives.
    for primitive in range(3):
        color = rng.integers(80, 240, size=3, dtype=np.uint8)
        x0 = int(rng.integers(0, WIDTH - 80))
        y0 = int(rng.integers(0, HEIGHT - 80))
        x1 = min(WIDTH, x0 + int(rng.integers(40, 120)))
        y1 = min(HEIGHT, y0 + int(rng.integers(40, 120)))
        if primitive < 2:
            base[y0:y1, x0:x1, :] = color
        else:
            yy, xx = np.ogrid[:HEIGHT, :WIDTH]
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            ry, rx = max(1, (y1 - y0) // 2), max(1, (x1 - x0) // 2)
            mask = ((yy - cy) ** 2) / (ry ** 2) + ((xx - cx) ** 2) / (rx ** 2) <= 1
            base[mask] = color
    noise = rng.integers(0, 25, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    pixels = np.clip(base.astype(np.uint16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def build_assertions(sample_id: str, seed: int) -> dict:
    rng = _rng(seed * 1000 + 7)
    count = 4 + int(rng.integers(0, 3))  # 4..6 assertions
    assertions = []
    for index in range(count):
        entity = ENTITIES[int(rng.integers(0, len(ENTITIES)))]
        attribute = ATTRIBUTES[int(rng.integers(0, len(ATTRIBUTES)))]
        value = round(float(rng.uniform(0.0, 1.0)), 4)
        assertions.append(
            {
                "assertion_id": f"a{index:02d}",
                "entity": entity,
                "attribute": attribute,
                "value": value,
            }
        )
    return {
        "sample_id": sample_id,
        "entities": list(ENTITIES),
        "assertions": assertions,
    }


def build_relations(sample_id: str, assertions: list[dict], seed: int) -> dict:
    rng = _rng(seed * 2000 + 13)
    relations = []
    used: set[str] = set()
    for index, assertion in enumerate(assertions):
        relation_id = f"rel{index:02d}"
        rtype = RELATION_TYPES[int(rng.integers(0, len(RELATION_TYPES)))]
        region = int(rng.integers(0, 16))
        weight = round(float(rng.uniform(0.4, 1.4)), 4)
        used.add(relation_id)
        relations.append(
            {
                "relation_id": relation_id,
                "relation_type": rtype,
                "assertion_id": assertion["assertion_id"],
                "region": region,
                "weight": weight,
            }
        )
    return {"sample_id": sample_id, "relations": relations}


DEFAULT_CONFIG = {
    "visual_dim": 16,
    "text_dim": 12,
    "grid_cols": 4,
    "grid_rows": 4,
    "quant_level": 12,
    "frac_bits": 16,
    "text_weight": 0.6,
    "visual_weight": 0.8,
    "fusion_temperature": 1.0,
    "confidence_threshold": 0.08,
    "conflict_discount": 0.5,
    "supplement_boost": 1.25,
    "max_workers": 4,
    "snapshot_interval": 2,
}


def write_config(path: Path, overrides: dict | None = None) -> None:
    cfg = dict(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=20261122)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        sample_id = f"sample_{index:03d}"
        seed = args.base_seed + index
        sample_dir = out / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "image.png").write_bytes(render_image(seed))
        assertions = build_assertions(sample_id, seed)
        (sample_dir / "assertions.json").write_text(
            json.dumps(assertions, indent=2, sort_keys=True), encoding="utf-8"
        )
        relations = build_relations(sample_id, assertions["assertions"], seed)
        (sample_dir / "relations.json").write_text(
            json.dumps(relations, indent=2, sort_keys=True), encoding="utf-8"
        )

    config_path = Path(args.config) if args.config else out.parent / "flow_config.yaml"
    write_config(config_path)
    print(f"generated {args.count} samples under {out}")
    print(f"config: {config_path}")


if __name__ == "__main__":
    main()
