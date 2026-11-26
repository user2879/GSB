"""Glue helpers that assemble a ready-to-run engine from a sample bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import RenderConfig
from .loader import (load_motion_table, load_prompts, load_vocabulary)
from .pipeline import PipelineState, RenderEngine


def load_state(bundle_dir: str | Path,
               cfg: RenderConfig | None = None) -> tuple[PipelineState, list]:
    bundle = Path(bundle_dir)
    cfg = cfg or RenderConfig.from_yaml(bundle / "render.yaml")
    vocabulary, vocab_checks = load_vocabulary(bundle / "vocabulary.txt")
    prompts, prompt_checks = load_prompts(bundle / "prompts.txt", vocabulary)
    if len(vocabulary) != cfg.text_dim:
        raise ValueError(
            f"E_DIM_MISMATCH: vocabulary {len(vocabulary)} != text_dim")
    images: dict[int, np.ndarray] = {}
    image_checks = []
    for frame_id in range(len(prompts)):
        path = bundle / "images" / f"cond_{frame_id:03d}.png"
        if path.exists():
            with Image.open(path) as image:
                images[frame_id] = np.asarray(
                    image.convert("RGB"), dtype=np.uint8)
    motion, motion_checks = load_motion_table(bundle / "motion.json", cfg)
    if motion is None:
        raise ValueError("E_MOTION_SHAPE: motion table failed validation")
    state = PipelineState(
        vocabulary=vocabulary, prompts=prompts, images=images,
        motion=motion, cfg=cfg)
    checks = vocab_checks + prompt_checks + image_checks + motion_checks
    return state, checks


def build_engine(bundle_dir: str | Path, cache_root: str | Path,
                 cfg: RenderConfig | None = None,
                 max_workers: int | None = None) -> tuple[RenderEngine, list]:
    state, checks = load_state(bundle_dir, cfg)
    engine = RenderEngine(state, cache_root, max_workers=max_workers)
    return engine, checks
