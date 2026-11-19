"""Content-fingerprint planning and cache-hit decisions."""

from __future__ import annotations

from typing import Any

from .config import Config
from .dag import (
    DECODE_STAGE,
    ENCODE_STAGE,
    IMAGE_ENCODE_STAGE,
    INIT_STAGE,
    Node,
    denoise_stage,
    denoise_step_of,
)
from .hashing import fingerprint_json
from .stages import vocab_fingerprint


def stage_param_fingerprint(config: Config, stage: str,
                            vocab_fp: str, image_pixel_fp: str | None) -> str:
    """Normalized fingerprint of the stage's own parameters (no deps)."""
    step = denoise_step_of(stage)
    if stage == ENCODE_STAGE:
        payload = {"stage": stage, "prompt": config.prompt,
                   "vocab_fingerprint": vocab_fp}
    elif stage == INIT_STAGE:
        payload = {"stage": stage, "seed": config.seed,
                   "config_fingerprint": config.effective_fingerprint()}
    elif stage == IMAGE_ENCODE_STAGE:
        payload = {"stage": stage, "image": config.initial_image,
                   "pixel_fingerprint": image_pixel_fp,
                   "config_fingerprint": config.effective_fingerprint()}
    elif stage == DECODE_STAGE:
        payload = {"stage": stage,
                   "config_fingerprint": config.effective_fingerprint()}
    elif step is not None:
        node_payload = {
            "stage": stage,
            "step": step,
            "guide_scale": config.guide_scale,
            "seed": config.seed,
            "config_fingerprint": config.effective_fingerprint(),
            "noise_table_entry": list(config.noise_table[step - 1]),
            "patches": [
                {
                    "top": p.top, "left": p.left,
                    "height": p.height, "width": p.width, "values": p.values,
                }
                for p in config.patches_for_step(step)
            ],
        }
        payload = node_payload
    else:  # pragma: no cover - defensive
        payload = {"stage": stage,
                   "config_fingerprint": config.effective_fingerprint()}
    return fingerprint_json({"schema": "dbench.stage-params.v1", "payload": payload})


def content_fingerprint(param_fp: str, dep_fps: dict[str, str]) -> str:
    return fingerprint_json({
        "schema": "dbench.content-fp.v1",
        "params": param_fp,
        "deps": dict(sorted(dep_fps.items())),
    })


def plan_cache(config: Config, nodes: dict[str, Node], store, vocab_fp: str,
               image_pixel_fp: str | None) -> dict[str, dict[str, Any]]:
    """Compute desired fingerprints and decide reuse vs recompute per stage.

    A stage is reused only when an existing artifact version carries exactly the
    desired content fingerprint (same normalized params AND same upstream
    fingerprints). Otherwise it is invalidated and scheduled for recomputation.
    No computation happens here; this is a pure planning pass.
    """
    plan: dict[str, dict[str, Any]] = {}
    # Walk in dependency order so upstream fingerprints resolve first.
    from .dag import topological_order

    for stage in topological_order(nodes):
        node = nodes[stage]
        param_fp = stage_param_fingerprint(config, stage, vocab_fp, image_pixel_fp)
        dep_fps = {dep: plan[dep]["content_fingerprint"] for dep in node.deps}
        target_fp = content_fingerprint(param_fp, dep_fps)
        store.check_continuity(stage)
        hit_version = store.find_version_by_fingerprint(stage, target_fp)
        plan[stage] = {
            "node": node,
            "param_fingerprint": param_fp,
            "dependency_fingerprints": dep_fps,
            "content_fingerprint": target_fp,
            "reuse": hit_version is not None,
            "reuse_version": hit_version,
        }
    return plan
