#!/usr/bin/env python3
"""Generate deterministic local samples with a fixed seed.

Outputs into a sample directory (default ``samples/``):
  - render_config.yaml
  - prompts.jsonl
  - edit_plan.json
  - init images under ``images/``

The plan covers single-batch and multi-batch sequences, two resolution tiers,
prompt replacement, seed/guidance/step changes, tier add/remove, init-image
replacement, patch add/remove (including a no-op and an out-of-range batch),
plus deliberately invalid batches so failure handling is observable.
"""

import argparse
import json
import os

import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_workbench.hashing import random_state
from render_workbench.png_codec import encode_png

SEED = 20240117
VOCAB = [
    "mountain", "river", "forest", "sunset", "mist", "stone", "bird",
    "valley", "cloud", "snow",
]


def _noise_tables(rng, steps):
    return [round(float(x), 6)
            for x in rng.uniform(0.05, 0.95, size=steps)]


def build_config():
    rng = np.random.RandomState(SEED)
    steps = 4
    tiers = [
        {"id": "low", "width": 12, "height": 9},
        {"id": "high", "width": 20, "height": 15},
    ]
    config = {
        "vocab": VOCAB,
        "embedding_dim": 8,
        "latent_channels": 4,
        "seed": 7001,
        "seed_min": 1,
        "seed_max": 100000,
        "denoise_steps": steps,
        "guidance_scale": 1.25,
        "img2img_replace_steps": 2,
        "patch_tolerance": 0.01,
        "concurrency": 2,
        "snapshot_interval": 2,
        "tiers": tiers,
        "noise_tables": {
            tier["id"]: _noise_tables(rng, steps) for tier in tiers
        },
        "guidance_tables": {
            tier["id"]: [round(float(x), 6)
                         for x in rng.uniform(0.1, 1.5, size=steps)]
            for tier in tiers
        },
    }
    return config, tiers


def write_init_image(path, width, height, salt):
    rng = random_state("init-image", salt, width, height)
    base = rng.randint(0, 256, size=(height, width, 3)).astype(np.uint8)
    gradient = np.tile(
        np.linspace(0, 255, width, dtype=np.uint8).reshape(1, width, 1),
        (height, 1, 3))
    pixels = np.clip(
        base.astype(np.uint16) // 2 + gradient.astype(np.uint16) // 2,
        0, 255).astype(np.uint8)
    with open(path, "wb") as handle:
        handle.write(encode_png(pixels))
    return pixels


def build_prompts(sample_dir, tiers):
    image_dir = os.path.join(sample_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    img_a = os.path.join("images", "init_a.png")
    img_b = os.path.join("images", "init_b.png")
    write_init_image(os.path.join(sample_dir, img_a), 16, 12, "a")
    write_init_image(os.path.join(sample_dir, img_b), 16, 12, "b")

    channels = 4
    low = tiers[0]
    patch_values = [round(float(v), 4)
                    for v in np.linspace(-0.5, 0.5,
                                         2 * 3 * channels).tolist()]
    noop_patch_values = [0.0] * (2 * 2 * channels)
    prompts = [
        {
            "id": "p_alpha",
            "tokens": ["mountain", "mist", "sunset", "mountain"],
        },
        {
            "id": "p_beta",
            "tokens": ["forest", "river", "bird"],
            "init_image": img_a,
        },
        {
            "id": "p_gamma",
            "tokens": ["valley", "cloud", "snow", "stone"],
            "patches": [
                {"step": 1, "x": 1, "y": 1, "width": 2, "height": 3,
                 "values": patch_values},
            ],
        },
        {
            "id": "p_delta",
            "tokens": ["snow", "stone"],
            "init_image": img_b,
            "patches": [
                {"step": 0, "x": 0, "y": 0, "width": 2, "height": 2,
                 "values": noop_patch_values},
            ],
        },
    ]
    return prompts, patch_values, noop_patch_values


def build_plan(patch_values, noop_patch_values):
    plan = {"description": "deterministic multi-batch edit plan", "batches": []}

    def add(batch_id, edits, expect=None, note=None):
        batch = {"id": batch_id, "edits": edits}
        if expect:
            batch["expected_failure_code"] = expect
        if note:
            batch["note"] = note
        plan["batches"].append(batch)

    add("b01_token_replace", [
        {"op": "replace_prompt_tokens", "prompt_id": "p_alpha",
         "tokens": ["mountain", "snow", "cloud"]},
    ], note="single batch: prompt edit invalidates encoding + downstream")

    add("b02_seed_and_guidance", [
        {"op": "set_seed", "seed": 8123},
        {"op": "set_guidance_scale", "guidance_scale": 1.75},
    ])

    add("b03_init_image_swap", [
        {"op": "replace_init_image", "prompt_id": "p_beta",
         "init_image": "images/init_b.png"},
    ])

    add("b04_patch_add", [
        {"op": "add_patch", "prompt_id": "p_gamma", "patch": {
            "step": 2, "x": 3, "y": 2, "width": 2, "height": 2,
            "values": [0.25] * (2 * 2 * 4)}},
    ])

    add("b04b_patch_seed", [
        {"op": "add_patch", "prompt_id": "p_delta", "patch": {
            "step": 1, "x": 1, "y": 1, "width": 2, "height": 2,
            "values": [0.3] * (2 * 2 * 4)}},
    ], note="seed a real patch so the next identical-write patch is a no-op")

    add("b05_patch_noop", [
        {"op": "add_patch", "prompt_id": "p_delta", "patch": {
            "step": 1, "x": 1, "y": 1, "width": 2, "height": 2,
            "values": [0.3005] * (2 * 2 * 4)}},
    ], note="sub-tolerance rewrite (0.0005 < 0.01) must be a no-op")

    add("b06_tier_add", [
        {"op": "add_tier", "id": "ultra", "width": 28, "height": 21,
         "noise": [0.11, 0.22, 0.33, 0.44],
         "guidance": [0.5, 0.6, 0.7, 0.8]},
    ])

    add("b07_tier_remove", [
        {"op": "remove_tier", "id": "ultra"},
    ])

    add("b08_steps_change", [
        {"op": "set_denoise_steps", "denoise_steps": 3},
    ], note="tables truncated to new step count")

    add("b09_steps_restore", [
        {"op": "set_denoise_steps", "denoise_steps": 4},
    ], note="previously cached tail nodes may hit again")

    add("b10_patch_remove", [
        {"op": "remove_patch", "prompt_id": "p_gamma", "step": 1,
         "region": {"x": 1, "y": 1, "width": 2, "height": 3}},
    ])

    add("b11_bad_token", [
        {"op": "replace_prompt_tokens", "prompt_id": "p_alpha",
         "tokens": ["mountain", "dragon"]},
    ], expect="E_TOKEN_OUT_OF_VOCAB",
        note="invalid batch rolled back, state preserved")

    add("b12_patch_out_of_range", [
        {"op": "add_patch", "prompt_id": "p_gamma", "patch": {
            "step": 0, "x": 100, "y": 100, "width": 4, "height": 4,
            "values": [0.1] * (4 * 4 * 4)}},
    ], expect="E_PATCH_OUT_OF_RANGE",
        note="out-of-range patch rejected, prior state retained")

    add("b13_bad_seed", [
        {"op": "set_seed", "seed": 99999999},
    ], expect="E_CONFIG_INVALID")

    add("b14_final_tweak", [
        {"op": "set_guidance_scale", "guidance_scale": 1.1},
        {"op": "replace_init_image", "prompt_id": "p_beta",
         "init_image": "images/init_a.png"},
    ], note="final batch used for full-vs-incremental pixel comparison")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    sample_dir = os.path.abspath(args.out)
    os.makedirs(sample_dir, exist_ok=True)

    config, tiers = build_config()
    with open(os.path.join(sample_dir, "render_config.yaml"), "w",
              encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, default_flow_style=False)

    prompts, patch_values, noop_patch_values = build_prompts(
        sample_dir, tiers)
    with open(os.path.join(sample_dir, "prompts.jsonl"), "w",
              encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt, sort_keys=True) + "\n")

    plan = build_plan(patch_values, noop_patch_values)
    with open(os.path.join(sample_dir, "edit_plan.json"), "w",
              encoding="utf-8") as handle:
        json.dump(plan, handle, sort_keys=True, indent=2)
    print("samples written to %s" % sample_dir)


if __name__ == "__main__":
    main()
