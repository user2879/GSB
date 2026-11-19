"""Offline, fixed-seed generation of all local sample assets.

Running this script is deterministic and never accesses the network:

    python -m dbench.sample_generators.generate_samples
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from ..config import IMAGE_SIZE, LATENT_C, NOISE_ENTRY_LEN

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAMPLES = os.path.join(ROOT, "samples")
CONFIGS = os.path.join(ROOT, "configs")

GEN_SEED = 20261120
STEPS = 6

VOCAB = [
    "cat", "dog", "river", "mountain", "sunset", "forest", "city", "ocean",
    "neon", "calm", "storm", "golden", "blue", "red", "mist", "road",
    "bridge", "flower", "star", "night", "dawn", "snow", "desert", "cloud",
]

# Prompts intentionally include an out-of-vocabulary word to exercise hashing.
PROMPTS = {
    "base": "calm river forest mist dawn",
    "city": "neon city night star blue",
    "pet": "golden cat dog sunset meadow",
}


_TAG_OFFSETS = {
    "image-alpha": 11, "image-beta": 22, "image-wrong": 33,
    "cfg-base": 101, "cfg-city": 102, "cfg-pet": 103, "cfg-i2i": 104,
    "cfg-patch": 105, "cfg-invalid": 106, "patch-values": 202,
}


def _rng(tag: str) -> np.random.Generator:
    # Explicit fixed offsets keep sample generation reproducible across runs
    # and interpreters (never rely on Python's salted built-in hash).
    return np.random.default_rng(GEN_SEED + _TAG_OFFSETS[tag])


def write_vocab() -> str:
    path = os.path.join(SAMPLES, "vocab", "vocab.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(VOCAB) + "\n")
    return path


def write_init_images() -> list[str]:
    paths = []
    for name in ("alpha", "beta"):
        rng = _rng("image-" + name)
        # Smooth gradient plus fixed-seed noise, rounded to uint8 deterministically.
        grid = np.arange(IMAGE_SIZE, dtype=np.float64)
        xx, yy = np.meshgrid(grid, grid)
        base = np.stack(
            [
                (xx * 255.0 / IMAGE_SIZE),
                (yy * 255.0 / IMAGE_SIZE),
                ((xx + yy) * 127.5 / IMAGE_SIZE),
            ],
            axis=-1,
        )
        noise = rng.normal(0.0, 12.0, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
        pixels = np.clip(np.rint(base + noise), 0, 255).astype(np.uint8)
        path = os.path.join(SAMPLES, "init_images", f"{name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(pixels).save(path, format="PNG")
        paths.append(path)

    # Deliberately wrong-sized image for the image-size injection test.
    rng = _rng("image-wrong")
    wrong = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    wrong_path = os.path.join(SAMPLES, "init_images", "wrong_size_16.png")
    Image.fromarray(wrong).save(wrong_path, format="PNG")
    paths.append(wrong_path)
    return paths


def noise_table(rng: np.random.Generator, steps: int) -> list[list[float]]:
    scales = rng.uniform(0.2, 1.1, size=(steps, NOISE_ENTRY_LEN))
    return np.round(scales, 6).tolist()


def base_config(vocab_path: str, name: str, prompt: str,
                rng: np.random.Generator) -> dict:
    return {
        "vocab_file": os.path.relpath(vocab_path, CONFIGS).replace("\\", "/"),
        "prompt": prompt,
        "seed": GEN_SEED + len(name),
        "steps": STEPS,
        "guide_scale": 2.5,
        "noise_table": noise_table(rng, STEPS),
        "latent_shape": [LATENT_C, 8, 8],
        "max_workers": 2,
        "image_replacement_steps": 0,
        "initial_image": None,
        "patches": [],
    }


def write_configs(vocab_path: str, image_paths: list[str]) -> None:
    os.makedirs(CONFIGS, exist_ok=True)
    alpha = os.path.relpath(image_paths[0], CONFIGS).replace("\\", "/")
    beta = os.path.relpath(image_paths[1], CONFIGS).replace("\\", "/")
    wrong = os.path.relpath(image_paths[2], CONFIGS).replace("\\", "/")

    valid: dict[str, dict] = {}
    for name, prompt in PROMPTS.items():
        cfg = base_config(vocab_path, name, prompt, _rng("cfg-" + name))
        cfg["max_workers"] = 1 if name == "base" else 3
        valid[name] = cfg

    # Image-to-image variant.
    i2i = base_config(vocab_path, "i2i", PROMPTS["city"], _rng("cfg-i2i"))
    i2i["initial_image"] = alpha
    i2i["image_replacement_steps"] = 3
    i2i["max_workers"] = 2
    valid["img2img"] = i2i

    # Patch variant (in-range rectangular latent replacement at step 4).
    patch_cfg = base_config(vocab_path, "patch", PROMPTS["base"],
                            _rng("cfg-patch"))
    values = np.round(
        _rng("patch-values").normal(0.0, 0.5, size=(2, 3, LATENT_C)), 6
    ).tolist()
    patch_cfg["patches"] = [{
        "step": 4, "top": 2, "left": 3, "height": 2, "width": 3,
        "values": values,
    }]
    valid["patch"] = patch_cfg

    for name, cfg in valid.items():
        with open(os.path.join(CONFIGS, f"valid_{name}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2, ensure_ascii=False)

    # Invalid configs, one per business code.
    def dump_invalid(fname: str, mutator) -> None:
        cfg = base_config(vocab_path, "invalid", PROMPTS["base"],
                          _rng("cfg-invalid"))
        mutator(cfg)
        with open(os.path.join(CONFIGS, fname), "w", encoding="utf-8") as handle:
            json.dump(cfg, handle, indent=2, ensure_ascii=False)

    dump_invalid("invalid_missing_field.json", lambda c: c.pop("seed"))
    dump_invalid("invalid_type.json", lambda c: c.update(seed="not-an-int"))
    dump_invalid("invalid_seed_range.json", lambda c: c.update(seed=2**33))
    dump_invalid("invalid_guide.json", lambda c: c.update(guide_scale=float("inf")))
    dump_invalid("invalid_step_zero.json", lambda c: c.update(steps=0,
                                                              noise_table=[]))
    dump_invalid("invalid_noise_length.json",
                 lambda c: c.update(noise_table=c["noise_table"][:-1]))
    dump_invalid("invalid_patch_range.json", lambda c: c.update(patches=[{
        "step": 2, "top": 7, "left": 7, "height": 3, "width": 3,
        "values": [[[0.0] * LATENT_C] * 3] * 3,
    }]))
    dump_invalid("invalid_patch_shape.json", lambda c: c.update(patches=[{
        "step": 2, "top": 0, "left": 0, "height": 2, "width": 2,
        "values": [[[0.0] * LATENT_C]],  # 1x1 vs declared 2x2
    }]))

    # Injection manifest covering all four real fault families.
    injections = {
        "injections": [
            {"id": "INJ-001", "type": "non_finite_latent",
             "stage": "denoise_s002", "step": 2,
             "params": {"coords": [0, 0, 0]}},
            {"id": "INJ-002", "type": "image_size_mismatch",
             "stage": "image_encode", "step": None,
             "params": {"wrong_image": os.path.join(
                 os.path.dirname(alpha), "wrong_size_16.png").replace("\\", "/")}},
            {"id": "INJ-003", "type": "noise_entry_truncated",
             "stage": "denoise_s003", "step": 3, "params": {"keep": 2}},
            {"id": "INJ-004", "type": "patch_out_of_range",
             "stage": "denoise_s005", "step": 5,
             "params": {"top": 7, "left": 7, "height": 2, "width": 2}},
        ]
    }
    inj_dir = os.path.join(SAMPLES, "injections")
    os.makedirs(inj_dir, exist_ok=True)
    with open(os.path.join(inj_dir, "faults.json"), "w",
              encoding="utf-8") as handle:
        json.dump(injections, handle, indent=2, ensure_ascii=False)

    # Single-fault injection set for clean failure-isolation verification.
    single = {"injections": [injections["injections"][0]]}
    with open(os.path.join(inj_dir, "single_fault.json"), "w",
              encoding="utf-8") as handle:
        json.dump(single, handle, indent=2, ensure_ascii=False)


def main() -> None:
    vocab_path = write_vocab()
    image_paths = write_init_images()
    write_configs(vocab_path, image_paths)
    print("samples generated under", SAMPLES)
    print("configs generated under", CONFIGS)


if __name__ == "__main__":
    main()
