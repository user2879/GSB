from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .pngcodec import encode_png_rgb


SEED = 290
FRAME_COUNT = 4
WIDTH = 16
HEIGHT = 8
IMAGE_SIZE = 8
BLOCK_W = 4
BLOCK_H = 4


def controlled_vocab() -> dict[str, list[str]]:
    return {
        "entities": ["cube", "sphere", "cylinder", "pyramid", "grid", "terrain"],
        "attributes": [
            "red", "green", "blue", "bright", "dark", "striped", "smooth", "rough"
        ],
    }


def vocab_terms() -> list[str]:
    vocab = controlled_vocab()
    return vocab["entities"] + vocab["attributes"]


def default_config(root: str | Path = ".") -> dict[str, Any]:
    return {
        "schema": "condrender-config-v1",
        "snapshot_schema": "condrender-snapshot-v1",
        "global_version": 1,
        "config_version": "cfg-sample-v1",
        "seed": SEED,
        "seed_min": 0,
        "seed_max": 4294967295,
        "frames": FRAME_COUNT,
        "width": WIDTH,
        "height": HEIGHT,
        "condition_image": {
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "block_cols": 2,
            "block_rows": 2,
        },
        "motion_grid": {
            "block_width": BLOCK_W,
            "block_height": BLOCK_H,
            "cols": WIDTH // BLOCK_W,
            "rows": HEIGHT // BLOCK_H,
            "vector_components": 2,
        },
        "condition_dims": {
            "text": 14,
            "image": 20,
            "motion": 16,
            "motion_row": 8,
            "fused": 42,
        },
        "fixed_point": {
            "frac_bits": 12,
            "quant_profile": "Q12_BALANCED",
            "rounding_mode": "HALF_AWAY_FROM_ZERO",
        },
        "fusion": {
            "weight_scale": 10000,
            "weights": {"text": "0.45", "image": "0.25", "motion": "0.30"},
        },
        "denoise_steps": 3,
        "max_noise_steps": 8,
        "guidance_scale": "1.50",
        "step_alpha_bp": [10000, 6500, 4000, 2500, 1500, 1000, 700, 500],
        "noise": {
            "required_version": "noise-v1",
            "allowed_versions": ["noise-v1", "noise-v2"],
            "table_path": "samples/noise_table.json",
        },
        "decoder": {
            "matrix_bp": [
                [10000, 1200, -800],
                [-600, 10000, 1000],
                [900, -1200, 10000],
            ],
            "bias_q": [0, 0, 0],
            "center_q_fraction": 128,
            "pixel_gain": "48.0",
        },
        "tile_rows": 2,
        "tile_scheduling": "zigzag",
        "max_workers": 4,
        "interaction_budget_ms": 250,
        "recompute_point_budget": 100000,
        "stage_costs": {
            "text": 1,
            "image": 2,
            "motion": 1,
            "fusion": 2,
            "initial": 1,
            "step": 3,
            "tile": 2,
            "frame_compose": 1,
        },
        "paths": {
            "prompts": "samples/prompt_table.yaml",
            "vocab": "samples/controlled_vocab.yaml",
            "motion": "samples/motion_vectors.yaml",
            "commands": "samples/command_stream.yaml",
            "injections": "samples/injections.yaml",
        },
    }


def prompt_table(seed: int = SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 1)
    vocab = controlled_vocab()
    entities = vocab["entities"]
    attributes = vocab["attributes"]
    frame_prompts: list[str] = []
    prompt_records: list[dict[str, str]] = []
    for frame in range(FRAME_COUNT):
        entity = entities[int(rng.integers(0, len(entities)))]
        attr_count = int(rng.integers(2, 4))
        attrs = [
            attributes[int(index)]
            for index in rng.choice(len(attributes), size=attr_count, replace=False)
        ]
        words = attrs + [entity]
        if frame % 2 == 0 and "grid" not in words and entity not in {"grid", "terrain"}:
            words.append("grid")
        text = " ".join(words)
        frame_prompts.append(text)
        prompt_records.append({"prompt_id": f"P{frame}", "text": text})
    return {
        "version": "prompt-table-v1",
        "seed": seed,
        "generation": "local PCG64 Generator; no global random state",
        "frame_prompts": frame_prompts,
        "prompts": prompt_records,
        "frame_assignments": [f"P{index}" for index in range(FRAME_COUNT)],
    }


def draw_condition_image(seed: int, size: int = IMAGE_SIZE) -> bytes:
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    block = size // 2
    for by in range(2):
        for bx in range(2):
            base = rng.integers(24, 190, size=3, dtype=np.int64)
            y0, x0 = by * block, bx * block
            image[y0:y0 + block, x0:x0 + block, :] = base.astype(np.uint8)
            for yy in range(y0, y0 + block):
                for xx in range(x0, x0 + block):
                    if (xx + yy + seed) % 3 == 0:
                        image[yy, xx, :] = np.clip(base + 48, 0, 255).astype(np.uint8)

    accent = tuple(int(v) for v in rng.integers(180, 256, size=3))
    image[1:3, 1:size - 1, :] = accent
    image[size - 3:size - 1, 2:size - 2, :] = tuple(
        int(v) for v in rng.integers(0, 90, size=3)
    )
    center = size // 2
    for y in range(size):
        for x in range(size):
            if (x - center) ** 2 + (y - center) ** 2 <= max(1, size // 4) ** 2:
                image[y, x, :] = (245, 245, 245)
    for y in range(2, size - 2):
        x = 1 + abs(center - y)
        if 0 <= x < size:
            image[y, x, :] = (20, 20, 20)
            image[y, min(size - 1, x + 1), :] = (20, 20, 20)
    return encode_png_rgb(image)


def motion_table(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or default_config()
    grid = config["motion_grid"]
    rng = np.random.default_rng(config["seed"] + 17)
    vectors = rng.integers(
        -4, 5,
        size=(config["frames"], grid["rows"], grid["cols"], 2),
        dtype=np.int64,
    )
    return {
        "version": "motion-table-v1",
        "frame_width": config["width"],
        "frame_height": config["height"],
        "block_width": grid["block_width"],
        "block_height": grid["block_height"],
        "vectors": vectors.tolist(),
    }


def noise_table(config: dict[str, Any] | None = None, version: str = "noise-v1") -> dict[str, Any]:
    config = config or default_config()
    latent_pixels = config["width"] * config["height"] * 3
    rng = np.random.default_rng(config["seed"] + 101)
    steps = rng.integers(
        -128, 129, size=(config["max_noise_steps"], latent_pixels), dtype=np.int64
    )
    return {
        "version": version,
        "shape": ["step", "flat_latent_channel"],
        "frac_bits_hint": config["fixed_point"]["frac_bits"],
        "entries": steps.tolist(),
    }


def command_stream() -> dict[str, Any]:
    return {
        "version": "command-stream-v1",
        "description": "Covers every edit class required by the incremental equivalence test.",
        "commands": [
            {"id": "C1", "op": "replace_prompt", "frame": 0,
             "text": "blue cube rough dark terrain"},
            {"id": "C2", "op": "replace_image", "frame": 2, "image_id": "condition_c"},
            {"id": "C3", "op": "delete_image", "frame": 3},
            {"id": "C4", "op": "adjust_motion", "frame": 1,
             "row": 0, "col": 1, "component": "dx", "value": 5},
            {"id": "C5", "op": "set_weights",
             "weights": {"text": "0.55", "image": "0.20", "motion": "0.25"}},
            {"id": "C6", "op": "set_denoise_steps", "steps": 5},
            {"id": "C7", "op": "set_tile_rows", "tile_rows": 4},
            {"id": "C8", "op": "set_guidance", "guidance_scale": "2.25"},
            {"id": "C9", "op": "set_quant_profile", "quant_profile": "Q16_DIAG"},
            {"id": "C10", "op": "set_denoise_steps", "steps": 4},
        ],
    }


def injection_config() -> dict[str, Any]:
    return {
        "version": "injection-config-v1",
        "description": "Independent fault injection; IDs must appear in diagnostics.",
        "injections": [
            {"id": "INJ-PNG-01", "type": "truncated_png", "target_frame": 0,
             "stage": "image_decode", "expected_code": "E_PNG_TRUNCATED"},
            {"id": "INJ-NAN-02", "type": "nonfinite_condition", "target_frame": 1,
             "stage": "image_encode", "expected_code": "E_NONFINITE_CONDITION"},
            {"id": "INJ-MOT-03", "type": "motion_block_mismatch", "target_frame": 2,
             "stage": "motion_validate", "expected_code": "E_MOTION_BLOCK_MISMATCH"},
            {"id": "INJ-NOISE-04", "type": "noise_version_missing", "target_frame": 3,
             "stage": "denoise_step", "expected_code": "E_NOISE_VERSION_MISSING"},
        ],
    }


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def generate_samples(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    paths: list[str] = []
    config = default_config(root)

    _write_yaml(root / "samples/controlled_vocab.yaml", controlled_vocab())
    _write_yaml(root / "samples/prompt_table.yaml", prompt_table(config["seed"]))
    _write_yaml(root / "samples/render_config.yaml", config)
    _write_yaml(root / "samples/motion_vectors.yaml", motion_table(config))
    _write_yaml(root / "samples/command_stream.yaml", command_stream())
    _write_yaml(root / "samples/injections.yaml", injection_config())

    for name, seed in [("condition_a", 701), ("condition_b", 702), ("condition_c", 703)]:
        png_path = root / f"samples/images/{name}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(draw_condition_image(seed))
        paths.append(str(png_path.relative_to(root)))

    noise_path = root / config["noise"]["table_path"]
    noise_path.parent.mkdir(parents=True, exist_ok=True)
    noise_path.write_text(json.dumps(noise_table(config)), encoding="utf-8")

    negative_dir = root / "samples/negative"
    negative_dir.mkdir(parents=True, exist_ok=True)
    bad_prompts = prompt_table()
    bad_prompts["frame_prompts"][0] = "red frobnicate cube"
    _write_yaml(negative_dir / "unknown_term_prompt.yaml", bad_prompts)

    bad_config = json.loads(json.dumps(config))
    bad_config["seed"] = -7
    bad_config["denoise_steps"] = 0
    bad_config["tile_rows"] = -2
    bad_config["fixed_point"]["quant_profile"] = "Q32_EXTREME"
    bad_config["fixed_point"]["frac_bits"] = 32
    _write_yaml(negative_dir / "bad_render_config.yaml", bad_config)

    bad_dim_config = json.loads(json.dumps(config))
    bad_dim_config["condition_dims"]["text"] = 999
    _write_yaml(negative_dir / "bad_dim_config.yaml", bad_dim_config)

    good_png = draw_condition_image(808)
    (negative_dir / "truncated_condition.png").write_bytes(good_png[: max(40, len(good_png) // 2)])
    small = draw_condition_image(809, size=4)
    (negative_dir / "wrong_resolution.png").write_bytes(small)
    (negative_dir / "bad_signature.png").write_bytes(b"NOTPNG" + good_png[8:])

    bad_motion = motion_table(config)
    bad_motion["vectors"] = np.asarray(bad_motion["vectors"])[:, :1, :, :].tolist()
    _write_yaml(negative_dir / "bad_motion_vectors.yaml", bad_motion)

    bad_noise = noise_table(config)
    bad_noise["version"] = "noise-v9-undeclared"
    (negative_dir / "bad_noise_version.json").write_text(json.dumps(bad_noise), encoding="utf-8")
    missing_noise = noise_table(config)
    missing_noise.pop("version")
    (negative_dir / "missing_noise_version.json").write_text(
        json.dumps(missing_noise), encoding="utf-8"
    )

    cases = {
        "version": "negative-cases-v1",
        "cases": [
            {"id": "NEG-VOCAB", "kind": "prompt", "path": "samples/negative/unknown_term_prompt.yaml",
             "expected_code": "E_VOCAB_UNKNOWN_TERM"},
            {"id": "NEG-CONFIG", "kind": "config", "path": "samples/negative/bad_render_config.yaml",
             "expected_codes": ["E_SEED_RANGE", "E_POSITIVE_INT", "E_QUANT_ENUM"]},
            {"id": "NEG-DIM", "kind": "bundle", "path": "samples/negative/bad_dim_config.yaml",
             "expected_code": "E_COND_DIM_MISMATCH"},
            {"id": "NEG-PNG-TRUNC", "kind": "png",
             "path": "samples/negative/truncated_condition.png",
             "expected_code": "E_PNG_TRUNCATED"},
            {"id": "NEG-PNG-RES", "kind": "png",
             "path": "samples/negative/wrong_resolution.png",
             "expected_code": "E_PNG_RESOLUTION"},
            {"id": "NEG-PNG-SIG", "kind": "png",
             "path": "samples/negative/bad_signature.png",
             "expected_code": "E_PNG_SIGNATURE"},
            {"id": "NEG-MOTION", "kind": "motion",
             "path": "samples/negative/bad_motion_vectors.yaml",
             "expected_code": "E_MOTION_BLOCK_MISMATCH"},
            {"id": "NEG-NOISE-VER", "kind": "noise",
             "path": "samples/negative/bad_noise_version.json",
             "expected_code": "E_NOISE_VERSION"},
            {"id": "NEG-NOISE-MISSING", "kind": "noise",
             "path": "samples/negative/missing_noise_version.json",
             "expected_code": "E_NOISE_VERSION_MISSING"},
        ],
    }
    _write_yaml(negative_dir / "cases.yaml", cases)

    readme = root / "samples/README.md"
    readme.write_text(
        "# 确定性样本\n\n"
        "由 `python -m condrender.cli generate-samples` 重新生成；所有随机数均来自固定种子的局部 Generator。\n",
        encoding="utf-8",
    )

    return {"files": paths, "root": [str(root)]}
