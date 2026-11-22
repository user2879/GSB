#!/usr/bin/env python3
"""Generate the deterministic local sample corpus used by tests and the demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VOCABULARY = [
    "amber",
    "forest",
    "river",
    "neon",
    "city",
    "slow",
    "orbit",
    "rain",
    "silver",
    "sunrise",
    "ocean",
    "glass",
    "mist",
    "light",
    "shadow",
    "bloom",
    "drift",
    "stone",
    "warm",
    "cool",
    "night",
    "dawn",
    "wave",
    "cloud",
    "copper",
    "meadow",
    "signal",
    "velvet",
    "canyon",
    "snow",
    "ember",
    "horizon",
]


def save_described_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        latent=array,
        shape=np.asarray(array.shape, dtype=np.int64),
        dtype=np.asarray(str(array.dtype)),
    )


def save_noise_table(path: Path, noise: np.ndarray, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        noise=noise.astype(np.int64, copy=False),
        version=np.asarray(version),
        shape=np.asarray(noise.shape, dtype=np.int64),
        dtype=np.asarray(str(noise.dtype)),
    )


def make_commands(output: Path) -> dict:
    return {
        "schema_version": 1,
        "description": "A multi-stage command stream covering prompt, seed, guidance, steps, frame count, image, patching, chunking, and quantizer changes.",
        "commands": [
            {"command_id": "c01", "op": "set_prompt", "prompt": "silver ocean sunrise glass"},
            {"command_id": "c02", "op": "set_seed", "seed": 20261124},
            {"command_id": "c03", "op": "set_guidance", "guidance": 3.25},
            {"command_id": "c04", "op": "set_steps", "steps": 20},
            {"command_id": "c05", "op": "set_frame_count", "frame_count": 8},
            {"command_id": "c06", "op": "set_chunk_rows", "chunk_rows": 4},
            {
                "command_id": "c07",
                "op": "add_patch",
                "patch": {
                    "patch_id": "p1",
                    "step": 7,
                    "frame": 2,
                    "x": 2,
                    "y": 1,
                    "width": 3,
                    "height": 2,
                    "delta": 0.4,
                },
            },
            {"command_id": "c08", "op": "set_initial_image", "path": "initial_latent_alt.npz"},
            {"command_id": "c09", "op": "set_prompt", "prompt": "neon city orbit rain mist"},
            {"command_id": "c10", "op": "remove_patch", "patch_id": "p1"},
            {"command_id": "c11", "op": "set_quantization_level", "quantization_level": "q4"},
            {
                "command_id": "c12",
                "op": "add_patch",
                "patch": {
                    "patch_id": "p2",
                    "step": 10,
                    "frame": 5,
                    "x": 0,
                    "y": 0,
                    "width": 4,
                    "height": 2,
                    "delta": -0.35,
                },
            },
            {"command_id": "c13", "op": "set_frame_count", "frame_count": 4},
            {"command_id": "c14", "op": "set_frame_count", "frame_count": 8},
            {
                "command_id": "c15",
                "op": "add_patch",
                "patch": {
                    "patch_id": "p3",
                    "step": 9,
                    "frame": 6,
                    "x": 1,
                    "y": 1,
                    "width": 2,
                    "height": 2,
                    "delta": 0.25,
                },
            },
            {"command_id": "c16", "op": "set_steps", "steps": 16},
            {"command_id": "c17", "op": "set_initial_image", "path": None},
            {"command_id": "c18", "op": "set_chunk_rows", "chunk_rows": 8},
            {
                "command_id": "c19",
                "op": "add_patch",
                "patch": {
                    "patch_id": "p_bad",
                    "step": 9,
                    "frame": 1,
                    "x": 99,
                    "y": 0,
                    "width": 2,
                    "height": 2,
                    "delta": 0.1,
                },
            },
            {"command_id": "c20", "op": "set_seed", "seed": 999999999999},
        ],
    }


def generate(output: Path, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "noise_tables").mkdir(exist_ok=True)
    (output / "vocabulary.txt").write_text("\n".join(VOCABULARY) + "\n", encoding="utf-8")

    noise_rng = np.random.Generator(np.random.PCG64(seed + 101))
    noise_shape = (24, 8, 3, 8, 8)
    noise_v1 = np.rint(noise_rng.standard_normal(noise_shape) * 72.0).astype(np.int64)
    noise_rng_v2 = np.random.Generator(np.random.PCG64(seed + 202))
    noise_v2 = np.rint(noise_rng_v2.standard_normal(noise_shape) * 84.0).astype(np.int64)
    save_noise_table(output / "noise_tables" / "nt-v1.npz", noise_v1, "nt-v1")
    save_noise_table(output / "noise_tables" / "nt-v2.npz", noise_v2, "nt-v2")

    image_rng = np.random.Generator(np.random.PCG64(seed + 303))
    initial = np.rint(image_rng.standard_normal((3, 8, 8)) * 0.26).astype(np.float32)
    save_described_array(output / "initial_latent.npz", initial)
    alt_rng = np.random.Generator(np.random.PCG64(seed + 404))
    alt = (alt_rng.standard_normal((3, 8, 8)) * 0.24 + 0.18).astype(np.float32)
    save_described_array(output / "initial_latent_alt.npz", alt)

    config = {
        "schema_version": 1,
        "name": "local-deterministic-fixedvideo-demo",
        "seed_range": [0, 4294967295],
        "seed": seed,
        "prompt": "amber forest river drift",
        "steps": 18,
        "guidance": 2.8,
        "frame_count": 6,
        "frame_resolution": {"height": 32, "width": 40},
        "chunk_rows": 8,
        "latent_shape": [3, 8, 8],
        "embedding_dim": 32,
        "allowed_fixed_point_bits": [8, 10, 12, 14],
        "fixed_point_bits": 12,
        "allowed_quantization_levels": ["q1", "q2", "q4"],
        "quantization_level": "q2",
        "interaction_latency_budget_ms": 8,
        "worker_count": 4,
        "snapshot_interval_commands": 3,
        "recompute_tile_budget": 8,
        "initial_skip_steps": 4,
        "noise_table_versions": ["nt-v1", "nt-v2"],
        "noise_table": {
            "path": "noise_tables/nt-v1.npz",
            "version": "nt-v1",
            "shape": list(noise_shape),
        },
        "vocabulary": {"path": "vocabulary.txt"},
        "initial_image": {
            "path": "initial_latent.npz",
            "shape": [3, 8, 8],
            "dtype": "float32",
        },
        "projection_seed": 9917,
        "condition_seed": 4421,
        "output_gain": 2,
    }
    (output / "render_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commands = make_commands(output)
    (output / "commands.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="samples", help="sample output directory")
    parser.add_argument("--seed", type=int, default=20261123, help="fixed sample-generation seed")
    args = parser.parse_args()
    generate(Path(args.out).resolve(), args.seed)
    print(json.dumps({"status": "ok", "output": str(Path(args.out).resolve()), "seed": args.seed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
