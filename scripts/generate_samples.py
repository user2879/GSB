"""Generate deterministic local sample directories and anomaly fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def texture(seed: int, size: int, frequency: float = 1.0) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    y, x = np.mgrid[0:size, 0:size]
    waves = (
        0.26 * np.sin((x + seed % 7) * frequency * 0.31)
        + 0.20 * np.cos((y - seed % 5) * frequency * 0.27)
        + 0.16 * np.sin((x + y) * frequency * 0.17)
    )
    noise = rng.normal(0.0, 0.10, size=(size, size))
    checker = np.where(((x // 5) + (y // 5)) % 2 == 0, 0.04, -0.04)
    image = np.clip(0.50 + waves + noise + checker, 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8)


def write_png(path: Path, array: np.ndarray, mode: str = "L") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def base_config() -> dict:
    return {
        "schema_version": 1,
        "seed": 20261124,
        "resolution_tiers": [16, 32, 48, 64],
        "resolution": 32,
        "noise_layer_count": 4,
        "noise_layers": [
            {"name": "macro", "texture": "textures/base_01.png", "frequency": 0.09, "amplitude": 0.22},
            {"name": "weave", "texture": "textures/base_02.png", "frequency": 0.17, "amplitude": 0.16},
            {"name": "scratch", "texture": "textures/base_03.png", "frequency": 0.31, "amplitude": 0.10},
            {"name": "micro", "texture": "textures/base_04.png", "frequency": 0.53, "amplitude": 0.06},
        ],
        "material_mix_mode": "weighted",
        "fixed_point": {
            "bit_width": 16,
            "fraction_bits": 8,
            "quantization_level": 2,
            "quantization_tiers": [1, 2, 4, 8],
            "rounding": {
                "diffuse": "half_up",
                "roughness": "half_even",
                "metallic": "half_up",
                "normal_perturbation": "half_even",
            },
            "weight_fraction_bits": 12,
        },
        "error_trace": {
            "bit_width_ladder": [12, 16, 20, 24],
            "tolerance": {
                "noise_composition": 0.00001,
                "quantization": 0.00001,
                "material_mix": 0.00001,
                "channel_synthesis": 0.00001,
            },
            "final_consistency_tolerance": 0.00001,
            "embed_pixel_diffs": True,
        },
        "concurrency": 4,
        "max_concurrency": 4,
        "point_budget": 4096,
        "latency_budget_ms": 500,
        "snapshot_commit_interval": 2,
        "max_boundary_samples": 10000,
    }


def materials() -> dict:
    return {
        "schema_version": 1,
        "materials": [
            {
                "id": "brushed_steel",
                "label": "Brushed steel",
                "layers": [
                    {"noise_layer": 0, "weights": {"diffuse": 0.25, "roughness": 0.30, "metallic": 0.55, "normal_perturbation": 0.20}},
                    {"noise_layer": 1, "weights": {"diffuse": 0.30, "roughness": 0.25, "metallic": 0.20, "normal_perturbation": 0.30}},
                    {"noise_layer": 2, "weights": {"diffuse": 0.25, "roughness": 0.30, "metallic": 0.15, "normal_perturbation": 0.30}},
                    {"noise_layer": 3, "weights": {"diffuse": 0.20, "roughness": 0.15, "metallic": 0.10, "normal_perturbation": 0.20}},
                ],
            },
            {
                "id": "painted_plastic",
                "label": "Painted plastic",
                "layers": [
                    {"noise_layer": 0, "weights": {"diffuse": 0.30, "roughness": 0.40, "metallic": 0.10, "normal_perturbation": 0.18}},
                    {"noise_layer": 1, "weights": {"diffuse": 0.30, "roughness": 0.20, "metallic": 0.10, "normal_perturbation": 0.22}},
                    {"noise_layer": 2, "weights": {"diffuse": 0.20, "roughness": 0.20, "metallic": 0.15, "normal_perturbation": 0.28}},
                    {"noise_layer": 3, "weights": {"diffuse": 0.20, "roughness": 0.20, "metallic": 0.65, "normal_perturbation": 0.32}},
                ],
            },
        ],
    }


def commands() -> dict:
    return {
        "schema_version": 1,
        "commands": [
            {"id": "cmd-001", "op": "set_layer", "layer": 1, "frequency": 0.19, "amplitude": 0.15},
            {"id": "cmd-002", "op": "set_weight", "material": "brushed_steel", "layer": 2, "channel": "roughness", "value": 0.34},
            {"id": "cmd-003", "op": "set_mix_mode", "value": "multiply"},
            {"id": "cmd-004", "op": "set_quantization", "level": 4},
            {"id": "cmd-005", "op": "set_weight", "material": "painted_plastic", "layer": 0, "channel": "diffuse", "value": 0.42},
            {"id": "cmd-006", "op": "set_bit_width", "bit_width": 20},
        ],
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_basic(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for index in range(1, 5):
        write_png(destination / f"textures/base_{index:02d}.png", texture(4100 + index, 32, 0.6 + index * 0.2))
    write_json(destination / "config.json", base_config())
    write_json(destination / "materials.json", materials())
    write_json(destination / "commands.json", commands())


def generate_injected(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(SAMPLES / "basic", destination)
    (destination / "textures/bad_header.png").write_bytes(b"NOT-A-PNG\x00\x01\x02")
    write_png(destination / "textures/wrong_size.png", texture(9991, 16))
    palette = Image.new("P", (32, 32))
    palette.putpalette([0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254)
    palette.save(destination / "textures/palette.png")
    injection = {
        "schema_version": 1,
        "injections": [
            {
                "id": "INJ-LOAD-001",
                "stage": "load",
                "action": "load_invalid_header",
                "target": "textures/bad_header.png",
                "expected_code": "E_TEXTURE_HEADER",
            },
            {
                "id": "INJ-LOAD-002",
                "stage": "load",
                "action": "load_wrong_size",
                "target": "textures/wrong_size.png",
                "expected_code": "E_TEXTURE_SIZE",
            },
            {
                "id": "INJ-LOAD-003",
                "stage": "load",
                "action": "load_unsupported_mode",
                "target": "textures/palette.png",
                "expected_code": "E_TEXTURE_MODE",
            },
            {
                "id": "INJ-LAYER-001",
                "stage": "layer_noise",
                "action": "delay_ms",
                "target_layer": 0,
                "delay_ms": 90,
                "expected_code": "E_RESPONSE_STALE",
            },
            {
                "id": "INJ-MAT-001",
                "stage": "material_mix",
                "action": "raise",
                "target_material": "painted_plastic",
                "expected_code": "E_MATERIAL_RUNTIME",
            },
            {
                "id": "INJ-SNAP-001",
                "stage": "snapshot",
                "action": "snapshot_hash_mismatch",
                "expected_code": "E_SNAPSHOT_VERSION",
            },
        ],
    }
    write_json(destination / "injection_config.json", injection)
    write_json(
        destination / "expected_codes.json",
        {row["id"]: row["expected_code"] for row in injection["injections"]},
    )
    write_json(
        destination / "expected_diagnostic_structure.json",
        {
            "report_type": "texture_material_generation_diagnostic",
            "schema_version": 1,
            "required_top_level": [
                "status",
                "codes",
                "issues",
                "layers",
                "materials",
                "final",
                "error_trace",
                "full_rebuild",
                "cache_ledger",
                "scheduler",
                "interactive",
                "injections",
                "snapshot",
                "artifacts",
            ],
            "expected_load_codes": ["E_TEXTURE_HEADER", "E_TEXTURE_SIZE", "E_TEXTURE_MODE"],
            "expected_runtime_codes": ["E_MATERIAL_RUNTIME"],
            "expected_latency_codes": ["E_RESPONSE_STALE"],
            "expected_snapshot_codes": ["E_SNAPSHOT_VERSION"],
            "injection_id_to_code": {
                row["id"]: row["expected_code"] for row in injection["injections"]
            },
        },
    )
    injected_config = base_config()
    injected_config["latency_budget_ms"] = 5
    write_json(destination / "config.json", injected_config)


def main(root: Path | None = None) -> None:
    global SAMPLES
    destination_root = (root or SAMPLES).resolve()
    SAMPLES = destination_root
    generate_basic(destination_root / "basic")
    generate_injected(destination_root / "injected")
    print(f"generated {destination_root / 'basic'} and {destination_root / 'injected'}")


if __name__ == "__main__":
    main()
