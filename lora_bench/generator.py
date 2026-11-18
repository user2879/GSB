"""Fixed-seed generation of synthetic samples, tasks and frozen base weights.

Nothing here references a real model. All randomness flows through
:func:`lora_bench.utils.rng_from_seed` so the whole input set is reproducible
from the recorded seeds.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .utils import (
    fingerprint_array,
    hash_json,
    rng_from_seed,
    write_json_atomic,
)

VOCAB_SIZE = 32
EMBED_DIM = 8
HIDDEN_DIM = 8
LAYER_COUNT = 4


@dataclass(frozen=True)
class TaskSpec:
    name: str
    category: str
    metric: str
    threshold: float
    train_count: int
    val_count: int


DEFAULT_TASKS = (
    TaskSpec("cls_mod4", "classification", "accuracy", 0.6, 12, 6),
    TaskSpec("cls_parity", "classification", "accuracy", 0.55, 10, 6),
    TaskSpec("reg_sum", "regression", "mse", 0.65, 12, 6),
    TaskSpec("reg_cycle", "regression", "mse", 0.7, 10, 6),
)


def generate_samples(
    tasks: tuple[TaskSpec, ...] = DEFAULT_TASKS,
    seed: int = 20240501,
    vocab_size: int = VOCAB_SIZE,
) -> list[dict]:
    """Generate one deterministic JSONL-ready record per sample id."""
    samples: list[dict] = []
    for task_index, spec in enumerate(tasks):
        rng = rng_from_seed(seed + task_index * 97)
        total = spec.train_count + spec.val_count
        for local_index in range(total):
            length = int(rng.integers(4, 9))
            tokens = [int(x) for x in rng.integers(0, vocab_size, size=length)]
            token_sum = sum(tokens)
            if spec.category == "classification":
                if spec.name == "cls_mod4":
                    label = token_sum % 4
                else:
                    label = sum(t % 2 for t in tokens) % 2
                target_token = int(label)
                label_value: float | int = int(label)
            else:
                if spec.name == "reg_sum":
                    label_value = round(token_sum / (length * vocab_size), 6)
                else:
                    label_value = round(
                        float(np.sin(tokens[0] + tokens[-1])), 6
                    )
                target_token = int(abs(hash((spec.name, local_index)))) % vocab_size
            sample_id = f"{spec.name}-{local_index:03d}"
            samples.append(
                {
                    "sample_id": sample_id,
                    "task": spec.name,
                    "input_tokens": tokens,
                    "target_tokens": [target_token],
                    "label": label_value,
                }
            )
    return samples


def write_samples_jsonl(path: str, samples: list[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(__import__("json").dumps(sample, sort_keys=True))
            handle.write("\n")


def generate_task_manifest(
    samples: list[dict],
    tasks: tuple[TaskSpec, ...] = DEFAULT_TASKS,
) -> dict:
    grouped: dict[str, list[str]] = {}
    for sample in samples:
        grouped.setdefault(sample["task"], []).append(sample["sample_id"])
    entries = []
    for spec in tasks:
        ids = sorted(grouped.get(spec.name, []))
        entries.append(
            {
                "task": spec.name,
                "category": spec.category,
                "generation_target": spec.metric,
                "metric": spec.metric,
                "metric_threshold": spec.threshold,
                "train_sample_ids": ids[: spec.train_count],
                "val_sample_ids": ids[spec.train_count :],
            }
        )
    return {"version": 1, "tasks": entries}


def generate_base_pack(
    out_dir: str,
    version: str,
    seed: int,
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int = EMBED_DIM,
    hidden_dim: int = HIDDEN_DIM,
    layer_count: int = LAYER_COUNT,
) -> dict:
    """Write layer weight/bias raw float32 binaries plus shape descriptors."""
    os.makedirs(out_dir, exist_ok=True)
    rng = rng_from_seed(seed)
    shapes: list[dict] = []
    fingerprints: dict[str, str] = {}

    def dump(name: str, array: np.ndarray) -> None:
        flat = np.ascontiguousarray(array, dtype=np.float32)
        with open(os.path.join(out_dir, name), "wb") as handle:
            handle.write(flat.tobytes())
        shapes.append(
            {"name": name, "shape": list(flat.shape), "dtype": "float32"}
        )
        fingerprints[name] = fingerprint_array(flat)

    embedding = (rng.normal(0, 0.5, (vocab_size, embed_dim))).astype(np.float32)
    dump("embedding.bin", embedding)
    for layer in range(layer_count):
        scale = 0.35 / (layer + 1)
        weight = rng.normal(0, scale, (hidden_dim, hidden_dim)).astype(np.float32)
        bias = rng.normal(0, 0.05, hidden_dim).astype(np.float32)
        dump(f"layer{layer}.weight.bin", weight)
        dump(f"layer{layer}.bias.bin", bias)

    descriptor = {
        "base_version": version,
        "seed": seed,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "layer_count": layer_count,
        "activation": "tanh",
        "tensors": shapes,
    }
    descriptor["base_fingerprint"] = hash_json(
        {
            "base_version": version,
            "seed": seed,
            "fingerprints": fingerprints,
        }
    )
    write_json_atomic(os.path.join(out_dir, "shapes.json"), descriptor)
    return descriptor


def build_default_dataset(output_root: str, seed: int = 20240501) -> dict:
    """Materialise the canonical sample directory and two base versions."""
    samples = generate_samples(seed=seed)
    sample_path = os.path.join(output_root, "samples.jsonl")
    write_samples_jsonl(sample_path, samples)
    manifest = generate_task_manifest(samples)
    manifest_path = os.path.join(output_root, "tasks.json")
    write_json_atomic(manifest_path, manifest)
    base_v1 = generate_base_pack(
        os.path.join(output_root, "base_v1"), "base-v1", seed=1001
    )
    base_v2 = generate_base_pack(
        os.path.join(output_root, "base_v2"), "base-v2", seed=2002
    )
    return {
        "samples_path": sample_path,
        "tasks_path": manifest_path,
        "base_v1": base_v1,
        "base_v2": base_v2,
    }
