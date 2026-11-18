"""Parsing and strict validation of samples, task manifests and base packs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .errors import (
    E_BASE_FINITE_MISMATCH,
    E_BASE_SHAPE_MISMATCH,
    E_SAMPLE_INVALID,
    E_TASK_LIST_INVALID,
    WorkbenchError,
    failure_record,
)
from .utils import hash_json, read_json
from .utils import fingerprint_array

REQUIRED_SAMPLE_FIELDS = (
    "sample_id",
    "task",
    "input_tokens",
    "target_tokens",
    "label",
)


def _is_int_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(token, int) and not isinstance(token, bool) for token in value)
    )


def load_samples(
    path: str,
    allowed_tasks: set[str],
    vocab_size: int,
) -> tuple[list[dict], list[dict]]:
    """Parse JSONL line by line, returning valid samples and failure records.

    Invalid lines never abort the whole load: they are classified with
    ``E_SAMPLE_INVALID`` and appended to the failure list. Sample-id uniqueness
    is enforced globally across the file.
    """
    samples: list[dict] = []
    failures: list[dict] = []
    seen_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: malformed JSON: {exc}",
                        line_no=line_no,
                    )
                )
                continue
            missing = [f for f in REQUIRED_SAMPLE_FIELDS if f not in record]
            if missing:
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: missing fields {sorted(missing)}",
                        line_no=line_no,
                    )
                )
                continue
            sample_id = record["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: empty sample id",
                        line_no=line_no,
                    )
                )
                continue
            if sample_id in seen_ids:
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: duplicate sample id {sample_id}",
                        line_no=line_no,
                        sample_id=sample_id,
                    )
                )
                continue
            if record["task"] not in allowed_tasks:
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: unknown task {record['task']}",
                        line_no=line_no,
                        sample_id=sample_id,
                    )
                )
                continue
            if not _is_int_list(record["input_tokens"]):
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: input_tokens must be a non-empty int list",
                        line_no=line_no,
                        sample_id=sample_id,
                    )
                )
                continue
            if not _is_int_list(record["target_tokens"]):
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: target_tokens must be a non-empty int list",
                        line_no=line_no,
                        sample_id=sample_id,
                    )
                )
                continue
            out_of_range = [
                token
                for token in (*record["input_tokens"], *record["target_tokens"])
                if token < 0 or token >= vocab_size
            ]
            if out_of_range:
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: tokens outside [0,{vocab_size})",
                        line_no=line_no,
                        sample_id=sample_id,
                    )
                )
                continue
            if not isinstance(record["label"], (int, float)) or isinstance(
                record["label"], bool
            ):
                failures.append(
                    failure_record(
                        E_SAMPLE_INVALID,
                        "load",
                        f"line {line_no}: label must be numeric",
                        line_no=line_no,
                        sample_id=sample_id,
                    )
                )
                continue
            seen_ids.add(sample_id)
            samples.append(record)
    samples.sort(key=lambda item: item["sample_id"])
    return samples, failures


def load_task_manifest(path: str) -> dict:
    try:
        manifest = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(
            E_TASK_LIST_INVALID, f"cannot read task manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), list):
        raise WorkbenchError(
            E_TASK_LIST_INVALID, "task manifest must contain a tasks list"
        )
    required = ("task", "train_sample_ids", "val_sample_ids")
    for entry in manifest["tasks"]:
        missing = [key for key in required if key not in entry]
        if missing:
            raise WorkbenchError(
                E_TASK_LIST_INVALID,
                f"task entry missing fields {sorted(missing)}",
            )
        if not isinstance(entry["train_sample_ids"], list) or not isinstance(
            entry["val_sample_ids"], list
        ):
            raise WorkbenchError(
                E_TASK_LIST_INVALID,
                f"task {entry.get('task')}: id lists must be arrays",
            )
    names = [entry["task"] for entry in manifest["tasks"]]
    if len(set(names)) != len(names):
        raise WorkbenchError(E_TASK_LIST_INVALID, "duplicate task names")
    manifest["tasks"].sort(key=lambda entry: entry["task"])
    return manifest


@dataclass
class BasePack:
    """In-memory frozen base model loaded from raw binary tensor files."""

    base_version: str
    base_fingerprint: str
    embedding: np.ndarray
    layer_weights: list[np.ndarray]
    layer_biases: list[np.ndarray]
    activation: str
    hidden_dim: int
    descriptor: dict

    @property
    def layer_count(self) -> int:
        return len(self.layer_weights)

    def embed(self, tokens: list[int]) -> np.ndarray:
        # Mean embedding gives a fixed-width input regardless of sequence length.
        return self.embedding[np.asarray(tokens, dtype=np.int64)].mean(axis=0)

    def forward(self, tokens: list[int]) -> dict:
        """Forward pass with fixed layer order and fixed tanh activation."""
        hidden = self.embed(tokens)
        cache: dict[str, np.ndarray] = {"embed": hidden.copy()}
        for layer in range(self.layer_count):
            cache[f"layer{layer}.input"] = hidden.copy()
            pre = self.layer_weights[layer] @ hidden + self.layer_biases[layer]
            if self.activation == "tanh":
                hidden = np.tanh(pre)
            elif self.activation == "relu":
                hidden = np.maximum(pre, 0.0)
            else:
                hidden = pre
            cache[f"layer{layer}.pre"] = pre
            cache[f"layer{layer}.output"] = hidden.copy()
        cache["final"] = hidden.copy()
        return cache


def load_base_pack(directory: str) -> BasePack:
    descriptor_path = os.path.join(directory, "shapes.json")
    try:
        descriptor = read_json(descriptor_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkbenchError(
            E_BASE_SHAPE_MISMATCH, f"missing shapes.json: {exc}"
        ) from exc
    expected = {item["name"]: item for item in descriptor["tensors"]}
    tensors: dict[str, np.ndarray] = {}
    for name, meta in expected.items():
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            raise WorkbenchError(
                E_BASE_SHAPE_MISMATCH, f"tensor file missing: {name}"
            )
        raw = np.fromfile(path, dtype=np.float32)
        shape = tuple(meta["shape"])
        if raw.size != int(np.prod(shape)):
            raise WorkbenchError(
                E_BASE_SHAPE_MISMATCH,
                f"{name}: file size {raw.size} != declared shape {shape}",
            )
        array = raw.reshape(shape)
        if not np.all(np.isfinite(array)):
            raise WorkbenchError(
                E_BASE_FINITE_MISMATCH, f"{name}: contains non-finite values"
            )
        tensors[name] = array
    recomputed = hash_json(
        {
            "base_version": descriptor["base_version"],
            "seed": descriptor["seed"],
            "fingerprints": {
                name: fingerprint_array(array)
                for name, array in sorted(tensors.items())
            },
        }
    )
    if recomputed != descriptor["base_fingerprint"]:
        raise WorkbenchError(
            E_BASE_SHAPE_MISMATCH,
            "base fingerprint does not match tensor contents",
        )
    weights = [
        tensors[f"layer{idx}.weight.bin"]
        for idx in range(descriptor["layer_count"])
    ]
    biases = [
        tensors[f"layer{idx}.bias.bin"]
        for idx in range(descriptor["layer_count"])
    ]
    return BasePack(
        base_version=descriptor["base_version"],
        base_fingerprint=descriptor["base_fingerprint"],
        embedding=tensors["embedding.bin"],
        layer_weights=weights,
        layer_biases=biases,
        activation=descriptor.get("activation", "tanh"),
        hidden_dim=descriptor["hidden_dim"],
        descriptor=descriptor,
    )
