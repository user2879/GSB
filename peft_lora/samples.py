from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import E_TASK_INVALID, Failure


@dataclass(frozen=True)
class InstructionSample:
    sample_id: str
    sample_key: str
    target_layer: str
    token_ids: tuple[int, ...]


def load_samples(path: str | Path) -> tuple[list[InstructionSample], list[Failure]]:
    failures: list[Failure] = []
    samples: list[InstructionSample] = []
    seen_ids: set[str] = set()
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        return [], [Failure(E_TASK_INVALID, f"instruction JSONL cannot be read: {exc}")]
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        f"invalid sample JSON at line {line_number}: {exc}",
                        details={"line": line_number},
                    )
                )
                continue
            if not isinstance(raw, dict):
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        f"sample at line {line_number} must be an object",
                        details={"line": line_number},
                    )
                )
                continue
            sample_id = raw.get("sample_id")
            sample_key = raw.get("sample_key")
            target_layer = raw.get("target_layer")
            token_ids = raw.get("token_ids", raw.get("tokens"))
            if not isinstance(sample_id, str) or not sample_id.strip():
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        f"sample at line {line_number} has an invalid sample_id",
                        details={"line": line_number},
                    )
                )
                continue
            if sample_id in seen_ids:
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        f"duplicate sample_id: {sample_id}",
                        details={"line": line_number, "sample_id": sample_id},
                    )
                )
                continue
            seen_ids.add(sample_id)
            if not isinstance(sample_key, str) or not sample_key.strip():
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        f"sample {sample_id} has an invalid sample_key",
                        details={"line": line_number, "sample_id": sample_id},
                    )
                )
                continue
            if not isinstance(target_layer, str) or not target_layer.strip():
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        f"sample {sample_id} has an invalid target_layer",
                        details={"line": line_number, "sample_id": sample_id},
                    )
                )
                continue
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or any(
                    isinstance(token, bool)
                    or not isinstance(token, int)
                    or token < 0
                    or token > 255
                    for token in token_ids
                )
            ):
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        f"sample {sample_id} token_ids must be a non-empty controlled-vocabulary list",
                        task_name=None,
                        target_layer=target_layer,
                        details={"line": line_number, "sample_id": sample_id},
                    )
                )
                continue
            samples.append(
                InstructionSample(
                    sample_id=sample_id.strip(),
                    sample_key=sample_key.strip(),
                    target_layer=target_layer.strip(),
                    token_ids=tuple(token_ids),
                )
            )
    failures.sort(key=lambda failure: (failure.target_layer or "", failure.message))
    return samples, failures


def samples_for_layer(
    samples: list[InstructionSample], target_layer: str
) -> list[InstructionSample]:
    return [sample for sample in samples if sample.target_layer == target_layer]


def sample_set_hash(samples: list[InstructionSample]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            json.dumps(
                {
                    "sample_id": sample.sample_id,
                    "sample_key": sample.sample_key,
                    "target_layer": sample.target_layer,
                    "token_ids": list(sample.token_ids),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_uniform(seed: str, index: int, modulus: int = 2_000_003) -> float:
    material = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest()[:8]
    value = int.from_bytes(material, "little") % modulus
    return (value / (modulus - 1)) * 2.0 - 1.0


def encode_sample(
    sample: InstructionSample,
    task_name: str,
    input_dim: int,
    output_dim: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    key_hash = hashlib.sha256(sample.sample_key.encode("utf-8")).digest()
    key_offset = int.from_bytes(key_hash[:4], "little")
    x = np.zeros(input_dim, dtype=np.float64)
    for position, token in enumerate(sample.token_ids):
        index = (token * 13 + position * 7 + key_offset) % input_dim
        x[index] += 1.0 + (token % 7) * 0.125
    norm = float(np.linalg.norm(x))
    if norm > 0.0:
        x /= math.sqrt(norm)

    token_sum = sum(sample.token_ids)
    label_seed = f"{task_name}|{sample.target_layer}|{sample.sample_key}|{sample.sample_id}"
    y = np.empty(output_dim, dtype=np.float64)
    for index in range(output_dim):
        first = _stable_uniform(label_seed + "|a", index)
        second = _stable_uniform(label_seed + "|b", index)
        y[index] = 0.58 * first + 0.27 * math.sin(
            token_sum * 0.071 + index * 0.411 + 0.13 * second
        )
    return x, y

