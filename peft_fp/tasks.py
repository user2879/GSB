"""Task manifest and instruction-sample loading, validation and the
deterministic mapping from token sequences to input/target vectors."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from .errors import E_TASK_INVALID, failure

VOCAB_SIZE = 64


@dataclass(frozen=True)
class Task:
    name: str
    target_layer: str
    rank: int
    alpha: float

    @property
    def sort_key(self):
        return (self.name, self.target_layer)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class Sample:
    sample_id: str
    sample_key: str
    target_layer: str
    tokens: tuple


def load_tasks(path: str, whitelist):
    """Returns (valid_tasks sorted by name, failures)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return [], [failure(E_TASK_INVALID, f"task manifest unreadable: {exc}")]
    entries = raw.get("tasks") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return [], [failure(E_TASK_INVALID, "task manifest is not a list")]

    tasks, failures, seen = [], [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append(failure(E_TASK_INVALID, "task entry is not an object"))
            continue
        name = entry.get("name")
        layer = entry.get("target_layer")
        rank = entry.get("rank")
        alpha = entry.get("alpha")
        if not isinstance(name, str) or not name:
            failures.append(failure(E_TASK_INVALID, "task missing name"))
            continue
        if name in seen:
            failures.append(failure(E_TASK_INVALID, "duplicate task name", task=name))
            continue
        seen.add(name)
        if not isinstance(layer, str) or layer not in whitelist:
            failures.append(
                failure(E_TASK_INVALID, f"target layer {layer!r} not in whitelist", task=name, layer=layer)
            )
            continue
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            failures.append(failure(E_TASK_INVALID, "rank must be a positive integer", task=name, layer=layer))
            continue
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            failures.append(failure(E_TASK_INVALID, "alpha must be a positive finite number", task=name, layer=layer))
            continue
        tasks.append(Task(name=name, target_layer=layer, rank=rank, alpha=float(alpha)))
    tasks.sort(key=lambda t: t.sort_key)
    return tasks, failures


def load_samples(path: str, known_layers):
    """Returns (samples sorted by sample_id, failures)."""
    samples, failures, seen = [], [], set()
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError as exc:
        return [], [failure(E_TASK_INVALID, f"samples unreadable: {exc}")]
    with fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append(failure(E_TASK_INVALID, f"line {lineno}: bad json: {exc}"))
                continue
            sid = entry.get("sample_id")
            skey = entry.get("sample_key")
            layer = entry.get("target_layer")
            tokens = entry.get("tokens")
            if not isinstance(sid, str) or not sid:
                failures.append(failure(E_TASK_INVALID, f"line {lineno}: missing sample_id"))
                continue
            if sid in seen:
                failures.append(failure(E_TASK_INVALID, f"line {lineno}: duplicate sample_id", layer=layer))
                continue
            seen.add(sid)
            if not isinstance(layer, str) or layer not in known_layers:
                failures.append(failure(E_TASK_INVALID, f"line {lineno}: unknown layer {layer!r}", layer=layer))
                continue
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(not isinstance(t, int) or isinstance(t, bool) or t < 0 or t >= VOCAB_SIZE for t in tokens)
            ):
                failures.append(
                    failure(E_TASK_INVALID, f"line {lineno}: tokens outside vocab [0,{VOCAB_SIZE})", layer=layer)
                )
                continue
            samples.append(Sample(sid, str(skey), layer, tuple(tokens)))
    samples.sort(key=lambda s: s.sample_id)
    return samples, failures


def token_embedding(token: int, dim: int) -> np.ndarray:
    """Deterministic per-token vector in [-1, 1); no global random state."""
    out = np.empty(dim, dtype=np.float64)
    for i in range(dim):
        digest = hashlib.sha256(f"tok|{token}|{i}".encode()).digest()
        out[i] = int.from_bytes(digest[:8], "little") / float(1 << 64) * 2.0 - 1.0
    return out


def sample_input(sample: Sample, in_dim: int) -> np.ndarray:
    acc = np.zeros(in_dim, dtype=np.float64)
    for token in sample.tokens:
        acc += token_embedding(token, in_dim)
    return acc / len(sample.tokens)


def load_teacher(path: str):
    """Teacher deltas per layer: {layer: {"A": [[..]], "B": [[..]], "scale": float}}."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return raw.get("layers", {})


def sample_target(sample: Sample, layer, teacher_delta: np.ndarray | None) -> np.ndarray:
    """y = (W + teacher_delta) x + b, all float64 (quantized later by caller)."""
    x = sample_input(sample, layer.shape[1])
    w = layer.weight if teacher_delta is None else layer.weight + teacher_delta
    return w @ x + layer.bias


def teacher_delta_matrix(teacher_spec, shape) -> np.ndarray | None:
    if not teacher_spec:
        return None
    a = np.asarray(teacher_spec["A"], dtype=np.float64)
    b = np.asarray(teacher_spec["B"], dtype=np.float64)
    scale = float(teacher_spec.get("scale", 1.0))
    delta = scale * (b @ a)
    if delta.shape != tuple(shape):
        return None
    return delta
