"""Stable hashing helpers for documents and animation content."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .models import Animation


def canonical_json_bytes(value: Any, *, allow_nan: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=allow_nan,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any, *, allow_nan: bool = False) -> str:
    return sha256_bytes(canonical_json_bytes(value, allow_nan=allow_nan))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def animation_semantic_document(animation: Animation) -> dict[str, Any]:
    return {
        "joints": list(animation.joint_names),
        "times": [float(value) for value in animation.times],
        "rotations_wxyz": [
            [[float(value) for value in quat] for quat in frame] for frame in animation.rotations
        ],
        "root_translation": [
            [float(value) for value in translation] for translation in animation.root_translation
        ],
        "interpolation": animation.interpolation,
    }


def animation_semantic_sha256(animation: Animation) -> str:
    return sha256_json(animation_semantic_document(animation))
