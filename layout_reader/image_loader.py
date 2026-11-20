"""PNG validation, deterministic grayscale decoding and cache artifacts."""

from __future__ import annotations

import base64
import io
import zlib
from typing import Any

import numpy as np
from PIL import Image

from .errors import E_IMAGE_INVALID, WorkbenchError
from .fingerprint import sha256_bytes


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_STAGE_VERSION = 1


def load_image_artifact(raw_bytes: bytes, validation: dict[str, Any]) -> dict[str, Any]:
    """Validate and decode PNG bytes into a reusable image-stage artifact."""

    if not raw_bytes.startswith(PNG_SIGNATURE):
        raise WorkbenchError(
            E_IMAGE_INVALID,
            "missing PNG container signature",
            "image",
            details={"received_bytes": len(raw_bytes)},
        )

    try:
        with Image.open(io.BytesIO(raw_bytes)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(raw_bytes)) as image:
            if image.format != "PNG":
                raise ValueError(f"decoded container is {image.format!r}, not PNG")
            width, height = image.size
            mode = image.mode
            expected_width = int(validation["width"])
            expected_height = int(validation["height"])
            if (width, height) != (expected_width, expected_height):
                raise WorkbenchError(
                    E_IMAGE_INVALID,
                    "image dimensions do not match configuration",
                    "image",
                    details={
                        "expected": [expected_width, expected_height],
                        "actual": [width, height],
                    },
                )
            if mode not in validation["allowed_modes"]:
                raise WorkbenchError(
                    E_IMAGE_INVALID,
                    "image pixel mode is not allowed",
                    "image",
                    details={"mode": mode, "allowed_modes": validation["allowed_modes"]},
                )
            image.load()
            gray_image = image if mode == "L" else image.convert("L")
            gray = np.asarray(gray_image, dtype=np.uint8).copy()
    except WorkbenchError:
        raise
    except Exception as exc:  # noqa: BLE001 - all decoder failures are page-local
        raise WorkbenchError(
            E_IMAGE_INVALID,
            f"unable to decode PNG: {exc}",
            "image",
            details={"received_bytes": len(raw_bytes)},
        ) from exc

    if gray.shape != (int(validation["height"]), int(validation["width"])):
        raise WorkbenchError(
            E_IMAGE_INVALID,
            "decoded grayscale array has unexpected shape",
            "image",
            details={"shape": list(gray.shape)},
        )

    compressed = zlib.compress(gray.tobytes(order="C"), level=9)
    return {
        "content_hash": sha256_bytes(raw_bytes),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "mode": mode,
        "gray_sha256": sha256_bytes(gray.tobytes(order="C")),
        "gray_zlib_b64": base64.b64encode(compressed).decode("ascii"),
    }


def gray_from_artifact(artifact: dict[str, Any]) -> np.ndarray:
    """Restore the exact grayscale array without touching the source PNG."""

    try:
        compressed = base64.b64decode(artifact["gray_zlib_b64"], validate=True)
        raw = zlib.decompress(compressed)
        gray = np.frombuffer(raw, dtype=np.uint8).reshape(
            int(artifact["height"]), int(artifact["width"])
        ).copy()
    except (KeyError, ValueError, zlib.error) as exc:
        raise WorkbenchError(
            E_IMAGE_INVALID,
            "persisted image artifact is corrupt",
            "image",
            details={"reason": str(exc)},
        ) from exc
    if sha256_bytes(gray.tobytes(order="C")) != artifact.get("gray_sha256"):
        raise WorkbenchError(
            E_IMAGE_INVALID,
            "persisted grayscale checksum mismatch",
            "image",
        )
    return gray
