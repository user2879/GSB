"""Sorted page discovery, PNG decoding, truth validation, and fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ParseConfig
from .png import PNGError, content_fingerprint, decode_png
from .validation import PageInvalid, normalize_truth


@dataclass
class PageInput:
    file_name: str
    stem: str
    image_path: Path
    truth_path: Path
    page_id: str
    image_info: dict[str, Any]
    gray: np.ndarray
    truth: dict[str, Any]
    content_fingerprint: str
    candidates: list[dict[str, Any]]
    detection_point_cost: int
    sort_key: str


@dataclass
class InvalidPage:
    file_name: str
    stem: str
    image_path: Path | None
    truth_path: Path | None
    detail_code: str
    message: str
    business_code: str = "E_PAGE_INVALID"
    sort_key: str = ""


def discover_pairs(sample_dir: str | Path) -> list[tuple[Path, Path]]:
    root = Path(sample_dir)
    if not root.is_dir():
        raise PageInvalid("E_INPUT_DIRECTORY", f"sample directory not found: {root}")
    image_paths = sorted(
        [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ],
        key=lambda path: path.name,
    )
    pairs: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        pairs.append((image_path, root / f"{image_path.stem}.json"))
    return pairs


def load_page(image_path: Path, truth_path: Path, config: ParseConfig) -> PageInput:
    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        raise PageInvalid("E_PAGE_READ", f"cannot read PNG: {exc}") from exc
    try:
        gray, image_info = decode_png(image_bytes)
    except PNGError as exc:
        raise PageInvalid("E_PNG_CONTAINER", str(exc)) from exc

    if config.expected_image_width is not None and gray.shape[1] != config.expected_image_width:
        raise PageInvalid(
            "E_IMAGE_SIZE",
            f"width {gray.shape[1]} != expected {config.expected_image_width}",
        )
    if config.expected_image_height is not None and gray.shape[0] != config.expected_image_height:
        raise PageInvalid(
            "E_IMAGE_SIZE",
            f"height {gray.shape[0]} != expected {config.expected_image_height}",
        )
    if (
        config.expected_pixel_mode is not None
        and image_info["pixel_mode"] != config.expected_pixel_mode
    ):
        raise PageInvalid(
            "E_IMAGE_MODE",
            f"pixel mode {image_info['pixel_mode']} != expected {config.expected_pixel_mode}",
        )

    try:
        truth_text = truth_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PageInvalid("E_TRUTH_READ", f"cannot read truth JSON: {exc}") from exc
    try:
        truth_raw = json.loads(truth_text)
    except json.JSONDecodeError as exc:
        raise PageInvalid("E_TRUTH_JSON", f"truth is not valid JSON: {exc}") from exc

    if isinstance(truth_raw, dict):
        declared_image = truth_raw.get("image")
        if declared_image is not None and declared_image != image_path.name:
            raise PageInvalid(
                "E_TRUTH_IMAGE_MISMATCH",
                f"truth image {declared_image!r} != {image_path.name!r}",
            )
        declared_size = truth_raw.get("image_size")
        if declared_size is not None:
            if (
                not isinstance(declared_size, list)
                or len(declared_size) != 2
                or declared_size != [gray.shape[1], gray.shape[0]]
            ):
                raise PageInvalid(
                    "E_TRUTH_IMAGE_SIZE",
                    f"truth image_size {declared_size!r} != {[gray.shape[1], gray.shape[0]]}",
                )
        declared_mode = truth_raw.get("pixel_mode")
        if declared_mode is not None and declared_mode != image_info["pixel_mode"]:
            raise PageInvalid(
                "E_TRUTH_IMAGE_MODE",
                f"truth pixel_mode {declared_mode!r} != {image_info['pixel_mode']!r}",
            )

    truth = normalize_truth(truth_raw, gray.shape[1], gray.shape[0])
    fingerprint = content_fingerprint(gray)
    page_id = truth.get("page_id", image_path.stem)
    if not isinstance(page_id, str) or not page_id:
        raise PageInvalid("E_TRUTH_SCHEMA", "page_id must be a non-empty string")
    return PageInput(
        file_name=image_path.name,
        stem=image_path.stem,
        image_path=image_path,
        truth_path=truth_path,
        page_id=page_id,
        image_info=image_info,
        gray=gray,
        truth=truth,
        content_fingerprint=fingerprint,
        candidates=[],
        detection_point_cost=0,
        sort_key=f"{image_path.name}|page",
    )


def load_pages(
    sample_dir: str | Path, config: ParseConfig, loader: Any
) -> tuple[list[PageInput], list[InvalidPage]]:
    pairs = discover_pairs(sample_dir)
    valid: list[PageInput] = []
    invalid: list[InvalidPage] = []
    futures = [
        loader.submit(load_page, image_path, truth_path, config)
        for image_path, truth_path in pairs
    ]
    for (image_path, truth_path), future in zip(pairs, futures):
        try:
            valid.append(future.result())
        except PageInvalid as exc:
            invalid.append(
                InvalidPage(
                    file_name=image_path.name,
                    stem=image_path.stem,
                    image_path=image_path,
                    truth_path=truth_path,
                    detail_code=exc.detail_code,
                    message=exc.message,
                    sort_key=f"{image_path.name}|page",
                )
            )
        except Exception as exc:  # defensive: invalid input must not stop a batch
            invalid.append(
                InvalidPage(
                    file_name=image_path.name,
                    stem=image_path.stem,
                    image_path=image_path,
                    truth_path=truth_path,
                    detail_code="E_PAGE_LOAD",
                    message=f"{type(exc).__name__}: {exc}",
                    sort_key=f"{image_path.name}|page",
                )
            )
    valid.sort(key=lambda page: page.file_name)
    invalid.sort(key=lambda page: page.file_name)
    return valid, invalid
