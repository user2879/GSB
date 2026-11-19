from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import StudioConfig, WEIGHT_SCALE, load_commands, load_config, sha256_json
from .errors import E_CONFIG_INVALID, E_IMAGE_DECODE, StudioError, failure_item


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class LoadedImage:
    pixels: np.ndarray
    content_sha256: str
    raw_sha256: str


@dataclass(frozen=True)
class StyleStats:
    mean_q: tuple[int, int, int]
    std_q: tuple[int, int, int]
    content_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_q": list(self.mean_q),
            "std_q": list(self.std_q),
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class LoadedSample:
    sample_id: str
    directory: Path
    config: StudioConfig
    source: LoadedImage
    style: LoadedImage
    style_stats: StyleStats
    commands: list[dict[str, Any]]

    def sample_fingerprint(self) -> str:
        return sha256_json(
            {
                "source": self.source.content_sha256,
                "style": self.style.content_sha256,
                "config": self.config.fingerprint(),
            }
        )


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _decode_png(path: Path, width: int, height: int, label: str) -> LoadedImage:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise StudioError(E_IMAGE_DECODE, f"{label} image cannot be read: {exc}", {"path": str(path)}) from exc
    if len(payload) < len(PNG_SIGNATURE) or payload[:8] != PNG_SIGNATURE:
        raise StudioError(E_IMAGE_DECODE, f"{label} image does not have a valid PNG header", {"path": str(path)})
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format != "PNG":
                raise StudioError(E_IMAGE_DECODE, f"{label} image is not a PNG container", {"path": str(path)})
            opened.load()
            rgb = opened.convert("RGB")
    except StudioError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise StudioError(E_IMAGE_DECODE, f"{label} image cannot be decoded: {exc}", {"path": str(path)}) from exc
    if rgb.size != (width, height):
        raise StudioError(
            E_CONFIG_INVALID,
            f"{label} image resolution differs from config",
            {
                "path": str(path),
                "expected": [width, height],
                "actual": [rgb.size[0], rgb.size[1]],
            },
        )
    pixels = np.asarray(rgb, dtype=np.uint8).copy()
    return LoadedImage(
        pixels=pixels,
        content_sha256=array_sha256(pixels),
        raw_sha256=hashlib.sha256(payload).hexdigest(),
    )


def compute_style_stats(image: LoadedImage) -> StyleStats:
    pixels = image.pixels.astype(np.int64)
    flattened = pixels.reshape(-1, pixels.shape[2])
    count = flattened.shape[0]
    sums = flattened.sum(axis=0, dtype=np.int64)
    sums_squared = np.square(flattened, dtype=np.int64).sum(axis=0, dtype=np.int64)
    means: list[int] = []
    deviations: list[int] = []
    for channel in range(3):
        total = int(sums[channel])
        total_squared = int(sums_squared[channel])
        mean_q = (total * WEIGHT_SCALE + count // 2) // count
        variance_numerator = max(0, total_squared * count - total * total)
        variance_q = (variance_numerator * WEIGHT_SCALE + count * count // 2) // (count * count)
        std_q = math.isqrt(variance_q * WEIGHT_SCALE)
        means.append(mean_q)
        deviations.append(std_q)
    return StyleStats(tuple(means), tuple(deviations), image.content_sha256)


def load_sample(directory: Path) -> LoadedSample:
    config = load_config(directory / "config.json")
    source = _decode_png(directory / config.image_file, config.width, config.height, "source")
    style = _decode_png(directory / config.style_file, config.width, config.height, "style")
    commands = load_commands(directory / config.commands_file)
    for index, command in enumerate(commands):
        operation = command.get("op", command.get("type"))
        if not isinstance(operation, str):
            raise StudioError(E_CONFIG_INVALID, f"command {index} has no string op")
    sample_id = directory.name
    return LoadedSample(
        sample_id=sample_id,
        directory=directory,
        config=config,
        source=source,
        style=style,
        style_stats=compute_style_stats(style),
        commands=commands,
    )


def discover_sample_directories(root: Path) -> list[Path]:
    if not root.is_dir():
        raise StudioError(E_CONFIG_INVALID, f"sample root is not a directory: {root}")
    if (root / "config.json").is_file():
        return [root]
    candidates = sorted(
        (entry for entry in root.iterdir() if entry.is_dir() and (entry / "config.json").is_file()),
        key=lambda entry: entry.name,
    )
    if not candidates:
        raise StudioError(E_CONFIG_INVALID, f"no sample directories found under: {root}")
    return candidates


def load_sample_set(root: Path) -> tuple[list[LoadedSample], list[dict[str, Any]]]:
    loaded: list[LoadedSample] = []
    failures: list[dict[str, Any]] = []
    directories = discover_sample_directories(root)
    for directory in directories:
        try:
            loaded.append(load_sample(directory))
        except StudioError as error:
            failures.append(failure_item(directory.name, error))
        except (OSError, ValueError, TypeError) as error:
            wrapped = StudioError(E_CONFIG_INVALID, f"unexpected sample load failure: {error}")
            failures.append(failure_item(directory.name, wrapped))
    failures.sort(key=lambda item: (item["sample_id"], item["code"], item["message"]))
    return loaded, failures


