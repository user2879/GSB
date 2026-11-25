from __future__ import annotations

import csv
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .config import RenderConfig
from .errors import InputInvalid
from .hashing import file_fingerprint, fingerprint_json
from .miniyaml import load_yaml


PNG_SIGNATURE = bytes((0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A))


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]
    index: dict[str, int]
    fingerprint: str


@dataclass
class ConditionBundle:
    vocab: Vocabulary
    prompt_tokens: tuple[str, ...]
    image: np.ndarray | None
    image_path: Path | None
    image_fingerprint: str | None
    motion: np.ndarray
    motion_path: Path
    motion_fingerprint: str
    source_fingerprints: dict[str, str]

    def condition_fingerprint(self) -> str:
        return fingerprint_json(
            {
                "prompt_tokens": list(self.prompt_tokens),
                "vocab_fingerprint": self.vocab.fingerprint,
                "image_fingerprint": self.image_fingerprint,
                "motion_fingerprint": self.motion_fingerprint,
                "motion_array_fingerprint": _array_fp(self.motion),
            }
        )


def _array_fp(array: np.ndarray) -> str:
    from .hashing import array_fingerprint

    return array_fingerprint(array)


def load_vocabulary(path: str | Path, expected_dim: int | None = None) -> Vocabulary:
    path = Path(path)
    field_name = "conditions.vocab_path"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputInvalid(f"cannot read vocabulary: {exc}", field_name=field_name) from exc
    tokens: list[str] = []
    errors: dict[str, str] = {}
    for line_number, raw in enumerate(lines, 1):
        token = raw.strip()
        if not token or token.startswith("#"):
            continue
        if any(ch.isspace() for ch in token):
            errors[f"vocab.line_{line_number}"] = "vocabulary tokens cannot contain whitespace"
        tokens.append(token)
    duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
    if duplicates:
        errors["vocab.duplicates"] = f"duplicate vocabulary tokens: {','.join(duplicates)}"
    if expected_dim is not None and len(tokens) != expected_dim:
        errors["text_dim"] = (
            f"text_dim={expected_dim} does not match vocabulary size {len(tokens)}"
        )
    if errors:
        raise InputInvalid("vocabulary validation rejected whole file", fields=errors)
    unique = tuple(tokens)
    return Vocabulary(
        tokens=unique,
        index={token: index for index, token in enumerate(unique)},
        fingerprint=fingerprint_json(list(unique)),
    )


def validate_prompt(tokens: Iterable[str], vocab: Vocabulary) -> None:
    unknown = sorted({token for token in tokens if token not in vocab.index})
    if unknown:
        raise InputInvalid(
            "prompt contains terms outside controlled vocabulary",
            field_name="conditions.prompt_tokens",
            details={"unknown_terms": unknown},
        )


def load_png_condition(
    path: str | Path, expected_width: int, expected_height: int
) -> tuple[np.ndarray, str, dict[str, Any]]:
    path = Path(path)
    field_name = "conditions.image_path"
    errors: dict[str, str] = {}
    try:
        header = path.read_bytes()[:33]
    except OSError as exc:
        raise InputInvalid(f"cannot read PNG: {exc}", field_name=field_name) from exc

    if len(header) < 33 or header[:8] != PNG_SIGNATURE:
        errors["png_signature"] = "invalid PNG container signature"
    else:
        try:
            chunk_length = struct.unpack(">I", header[8:12])[0]
            chunk_type = header[12:16]
            if chunk_type != b"IHDR" or chunk_length != 13:
                errors["png_ihdr"] = "first chunk must be IHDR with length 13"
            else:
                width, height, bit_depth, color_type, compression, filter_method, interlace = (
                    struct.unpack(">IIBBBBB", header[16:29])
                )
                stored_crc = struct.unpack(">I", header[29:33])[0]
                actual_crc = zlib.crc32(header[12:29]) & 0xFFFFFFFF
                if stored_crc != actual_crc:
                    errors["png_crc"] = "IHDR CRC mismatch"
                if (width, height) != (expected_width, expected_height):
                    errors["image_resolution"] = (
                        f"expected {expected_width}x{expected_height}, got {width}x{height}"
                    )
                if bit_depth != 8 or color_type != 2:
                    errors["image_mode"] = (
                        f"expected 8-bit RGB (bit_depth=8,color_type=2), got "
                        f"bit_depth={bit_depth},color_type={color_type}"
                    )
                if compression != 0 or filter_method != 0:
                    errors["png_container"] = "unsupported PNG compression/filter method"
                if interlace != 0:
                    errors["png_interlace"] = "interlaced PNG conditions are not accepted"
        except struct.error as exc:
            errors["png_ihdr"] = f"truncated IHDR: {exc}"

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.mode != "RGB":
                errors.setdefault("image_mode", f"Pillow reports mode {image.mode}, expected RGB")
            if image.size != (expected_width, expected_height):
                errors.setdefault(
                    "image_resolution",
                    f"Pillow reports {image.size}, expected "
                    f"{expected_width}x{expected_height}",
                )
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except InputInvalid:
        raise
    except Exception as exc:
        errors["png_decode"] = f"Pillow could not decode PNG: {exc}"
        pixels = None

    if errors:
        raise InputInvalid(
            "image condition rejected as a whole file",
            field_name=field_name,
            fields=errors,
            details={"path": str(path)},
        )
    assert pixels is not None
    return pixels, file_fingerprint(path), {
        "width": expected_width,
        "height": expected_height,
        "mode": "RGB",
    }


def load_motion_table(config: RenderConfig) -> tuple[np.ndarray, str]:
    path = config.resolve(config.conditions.motion_path)
    assert path is not None
    field_name = "conditions.motion_path"
    errors: dict[str, str] = {}
    if not path.exists():
        raise InputInvalid(f"motion table does not exist: {path}", field_name=field_name)

    block_rows = config.frame_height // config.motion_block_h
    block_cols = config.frame_width // config.motion_block_w
    expected_count = block_rows * block_cols
    vectors: dict[tuple[int, int], tuple[int, int]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"block_row", "block_col", "dx_q8", "dy_q8"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise InputInvalid(
                    "motion CSV must contain block_row,block_col,dx_q8,dy_q8",
                    field_name=field_name,
                )
            for line_number, row in enumerate(reader, 2):
                try:
                    br = int(row["block_row"])
                    bc = int(row["block_col"])
                    dx = int(row["dx_q8"])
                    dy = int(row["dy_q8"])
                except (TypeError, ValueError) as exc:
                    errors[f"motion.line_{line_number}"] = f"non-integer field: {exc}"
                    continue
                if not (0 <= br < block_rows and 0 <= bc < block_cols):
                    errors[f"motion.line_{line_number}"] = (
                        f"block ({br},{bc}) outside {block_rows}x{block_cols}"
                    )
                    continue
                if (br, bc) in vectors:
                    errors[f"motion.line_{line_number}"] = f"duplicate block ({br},{bc})"
                    continue
                if abs(dx) > 32 * 256 or abs(dy) > 32 * 256:
                    errors[f"motion.line_{line_number}"] = "motion vector exceeds +/-32 pixels"
                vectors[(br, bc)] = (dx, dy)
    except InputInvalid:
        raise
    except OSError as exc:
        raise InputInvalid(f"cannot read motion table: {exc}", field_name=field_name) from exc

    if len(vectors) != expected_count:
        errors["motion_block_count"] = (
            f"expected {expected_count} vectors for {block_rows}x{block_cols} blocks, "
            f"found {len(vectors)}"
        )
    if errors:
        raise InputInvalid(
            "motion vector table rejected as a whole file",
            field_name=field_name,
            fields=errors,
        )

    table = np.zeros((block_rows, block_cols, 2), dtype=np.int64)
    for br in range(block_rows):
        for bc in range(block_cols):
            table[br, bc] = vectors[(br, bc)]
    return table, file_fingerprint(path)


def load_conditions(config: RenderConfig) -> ConditionBundle:
    vocab_path = config.resolve(config.conditions.vocab_path)
    if vocab_path is None or not vocab_path.exists():
        raise InputInvalid(
            "vocabulary file does not exist", field_name="conditions.vocab_path"
        )
    vocab = load_vocabulary(vocab_path, expected_dim=config.text_dim)
    validate_prompt(config.conditions.prompt_tokens, vocab)

    image = None
    image_fp = None
    image_path = config.resolve(config.conditions.image_path)
    if config.conditions.image_path is not None:
        if image_path is None or not image_path.exists():
            raise InputInvalid(
                "image condition does not exist", field_name="conditions.image_path"
            )
        image, image_fp, _ = load_png_condition(
            image_path, config.image_width, config.image_height
        )

    motion, motion_fp = load_motion_table(config)
    if config.motion_dim != motion.shape[0] * motion.shape[1] * 2:
        raise InputInvalid(
            "motion vector dimension does not match configuration",
            field_name="motion_dim",
            details={"actual": int(motion.size), "configured": config.motion_dim},
        )

    return ConditionBundle(
        vocab=vocab,
        prompt_tokens=config.conditions.prompt_tokens,
        image=image,
        image_path=image_path,
        image_fingerprint=image_fp,
        motion=motion,
        motion_path=config.resolve(config.conditions.motion_path),  # type: ignore[arg-type]
        motion_fingerprint=motion_fp,
        source_fingerprints={
            "vocab": vocab.fingerprint,
            "image": image_fp or "absent",
            "motion": motion_fp,
        },
    )


def load_command_stream(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    try:
        if path.suffix.lower() == ".jsonl":
            commands = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            data = load_yaml(path)
            commands = data.get("commands", data) if isinstance(data, dict) else data
    except Exception as exc:
        raise InputInvalid(
            f"cannot parse command stream: {exc}", field_name="command_file"
        ) from exc
    if not isinstance(commands, list) or not commands:
        raise InputInvalid(
            "command stream must be a non-empty list", field_name="command_file"
        )
    for index, command in enumerate(commands):
        if not isinstance(command, dict) or "op" not in command:
            raise InputInvalid(
                f"command {index} must be a mapping with op",
                field_name=f"commands[{index}]",
            )
    return commands
