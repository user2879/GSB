"""Deterministic sample loading and strict, auditable validation.

Every invalid entry produces a fixed business code and is skipped rather than
silently corrected. Validation conclusions are returned as CheckItem lists so
they can be reviewed entry by entry.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from .codes import BizCode, CheckItem
from .config import RenderConfig

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_vocabulary(path: str | Path) -> tuple[list[str], list[CheckItem]]:
    checks: list[CheckItem] = []
    terms: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            token = raw.strip()
            if not token or token.startswith("#"):
                continue
            if token in seen:
                checks.append(CheckItem("vocabulary", index, BizCode.E_PARSE,
                                        f"duplicate term {token!r}", False))
                continue
            seen.add(token)
            terms.append(token)
    if not terms:
        checks.append(CheckItem("vocabulary", None, BizCode.E_PARSE,
                                "empty vocabulary", False))
    return terms, checks


def load_prompts(path: str | Path, vocabulary: list[str]) -> tuple[list[list[str]], list[CheckItem]]:
    vocab = set(vocabulary)
    accepted: list[list[str]] = []
    checks: list[CheckItem] = []
    with open(path, "r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            terms = [tok for tok in line.split() if tok]
            if not terms:
                continue
            oov = [tok for tok in terms if tok not in vocab]
            if oov:
                checks.append(CheckItem(
                    "prompts", index, BizCode.E_PROMPT_OOV,
                    f"out-of-vocabulary terms={oov}", False))
                continue
            accepted.append(terms)
            checks.append(CheckItem("prompts", index, BizCode.OK,
                                    f"{len(terms)} terms", True))
    return accepted, checks


def inspect_png_container(path: str | Path, cfg: RenderConfig) -> list[CheckItem]:
    checks: list[CheckItem] = []
    raw = Path(path).read_bytes()
    if not raw.startswith(PNG_SIGNATURE):
        checks.append(CheckItem(str(path), None, BizCode.E_PNG_HEADER,
                                "missing PNG signature", False))
        return checks
    checks.append(CheckItem(str(path), 0, BizCode.OK, "png signature", True))
    offset = len(PNG_SIGNATURE)
    ihdr_seen = False
    declared_width = declared_height = None
    bit_depth = color_type = None
    while offset + 8 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        chunk_type = raw[offset + 4:offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        if data_end + 4 > len(raw):
            checks.append(CheckItem(str(path), None, BizCode.E_PNG_HEADER,
                                    "truncated chunk", False))
            break
        chunk_data = raw[data_start:data_end]
        stored_crc = struct.unpack(">I", raw[data_end:data_end + 4])[0]
        if zlib.crc32(raw[offset + 4:data_end]) & 0xFFFFFFFF != stored_crc:
            checks.append(CheckItem(str(path), None, BizCode.E_PNG_HEADER,
                                    f"crc mismatch in {chunk_type!r}", False))
        if chunk_type == b"IHDR":
            ihdr_seen = True
            declared_width, declared_height, bit_depth, color_type, *_ = \
                struct.unpack(">IIBBBBB", chunk_data)
        if chunk_type == b"IEND":
            break
        offset = data_end + 4
    if not ihdr_seen:
        checks.append(CheckItem(str(path), None, BizCode.E_PNG_HEADER,
                                "missing IHDR", False))
        return checks
    if (declared_width, declared_height) != (cfg.image_width, cfg.image_height):
        checks.append(CheckItem(
            str(path), None, BizCode.E_PNG_SIZE,
            f"size={declared_width}x{declared_height} "
            f"expected={cfg.image_width}x{cfg.image_height}", False))
    else:
        checks.append(CheckItem(str(path), 1, BizCode.OK,
                                f"size {declared_width}x{declared_height}", True))
    # color_type 2 = RGB (Pillow mode "RGB"), 8-bit
    if bit_depth != 8 or color_type != 2:
        checks.append(CheckItem(str(path), None, BizCode.E_PNG_MODE,
                                f"bit_depth={bit_depth} color_type={color_type}",
                                False))
    else:
        checks.append(CheckItem(str(path), 2, BizCode.OK,
                                "8-bit RGB", True))
    try:
        with Image.open(path) as image:
            if image.mode != "RGB":
                checks.append(CheckItem(str(path), None, BizCode.E_PNG_MODE,
                                        f"Pillow mode={image.mode}", False))
            if image.size != (cfg.image_width, cfg.image_height):
                checks.append(CheckItem(str(path), None, BizCode.E_PNG_SIZE,
                                        f"Pillow size={image.size}", False))
    except Exception as exc:
        checks.append(CheckItem(str(path), None, BizCode.E_PNG_HEADER,
                                f"Pillow rejected image: {exc}", False))
    return checks


def load_image_condition(path: str | Path, cfg: RenderConfig
                         ) -> tuple[np.ndarray | None, list[CheckItem]]:
    checks = inspect_png_container(path, cfg)
    if any(not item.accepted and item.code in (
            BizCode.E_PNG_HEADER, BizCode.E_PNG_MODE, BizCode.E_PNG_SIZE)
            for item in checks):
        return None, checks
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return array, checks


def load_motion_table(path: str | Path, cfg: RenderConfig
                      ) -> tuple[np.ndarray | None, list[CheckItem]]:
    checks: list[CheckItem] = []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, [CheckItem(str(path), None, BizCode.E_PARSE, str(exc), False)]
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None, [CheckItem(str(path), None, BizCode.E_MOTION_SHAPE,
                                "missing rows", False)]
    expected_cols = cfg.tile_cols
    table: list[list[int]] = []
    for r_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != expected_cols:
            checks.append(CheckItem(str(path), r_index, BizCode.E_MOTION_SHAPE,
                                    f"row width={None if not isinstance(row, list) else len(row)}",
                                    False))
            continue
        clean_row = []
        bad = False
        for c_index, cell in enumerate(row):
            if not isinstance(cell, dict) or set(cell) != {"dx", "dy"}:
                checks.append(CheckItem(str(path), r_index, BizCode.E_MOTION_SHAPE,
                                        f"cell=({r_index},{c_index})", False))
                bad = True
                break
            dx, dy = cell["dx"], cell["dy"]
            if not (isinstance(dx, int) and isinstance(dy, int)
                    and not isinstance(dx, bool) and not isinstance(dy, bool)):
                checks.append(CheckItem(str(path), r_index, BizCode.E_MOTION_VALUE,
                                        f"non-integer cell=({r_index},{c_index})",
                                        False))
                bad = True
                break
            clean_row.append([dx, dy])
        if not bad:
            table.append(clean_row)
    if len(table) != cfg.tile_rows:
        checks.append(CheckItem(str(path), None, BizCode.E_MOTION_SHAPE,
                                f"rows={len(table)} expected={cfg.tile_rows}",
                                False))
        return None, checks
    checks.append(CheckItem(str(path), None, BizCode.OK,
                            f"motion {cfg.tile_rows}x{cfg.tile_cols}", True))
    return np.asarray(table, dtype=np.int64), checks


def load_edit_stream(path: str | Path) -> tuple[list[dict], list[CheckItem]]:
    checks: list[CheckItem] = []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [], [CheckItem(str(path), None, BizCode.E_PARSE, str(exc), False)]
    commands = data.get("commands") if isinstance(data, dict) else data
    allowed = {
        "replace_prompt", "replace_image", "delete_image", "adjust_motion",
        "set_weight", "set_denoise_steps", "set_tile_rows", "set_quant",
        "set_exchange_version",
    }
    accepted: list[dict] = []
    if not isinstance(commands, list):
        return [], [CheckItem(str(path), None, BizCode.E_PARSE,
                              "commands must be a list", False)]
    for index, command in enumerate(commands):
        if not isinstance(command, dict) or "op" not in command:
            checks.append(CheckItem(str(path), index, BizCode.E_BAD_COMMAND,
                                    "missing op", False))
            continue
        if command["op"] not in allowed:
            checks.append(CheckItem(str(path), index, BizCode.E_BAD_COMMAND,
                                    f"op={command['op']!r}", False))
            continue
        checks.append(CheckItem(str(path), index, BizCode.OK,
                                command["op"], True))
        accepted.append(dict(command))
    return accepted, checks
