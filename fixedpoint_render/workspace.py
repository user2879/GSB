"""Workspace scanning and input validation (business rule 1).

A workspace directory contains::

    config.json
    prompts/frame_XX.txt
    images/frame_XX.png
    motion/frame_XX.json   (block motion vector tables)
    commands.json          (optional edit command stream)

Validation produces structured ``E_`` errors with file + field locators.  When
any error exists, :class:`ErrorBundle` is raised and rendering never starts.
"""
from __future__ import annotations

import json
import os

from PIL import Image

from .config import CONDITION_TYPES, RenderConfig, load_config
from .errors import (
    E_COMMAND_PARSE, E_COMMAND_TARGET, E_DIM_MISMATCH, E_FILE_IO,
    E_FILE_MISSING, E_MOTION_FIELD, E_MOTION_PARSE, E_MOTION_TILES,
    E_PNG_HEADER, E_PNG_MODE, E_PNG_RESOLUTION, E_PROMPT_PARSE, E_VOCAB_TERM,
    ErrorBundle, RenderError, E_BUDGET_POSITIVE,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Command verbs supported by the edit stream.
COMMAND_VERBS = {
    "set_prompt", "set_image", "delete_image", "set_motion",
    "set_weight", "set_frame_budget", "set_qmode", "set_frac_bits",
}


def _frame_id(index: int) -> str:
    return f"frame_{index:02d}"


def discover_frames(workdir: str) -> list[int]:
    """Return sorted frame indices present in the workspace."""
    indices: set[int] = set()
    for sub in ("prompts", "images", "motion"):
        folder = os.path.join(workdir, sub)
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            stem, _ = os.path.splitext(name)
            if stem.startswith("frame_") and stem[6:].isdigit():
                indices.add(int(stem[6:]))
    return sorted(indices)


def _check_png_container(path: str, errors: list[RenderError]) -> bool:
    try:
        with open(path, "rb") as fh:
            signature = fh.read(8)
    except OSError as exc:
        errors.append(RenderError(E_FILE_IO, os.path.basename(path), "bytes",
                                  str(exc)))
        return False
    if signature != PNG_SIGNATURE:
        errors.append(RenderError(E_PNG_HEADER, os.path.basename(path),
                                  "signature", "not a PNG container"))
        return False
    return True


def load_png(path: str, config: RenderConfig, errors: list[RenderError]):
    name = os.path.basename(path)
    if not os.path.exists(path):
        errors.append(RenderError(E_FILE_MISSING, name, "image", "file missing"))
        return None
    if not _check_png_container(path, errors):
        return None
    try:
        with Image.open(path) as img:
            img.load()
            if img.mode != config.pixel_mode:
                errors.append(RenderError(E_PNG_MODE, name, "mode",
                                          f"got {img.mode}, want {config.pixel_mode}"))
            if img.size != (config.width, config.height):
                errors.append(RenderError(E_PNG_RESOLUTION, name, "size",
                                          f"got {img.size}, want "
                                          f"{(config.width, config.height)}"))
            if errors and any(e.location == name for e in errors):
                return None
            return img.copy()
    except OSError as exc:
        errors.append(RenderError(E_FILE_IO, name, "image", str(exc)))
        return None


def load_prompt(path: str, config: RenderConfig, errors: list[RenderError]) -> list[str] | None:
    name = os.path.basename(path)
    if not os.path.exists(path):
        errors.append(RenderError(E_FILE_MISSING, name, "prompt", "file missing"))
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        errors.append(RenderError(E_FILE_IO, name, "prompt", str(exc)))
        return None
    terms = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]
    if not terms:
        errors.append(RenderError(E_PROMPT_PARSE, name, "prompt",
                                  "prompt contains no terms"))
        return None
    vocab = set(config.vocab)
    for term in terms:
        if term not in vocab:
            errors.append(RenderError(E_VOCAB_TERM, name, f"prompt:{term}",
                                      f"term {term!r} outside controlled vocab"))
    return terms


def load_motion(path: str, config: RenderConfig,
                errors: list[RenderError]) -> dict | None:
    name = os.path.basename(path)
    if not os.path.exists(path):
        errors.append(RenderError(E_FILE_MISSING, name, "motion", "file missing"))
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(RenderError(E_MOTION_PARSE, name, "motion", str(exc)))
        return None
    if not isinstance(data, dict) or "tiles" not in data:
        errors.append(RenderError(E_MOTION_PARSE, name, "tiles",
                                  "motion table must contain a tiles array"))
        return None
    tiles = data["tiles"]
    if not isinstance(tiles, list) or len(tiles) != config.tile_rows:
        errors.append(RenderError(E_MOTION_TILES, name, "tiles",
                                  f"want {config.tile_rows} tile rows, got "
                                  f"{len(tiles) if isinstance(tiles, list) else 'n/a'}"))
        return None
    for row, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            errors.append(RenderError(E_MOTION_FIELD, name, f"tiles[{row}]",
                                      "tile must be an object"))
            continue
        for field in ("vx", "vy"):
            if field not in tile or not isinstance(tile[field], int) \
                    or isinstance(tile[field], bool):
                errors.append(RenderError(E_MOTION_FIELD, name,
                                          f"tiles[{row}].{field}",
                                          "must be an integer"))
        vec = tile.get("vector")
        if vec is not None:
            if not isinstance(vec, list) or len(vec) != config.condition_dim \
                    or not all(isinstance(v, int) and not isinstance(v, bool)
                               for v in vec):
                errors.append(RenderError(E_DIM_MISMATCH, name,
                                          f"tiles[{row}].vector",
                                          f"must be {config.condition_dim} ints"))
    if "frame_cover" in data:
        cover = data["frame_cover"]
        if not isinstance(cover, int) or isinstance(cover, bool) or not 0 <= cover <= 100:
            errors.append(RenderError(E_MOTION_FIELD, name, "frame_cover",
                                      "must be an integer percentage 0..100"))
    return data


def validate_commands(commands: list[dict], config: RenderConfig,
                      frame_count: int) -> list[RenderError]:
    errors: list[RenderError] = []
    location = "commands.json"
    for idx, cmd in enumerate(commands):
        if not isinstance(cmd, dict) or "verb" not in cmd:
            errors.append(RenderError(E_COMMAND_PARSE, location,
                                      f"[{idx}]", "command must have a verb"))
            continue
        verb = cmd["verb"]
        if verb not in COMMAND_VERBS:
            errors.append(RenderError(E_COMMAND_PARSE, location,
                                      f"[{idx}].verb", f"unknown verb {verb!r}"))
            continue
        frame = cmd.get("frame")
        frame_scoped = verb in {
            "set_prompt", "set_image", "delete_image", "set_motion",
            "set_frame_budget",
        }
        if frame_scoped:
            if not isinstance(frame, int) or isinstance(frame, bool) \
                    or not 0 <= frame < frame_count:
                errors.append(RenderError(E_COMMAND_TARGET, location,
                                          f"[{idx}].frame",
                                          f"frame out of range 0..{frame_count - 1}"))
        if verb == "set_weight":
            cond = cmd.get("condition")
            if cond not in CONDITION_TYPES:
                errors.append(RenderError(E_COMMAND_TARGET, location,
                                          f"[{idx}].condition",
                                          f"must be one of {CONDITION_TYPES}"))
        if verb == "set_qmode" and cmd.get("qmode") not in ("trunc", "nearest", "floor"):
            errors.append(RenderError(E_COMMAND_TARGET, location,
                                      f"[{idx}].qmode", "bad qmode"))
        if verb == "set_frac_bits" and cmd.get("frac_bits") not in (8, 12, 16):
            errors.append(RenderError(E_COMMAND_TARGET, location,
                                      f"[{idx}].frac_bits", "bad frac_bits"))
        if verb == "set_frame_budget" and (
                not isinstance(cmd.get("budget"), int)
                or isinstance(cmd.get("budget"), bool) or cmd.get("budget") <= 0):
            errors.append(RenderError(E_BUDGET_POSITIVE, location,
                                      f"[{idx}].budget", "must be positive int"))
    return errors


def load_commands(workdir: str, frame_count: int,
                  config: RenderConfig) -> list[dict]:
    path = os.path.join(workdir, "commands.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            commands = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise ErrorBundle([RenderError(E_COMMAND_PARSE, "commands.json",
                                       "commands", str(exc))])
    if not isinstance(commands, list):
        raise ErrorBundle([RenderError(E_COMMAND_PARSE, "commands.json",
                                       "commands", "must be a list")])
    errors = validate_commands(commands, config, frame_count)
    if errors:
        raise ErrorBundle(errors)
    return commands


def load_workspace(workdir: str) -> dict:
    """Scan and validate every input. Raises ErrorBundle before rendering."""
    config = load_config(os.path.join(workdir, "config.json"))
    frames = discover_frames(workdir)
    if not frames:
        raise ErrorBundle([RenderError(E_FILE_MISSING, workdir, "frames",
                                       "no frame_* inputs discovered")])
    if frames != list(range(len(frames))):
        raise ErrorBundle([RenderError(E_FILE_MISSING, "workspace", "frames",
                                       f"frame indices must be contiguous, got {frames}")])
    errors: list[RenderError] = []
    prompts, images, motions = {}, {}, {}
    for index in frames:
        fid = _frame_id(index)
        prompts[index] = load_prompt(
            os.path.join(workdir, "prompts", fid + ".txt"), config, errors)
        images[index] = load_png(
            os.path.join(workdir, "images", fid + ".png"), config, errors)
        motions[index] = load_motion(
            os.path.join(workdir, "motion", fid + ".json"), config, errors)
    commands = load_commands(workdir, len(frames), config)
    if errors:
        raise ErrorBundle(errors)
    return {
        "config": config,
        "frame_indices": frames,
        "prompts": prompts,
        "images": images,
        "motions": motions,
        "commands": commands,
    }
