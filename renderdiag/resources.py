from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import Config, VocabularyTerm
from .errors import E_CONFIG_INVALID, fail
from .util import fingerprint, is_int, load_json, to_decimal, to_int


@dataclass(frozen=True)
class TextPrompt:
    name: str
    entities: tuple[str, ...]
    attributes: tuple[str, ...]

    def sequence(self) -> tuple[str, ...]:
        return self.entities + self.attributes

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "name": self.name,
                "entities": list(self.entities),
                "attributes": list(self.attributes),
            },
            "text-prompt-v1",
        )


@dataclass(frozen=True)
class PromptSet:
    prompts: dict[str, TextPrompt]
    source_path: str
    vocabulary: tuple[VocabularyTerm, ...]

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            [self.prompts[name].fingerprint for name in sorted(self.prompts)],
            "prompt-set-v1",
        )


@dataclass(frozen=True)
class ImageCondition:
    path: str
    width: int
    height: int
    mode: str
    pixels: np.ndarray
    content_fingerprint: str


@dataclass(frozen=True)
class MotionTable:
    path: str
    values: np.ndarray
    content_fingerprint: str

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.values.shape)


@dataclass(frozen=True)
class EditCommand:
    index: int
    op: str
    payload: dict[str, Any]
    source: str
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "op": self.op, "payload": self.payload}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    stage: str
    kind: str
    target: dict[str, Any]
    value: Any

    def matches(self, stage: str, kind: str, target: dict[str, Any] | None = None) -> bool:
        if self.stage != stage or self.kind != kind:
            return False
        if not self.target or target is None:
            return True
        return all(target.get(key) == value for key, value in self.target.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.injection_id,
            "stage": self.stage,
            "kind": self.kind,
            "target": self.target,
            "value": self.value,
        }


def _require_mapping(value: Any, entry: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(E_CONFIG_INVALID, entry, "expected an object", detail={"value": repr(value)})
    return value


def _require_string(value: Any, entry: str) -> str:
    if not isinstance(value, str) or not value:
        fail(E_CONFIG_INVALID, entry, "expected a non-empty string", detail={"value": repr(value)})
    return value


def _string_list(value: Any, entry: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        fail(E_CONFIG_INVALID, entry, "expected a string list", detail={"value": repr(value)})
    result: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item:
            fail(E_CONFIG_INVALID, f"{entry}[{idx}]", "expected a non-empty string")
        result.append(item)
    return tuple(result)


def load_prompt_set(path: str | os.PathLike[str], config: Config) -> PromptSet:
    raw = load_json(path, "resources.prompts")
    root = _require_mapping(raw, "prompts")
    prompts_raw = _require_mapping(root.get("prompts"), "prompts.prompts")
    if not prompts_raw:
        fail(E_CONFIG_INVALID, "prompts.prompts", "at least one prompt is required")
    by_term = {item.term: item for item in config.vocabulary}
    prompts: dict[str, TextPrompt] = {}
    for name in sorted(prompts_raw):
        if not isinstance(name, str) or not name:
            fail(E_CONFIG_INVALID, "prompts.prompts", "prompt names must be non-empty strings")
        item = _require_mapping(prompts_raw[name], f"prompts.prompts.{name}")
        entities = _string_list(item.get("entities", []), f"prompts.prompts.{name}.entities")
        attributes = _string_list(item.get("attributes", []), f"prompts.prompts.{name}.attributes")
        if not entities and not attributes:
            fail(
                E_CONFIG_INVALID,
                f"prompts.prompts.{name}",
                "prompt must contain at least one entity or attribute",
            )
        for kind, items in (("entity", entities), ("attribute", attributes)):
            for idx, term in enumerate(items):
                entry = f"prompts.prompts.{name}.{kind}s[{idx}]"
                vocab_item = by_term.get(term)
                if vocab_item is None:
                    fail(
                        E_CONFIG_INVALID,
                        entry,
                        "prompt term is not in the controlled vocabulary",
                        detail={"term": term},
                    )
                if vocab_item.kind != kind:
                    fail(
                        E_CONFIG_INVALID,
                        entry,
                        "prompt term kind does not match its vocabulary category",
                        detail={"term": term, "expected_kind": kind, "actual_kind": vocab_item.kind},
                    )
        prompts[name] = TextPrompt(name=name, entities=entities, attributes=attributes)
    return PromptSet(
        prompts=prompts,
        source_path=str(Path(path).resolve()),
        vocabulary=config.vocabulary,
    )


def _parse_png_header(path: str) -> tuple[int, int, int, int, int, int, int]:
    try:
        with open(path, "rb") as fh:
            header = fh.read(33)
    except FileNotFoundError:
        fail(E_CONFIG_INVALID, "image.path", "PNG file does not exist", detail={"path": path})
    if len(header) < 33 or header[:8] != b"\x89PNG\r\n\x1a\n":
        fail(E_CONFIG_INVALID, "image.header", "invalid PNG signature", detail={"path": path})
    if int.from_bytes(header[8:12], "big") != 13 or header[12:16] != b"IHDR":
        fail(E_CONFIG_INVALID, "image.header", "PNG must start with a 13-byte IHDR", detail={"path": path})
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    bit_depth = header[24]
    color_type = header[25]
    compression = header[26]
    filter_method = header[27]
    interlace = header[28]
    return width, height, bit_depth, color_type, compression, filter_method, interlace


def load_image(path: str | os.PathLike[str], config: Config) -> ImageCondition:
    source = str(Path(path).resolve())
    width, height, bit_depth, color_type, compression, filter_method, interlace = _parse_png_header(source)
    if width != config.width or height != config.height:
        fail(
            E_CONFIG_INVALID,
            "image.resolution",
            "PNG dimensions do not match the configured frame size",
            detail={"expected": [config.width, config.height], "actual": [width, height]},
        )
    if bit_depth != 8 or color_type != 2:
        fail(
            E_CONFIG_INVALID,
            "image.pixel_mode",
            "PNG must be 8-bit RGB (color type 2)",
            detail={"bit_depth": bit_depth, "color_type": color_type},
        )
    if compression != 0 or filter_method != 0 or interlace != 0:
        fail(
            E_CONFIG_INVALID,
            "image.png_header",
            "unsupported PNG compression, filter or interlace method",
            detail={"compression": compression, "filter": filter_method, "interlace": interlace},
        )
    try:
        with Image.open(source) as image:
            if image.format != "PNG":
                fail(E_CONFIG_INVALID, "image.format", "image is not a PNG", detail={"format": image.format})
            image.verify()
        with Image.open(source) as image:
            if image.mode != "RGB":
                fail(
                    E_CONFIG_INVALID,
                    "image.pixel_mode",
                    "decoded PNG mode must be RGB",
                    detail={"mode": image.mode},
                )
            if image.size != (config.width, config.height):
                fail(E_CONFIG_INVALID, "image.resolution", "decoded PNG size mismatch")
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as exc:
        fail(E_CONFIG_INVALID, "image.container", "PNG container could not be decoded", detail={"error": str(exc)})
    content_fp = fingerprint(
        {
            "width": int(pixels.shape[1]),
            "height": int(pixels.shape[0]),
            "mode": "RGB",
            "pixels_sha256": fingerprint(pixels.tobytes().hex(), "image-pixels-v1"),
        },
        "image-condition-v1",
    )
    return ImageCondition(
        path=source,
        width=config.width,
        height=config.height,
        mode="RGB",
        pixels=pixels,
        content_fingerprint=content_fp,
    )


def load_motion_table(path: str | os.PathLike[str], config: Config) -> MotionTable:
    source = str(Path(path).resolve())
    expected_shape = (
        config.frame_count,
        config.motion_block_rows,
        config.motion_block_cols,
        2,
    )
    try:
        values = np.load(source, allow_pickle=False)
    except FileNotFoundError:
        fail(E_CONFIG_INVALID, "motion.path", "motion table does not exist", detail={"path": source})
    except (OSError, ValueError) as exc:
        fail(E_CONFIG_INVALID, "motion.container", "motion table could not be loaded", detail={"error": str(exc)})
    if not isinstance(values, np.ndarray) or values.dtype.kind not in "iu":
        fail(
            E_CONFIG_INVALID,
            "motion.dtype",
            "motion table must be a signed or unsigned integer array",
            detail={"dtype": str(getattr(values, "dtype", type(values)))},
        )
    if values.shape != expected_shape:
        fail(
            E_CONFIG_INVALID,
            "motion.shape",
            "motion table shape is incompatible with the configured frame and block dimensions",
            detail={"expected": list(expected_shape), "actual": list(values.shape)},
        )
    values = values.astype(np.int64, copy=True)
    content_fp = fingerprint(
        {"shape": list(values.shape), "values": values.reshape(-1).tolist()},
        "motion-table-v1",
    )
    return MotionTable(path=source, values=values, content_fingerprint=content_fp)


_COMMAND_OPS = {
    "set_prompt",
    "set_image",
    "delete_image",
    "set_motion",
    "set_weights",
    "set_denoise_steps",
    "set_block_rows",
    "set_quant_level",
    "set_guidance",
    "set_seed",
    "set_batch_budget",
}


def _validate_command_payload(op: str, payload: dict[str, Any], entry: str) -> None:
    if op == "set_prompt":
        _require_string(payload.get("prompt"), f"{entry}.prompt")
    elif op in {"set_image", "set_motion"}:
        _require_string(payload.get("path"), f"{entry}.path")
    elif op == "delete_image":
        return
    elif op == "set_weights":
        weights = _require_mapping(payload.get("weights"), f"{entry}.weights")
        if not weights:
            fail(E_CONFIG_INVALID, f"{entry}.weights", "at least one weight is required")
        for name, value in weights.items():
            if name not in {"text", "image", "motion"}:
                fail(E_CONFIG_INVALID, f"{entry}.weights.{name}", "unknown fusion weight")
            to_decimal(value, f"{entry}.weights.{name}", nonnegative=True)
    elif op in {"set_denoise_steps", "set_block_rows", "set_batch_budget"}:
        key = {
            "set_denoise_steps": "steps",
            "set_block_rows": "rows",
            "set_batch_budget": "points",
        }[op]
        to_int(payload.get(key), f"{entry}.{key}", minimum=1)
    elif op == "set_quant_level":
        to_int(payload.get("level"), f"{entry}.level", minimum=1)
    elif op == "set_guidance":
        to_decimal(payload.get("guidance"), f"{entry}.guidance", nonnegative=True)
    elif op == "set_seed":
        to_int(payload.get("seed"), f"{entry}.seed", minimum=0)


def load_command_stream(path: str | os.PathLike[str]) -> list[EditCommand]:
    source = str(Path(path).resolve())
    try:
        lines = Path(source).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        fail(E_CONFIG_INVALID, "commands", "command stream does not exist", detail={"path": source})
    commands: list[EditCommand] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        entry = f"commands[{index}]"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(
                E_CONFIG_INVALID,
                entry,
                "invalid command JSON",
                detail={"line": exc.lineno, "column": exc.colno},
            )
        if not isinstance(raw, dict):
            fail(E_CONFIG_INVALID, entry, "command must be an object")
        op = raw.get("op")
        if not isinstance(op, str) or not op:
            fail(E_CONFIG_INVALID, f"{entry}.op", "command op must be a non-empty string")
        if op not in _COMMAND_OPS:
            fail(E_CONFIG_INVALID, f"{entry}.op", "unknown command op", detail={"op": op})
        payload = {key: value for key, value in raw.items() if key != "op"}
        _validate_command_payload(op, payload, entry)
        commands.append(EditCommand(index=index, op=op, payload=payload, source=source, raw=raw))
    return commands


_INJECTION_RULES: dict[str, set[str]] = {
    "text_encode": {"dimension_mismatch"},
    "image_encode": {"dimension_mismatch"},
    "motion_encode": {"dimension_mismatch"},
    "fusion": {"dimension_mismatch"},
    "init": {"nonfinite_latent"},
    "denoise": {"nonfinite_latent"},
    "decode_row": {"block_row_oob"},
    "quantization": {"quant_level_oob"},
    "snapshot": {"stale_snapshot_version"},
}


def load_injections(path: str | os.PathLike[str] | None) -> tuple[Injection, ...]:
    if path is None:
        return ()
    raw = load_json(path, "injections")
    if not isinstance(raw, list):
        fail(E_CONFIG_INVALID, "injections", "injection list must be an array")
    result: list[Injection] = []
    seen: set[str] = set()
    for idx, item_raw in enumerate(raw):
        entry = f"injections[{idx}]"
        item = _require_mapping(item_raw, entry)
        injection_id = _require_string(item.get("id"), f"{entry}.id")
        stage = _require_string(item.get("stage"), f"{entry}.stage")
        kind = _require_string(item.get("kind"), f"{entry}.kind")
        if injection_id in seen:
            fail(E_CONFIG_INVALID, f"{entry}.id", "duplicate injection id", detail={"id": injection_id})
        if stage not in _INJECTION_RULES:
            fail(E_CONFIG_INVALID, f"{entry}.stage", "unknown injection stage", detail={"stage": stage})
        if kind not in _INJECTION_RULES[stage]:
            fail(
                E_CONFIG_INVALID,
                f"{entry}.kind",
                "injection kind is not valid for this stage",
                detail={"stage": stage, "kind": kind},
            )
        target_raw = item.get("target", {})
        if not isinstance(target_raw, dict):
            fail(E_CONFIG_INVALID, f"{entry}.target", "target must be an object")
        target: dict[str, Any] = {}
        for key, value in target_raw.items():
            if key not in {"frame", "step", "row"}:
                fail(E_CONFIG_INVALID, f"{entry}.target.{key}", "unknown target selector")
            target[key] = to_int(value, f"{entry}.target.{key}", minimum=0)
        seen.add(injection_id)
        result.append(
            Injection(
                injection_id=injection_id,
                stage=stage,
                kind=kind,
                target=target,
                value=item.get("value"),
            )
        )
    return tuple(result)
