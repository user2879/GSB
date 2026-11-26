from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from .errors import E_CONFIG_INVALID, fail
from .util import decimal_text, fingerprint, to_decimal, to_int


_TERM_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True)
class VocabularyTerm:
    term: str
    kind: str


@dataclass(frozen=True)
class ResourcePaths:
    prompts: str
    image: str | None
    motion: str

    def as_dict(self) -> dict[str, str | None]:
        return {"prompts": self.prompts, "image": self.image, "motion": self.motion}


@dataclass(frozen=True)
class Config:
    source_path: str
    source_dir: str
    width: int
    height: int
    channels: int
    frame_count: int
    latent_height: int
    latent_width: int
    latent_channels: int
    condition_dim: int
    frac_bits: int
    allowed_frac_bits: tuple[int, ...]
    quantization_level: int
    allowed_quantization_levels: tuple[int, ...]
    denoise_steps: int
    seed: int
    noise_seed: int
    guidance: Decimal
    batch_point_budget: int
    block_rows: int
    max_concurrency: int
    interaction_latency_budget_ms: int
    image_block_size: int
    motion_block_width: int
    motion_block_height: int
    fusion_weights: tuple[tuple[str, Decimal], ...]
    vocabulary: tuple[VocabularyTerm, ...]
    resources: ResourcePaths
    snapshot_version: int

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def motion_block_rows(self) -> int:
        return self.height // self.motion_block_height

    @property
    def motion_block_cols(self) -> int:
        return self.width // self.motion_block_width

    @property
    def image_block_rows(self) -> int:
        return self.height // self.image_block_size

    @property
    def image_block_cols(self) -> int:
        return self.width // self.image_block_size

    @property
    def vocabulary_terms(self) -> tuple[str, ...]:
        return tuple(item.term for item in self.vocabulary)

    @property
    def vocabulary_fingerprint(self) -> str:
        return fingerprint(
            [{"term": item.term, "kind": item.kind} for item in self.vocabulary],
            "vocabulary-v1",
        )

    @property
    def config_fingerprint(self) -> str:
        return fingerprint(self.normalized_dict(), "config-v1")

    def weights_dict(self) -> dict[str, Decimal]:
        return {name: value for name, value in self.fusion_weights}

    def normalized_dict(self) -> dict[str, Any]:
        return {
            "frame": {
                "width": self.width,
                "height": self.height,
                "channels": self.channels,
                "count": self.frame_count,
            },
            "latent": {
                "height": self.latent_height,
                "width": self.latent_width,
                "channels": self.latent_channels,
            },
            "condition": {
                "dim": self.condition_dim,
                "frac_bits": self.frac_bits,
                "allowed_frac_bits": list(self.allowed_frac_bits),
                "quantization_level": self.quantization_level,
                "allowed_quantization_levels": list(self.allowed_quantization_levels),
            },
            "denoise": {
                "steps": self.denoise_steps,
                "seed": self.seed,
                "noise_seed": self.noise_seed,
                "guidance": decimal_text(self.guidance),
                "batch_point_budget": self.batch_point_budget,
            },
            "render": {
                "block_rows": self.block_rows,
                "max_concurrency": self.max_concurrency,
                "interaction_latency_budget_ms": self.interaction_latency_budget_ms,
            },
            "image": {"block_size": self.image_block_size},
            "motion": {
                "block_width": self.motion_block_width,
                "block_height": self.motion_block_height,
                "block_rows": self.motion_block_rows,
                "block_cols": self.motion_block_cols,
            },
            "fusion": {
                "weights": {name: decimal_text(value) for name, value in self.fusion_weights}
            },
            "vocabulary": [
                {"term": item.term, "kind": item.kind} for item in self.vocabulary
            ],
            "snapshot_version": self.snapshot_version,
        }


def _require_mapping(value: Any, entry: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(E_CONFIG_INVALID, entry, "expected a mapping", detail={"value": repr(value)})
    return value


def _required(mapping: dict[str, Any], key: str, entry: str) -> Any:
    if key not in mapping:
        fail(E_CONFIG_INVALID, f"{entry}.{key}", "required field is missing")
    return mapping[key]


def _enum_int(
    value: Any,
    entry: str,
    allowed: tuple[int, ...],
) -> int:
    result = to_int(value, entry)
    if result not in allowed:
        fail(
            E_CONFIG_INVALID,
            entry,
            "value is not in the allowed enumeration",
            detail={"value": result, "allowed": list(allowed)},
        )
    return result


def _int_tuple(value: Any, entry: str, *, positive_required: bool = True) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        fail(E_CONFIG_INVALID, entry, "expected a non-empty integer list", detail={"value": repr(value)})
    values = tuple(to_int(item, f"{entry}[{idx}]", minimum=1 if positive_required else None) for idx, item in enumerate(value))
    if len(set(values)) != len(values):
        fail(E_CONFIG_INVALID, entry, "enumeration values must be unique")
    return values


def load_config(path: str | os.PathLike[str]) -> Config:
    source = str(Path(path).resolve())
    entry_root = "config"
    try:
        with open(source, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        fail(E_CONFIG_INVALID, entry_root, "file does not exist", detail={"path": source})
    except yaml.YAMLError as exc:
        fail(E_CONFIG_INVALID, entry_root, "invalid YAML", detail={"error": str(exc)})

    root = _require_mapping(raw, entry_root)
    frame = _require_mapping(_required(root, "frame", entry_root), "frame")
    latent = _require_mapping(_required(root, "latent", entry_root), "latent")
    condition = _require_mapping(_required(root, "condition", entry_root), "condition")
    denoise = _require_mapping(_required(root, "denoise", entry_root), "denoise")
    render = _require_mapping(_required(root, "render", entry_root), "render")
    image = _require_mapping(_required(root, "image", entry_root), "image")
    motion = _require_mapping(_required(root, "motion", entry_root), "motion")
    fusion = _require_mapping(_required(root, "fusion", entry_root), "fusion")
    cache = _require_mapping(root.get("cache", {}), "cache")
    resources_raw = _require_mapping(_required(root, "resources", entry_root), "resources")

    width = to_int(_required(frame, "width", "frame"), "frame.width", minimum=1)
    height = to_int(_required(frame, "height", "frame"), "frame.height", minimum=1)
    channels = to_int(_required(frame, "channels", "frame"), "frame.channels", minimum=1)
    frame_count = to_int(_required(frame, "count", "frame"), "frame.count", minimum=1)
    if channels != 3:
        fail(E_CONFIG_INVALID, "frame.channels", "only RGB (3 channels) is supported", detail={"value": channels})

    latent_height = to_int(_required(latent, "height", "latent"), "latent.height", minimum=1)
    latent_width = to_int(_required(latent, "width", "latent"), "latent.width", minimum=1)
    latent_channels = to_int(_required(latent, "channels", "latent"), "latent.channels", minimum=1)
    condition_dim = to_int(_required(condition, "dim", "condition"), "condition.dim", minimum=1)

    allowed_frac_bits = _int_tuple(
        condition.get("allowed_frac_bits", [4, 8, 12]),
        "condition.allowed_frac_bits",
    )
    frac_bits = _enum_int(
        _required(condition, "frac_bits", "condition"),
        "condition.frac_bits",
        allowed_frac_bits,
    )
    allowed_quantization_levels = _int_tuple(
        condition.get("allowed_quantization_levels", [1, 2, 4]),
        "condition.allowed_quantization_levels",
    )
    quantization_level = _enum_int(
        _required(condition, "quantization_level", "condition"),
        "condition.quantization_level",
        allowed_quantization_levels,
    )

    denoise_steps = to_int(_required(denoise, "steps", "denoise"), "denoise.steps", minimum=1)
    seed = to_int(_required(denoise, "seed", "denoise"), "denoise.seed", minimum=0)
    noise_seed = to_int(denoise.get("noise_seed", seed), "denoise.noise_seed", minimum=0)
    guidance = to_decimal(_required(denoise, "guidance", "denoise"), "denoise.guidance", nonnegative=True)
    batch_point_budget = to_int(
        denoise.get("batch_point_budget", 1),
        "denoise.batch_point_budget",
        minimum=1,
    )

    block_rows = to_int(_required(render, "block_rows", "render"), "render.block_rows", minimum=1)
    max_concurrency = to_int(
        _required(render, "max_concurrency", "render"),
        "render.max_concurrency",
        minimum=1,
    )
    interaction_latency_budget_ms = to_int(
        render.get("interaction_latency_budget_ms", 100),
        "render.interaction_latency_budget_ms",
        minimum=1,
    )
    if height % block_rows != 0:
        fail(
            E_CONFIG_INVALID,
            "render.block_rows",
            "block row count must divide frame height",
            detail={"block_rows": block_rows, "frame_height": height},
        )

    image_block_size = to_int(
        _required(image, "block_size", "image"),
        "image.block_size",
        minimum=1,
    )
    if width % image_block_size != 0 or height % image_block_size != 0:
        fail(
            E_CONFIG_INVALID,
            "image.block_size",
            "image block size must divide frame width and height",
            detail={"block_size": image_block_size, "width": width, "height": height},
        )

    motion_block_width = to_int(
        _required(motion, "block_width", "motion"),
        "motion.block_width",
        minimum=1,
    )
    motion_block_height = to_int(
        _required(motion, "block_height", "motion"),
        "motion.block_height",
        minimum=1,
    )
    if width % motion_block_width != 0 or height % motion_block_height != 0:
        fail(
            E_CONFIG_INVALID,
            "motion.block_width",
            "motion block dimensions must divide frame dimensions",
            detail={
                "block_width": motion_block_width,
                "block_height": motion_block_height,
                "width": width,
                "height": height,
            },
        )

    weights_raw = _require_mapping(_required(fusion, "weights", "fusion"), "fusion.weights")
    weights: list[tuple[str, Decimal]] = []
    for name in ("text", "image", "motion"):
        if name not in weights_raw:
            fail(E_CONFIG_INVALID, f"fusion.weights.{name}", "required fusion weight is missing")
        weights.append(
            (
                name,
                to_decimal(weights_raw[name], f"fusion.weights.{name}", nonnegative=True),
            )
        )
    unknown_weight_names = sorted(set(weights_raw) - {"text", "image", "motion"})
    if unknown_weight_names:
        fail(
            E_CONFIG_INVALID,
            f"fusion.weights.{unknown_weight_names[0]}",
            "unknown fusion weight entry",
        )

    vocabulary_raw = _required(root, "vocabulary", entry_root)
    if not isinstance(vocabulary_raw, list) or not vocabulary_raw:
        fail(E_CONFIG_INVALID, "vocabulary", "expected a non-empty vocabulary list")
    vocabulary: list[VocabularyTerm] = []
    seen_terms: set[str] = set()
    for idx, item_raw in enumerate(vocabulary_raw):
        item = _require_mapping(item_raw, f"vocabulary[{idx}]")
        term = _required(item, "term", f"vocabulary[{idx}]")
        kind = _required(item, "kind", f"vocabulary[{idx}]")
        if not isinstance(term, str) or not _TERM_RE.match(term):
            fail(
                E_CONFIG_INVALID,
                f"vocabulary[{idx}].term",
                "term must match ^[a-z][a-z0-9_-]*$",
                detail={"value": repr(term)},
            )
        if term in seen_terms:
            fail(E_CONFIG_INVALID, f"vocabulary[{idx}].term", "duplicate vocabulary term", detail={"term": term})
        if kind not in {"entity", "attribute"}:
            fail(
                E_CONFIG_INVALID,
                f"vocabulary[{idx}].kind",
                "kind must be entity or attribute",
                detail={"value": repr(kind)},
            )
        seen_terms.add(term)
        vocabulary.append(VocabularyTerm(term=term, kind=kind))

    if not any(item.kind == "entity" for item in vocabulary):
        fail(E_CONFIG_INVALID, "vocabulary", "vocabulary must contain at least one entity")
    if not any(item.kind == "attribute" for item in vocabulary):
        fail(E_CONFIG_INVALID, "vocabulary", "vocabulary must contain at least one attribute")

    prompts_path = _required(resources_raw, "prompts", "resources")
    image_path = resources_raw.get("image")
    motion_path = _required(resources_raw, "motion", "resources")
    for field, value in (("prompts", prompts_path), ("image", image_path), ("motion", motion_path)):
        if value is None and field == "image":
            continue
        if not isinstance(value, str) or not value:
            fail(E_CONFIG_INVALID, f"resources.{field}", "expected a non-empty path string")

    source_dir = str(Path(source).parent)
    resource_paths = ResourcePaths(
        prompts=str((Path(source_dir) / str(prompts_path)).resolve()),
        image=None if image_path is None else str((Path(source_dir) / str(image_path)).resolve()),
        motion=str((Path(source_dir) / str(motion_path)).resolve()),
    )
    snapshot_version = to_int(
        cache.get("snapshot_version", 1),
        "cache.snapshot_version",
        minimum=1,
    )

    return Config(
        source_path=source,
        source_dir=source_dir,
        width=width,
        height=height,
        channels=channels,
        frame_count=frame_count,
        latent_height=latent_height,
        latent_width=latent_width,
        latent_channels=latent_channels,
        condition_dim=condition_dim,
        frac_bits=frac_bits,
        allowed_frac_bits=allowed_frac_bits,
        quantization_level=quantization_level,
        allowed_quantization_levels=allowed_quantization_levels,
        denoise_steps=denoise_steps,
        seed=seed,
        noise_seed=noise_seed,
        guidance=guidance,
        batch_point_budget=batch_point_budget,
        block_rows=block_rows,
        max_concurrency=max_concurrency,
        interaction_latency_budget_ms=interaction_latency_budget_ms,
        image_block_size=image_block_size,
        motion_block_width=motion_block_width,
        motion_block_height=motion_block_height,
        fusion_weights=tuple(weights),
        vocabulary=tuple(vocabulary),
        resources=resource_paths,
        snapshot_version=snapshot_version,
    )
