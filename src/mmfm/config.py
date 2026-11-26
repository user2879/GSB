"""Render configuration and controlled vocabulary loading/validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import content_digest
from .errors import (
    E_CHUNK_ROWS,
    E_CONFIG_INVALID,
    E_CONFIG_MISSING,
    E_DENOISE_STEPS,
    E_DIMENSION,
    E_ENUM_INVALID,
    E_ENUM_MISSING,
    E_EXCHANGE_VERSION,
    E_FIXED_SCALE,
    E_FRAME_COUNT,
    E_QUANT_PROFILE,
    E_UNKNOWN_TERM,
    BusinessError,
    fail,
)
from .fixed import QuantProfile, make_profile


def load_structured(path: str | Path) -> Any:
    """Load JSON or YAML based on suffix (YAML needs PyYAML installed)."""

    source = Path(path)
    if not source.exists():
        fail(E_CONFIG_MISSING, "configuration file does not exist", path=str(source))
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        fail(E_CONFIG_INVALID, "configuration file could not be read", path=str(source), error=str(exc))
    suffix = source.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            fail(E_CONFIG_INVALID, "YAML configuration requires PyYAML", path=str(source))
        try:
            return yaml.safe_load(text)
        except Exception as exc:  # pragma: no cover - exact YAML errors are library-defined
            fail(E_CONFIG_INVALID, "invalid YAML configuration", path=str(source), error=str(exc))
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        fail(
            E_CONFIG_INVALID,
            "invalid JSON configuration",
            path=str(source),
            line=exc.lineno,
            column=exc.colno,
            error=exc.msg,
        )


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(E_CONFIG_INVALID, f"{name} must be an object", field=name)
    return value


def require_positive_int(raw: dict[str, Any], name: str, code: str) -> int:
    if name not in raw:
        fail(E_ENUM_MISSING, f"required positive integer field {name} is missing", field=name)
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        fail(code, f"{name} must be a positive integer", field=name, value=value)
    return value


def require_nonnegative_int(raw: dict[str, Any], name: str, default: int | None = None) -> int:
    if name not in raw:
        if default is None:
            fail(E_ENUM_MISSING, f"required integer field {name} is missing", field=name)
        return default
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(E_CONFIG_INVALID, f"{name} must be a non-negative integer", field=name, value=value)
    return value


@dataclass(frozen=True)
class ImageConfig:
    width: int
    height: int
    mode: str
    tile_size: int


@dataclass(frozen=True)
class MotionConfig:
    block_size: int

    def grid_shape(self, image: ImageConfig) -> tuple[int, int]:
        rows = (image.height + self.block_size - 1) // self.block_size
        cols = (image.width + self.block_size - 1) // self.block_size
        return rows, cols


@dataclass(frozen=True)
class RenderConfig:
    raw: dict[str, Any]
    source_path: str
    seed: int
    frame_count: int
    condition_dim: int
    latent_dim: int
    image: ImageConfig
    motion: MotionConfig
    denoise_steps: int
    chunk_rows: int
    fusion_weights_permille: dict[str, int]
    quant_profile: QuantProfile
    quant_profiles: dict[str, QuantProfile]
    allowed_quant_profiles: tuple[str, ...]
    allowed_fraction_bits: tuple[int, ...]
    fixed_milli_scale: int
    noise_scheme: str
    noise_schemes: dict[str, int]
    guidance_permille_default: int
    projection_version: str
    projection_coefficient_radius: int
    projection_denominator_scale: int
    exchange_version: str
    max_workers: int
    max_denoise_points: int | None
    vocab_path: Path

    @property
    def config_hash(self) -> str:
        return content_digest(self.raw)

    @property
    def latent_config_fingerprint(self) -> str:
        """Only values that actually define initial latents/decoding."""

        return content_digest(
            {
                "seed": self.seed,
                "latent_dim": self.latent_dim,
                "condition_dim": self.condition_dim,
                "quant_profile": self.quant_profile.name,
                "projection_version": self.projection_version,
                "projection_radius": self.projection_coefficient_radius,
                "projection_denominator": self.projection_denominator_scale,
            }
        )

    def with_overrides(
        self,
        *,
        denoise_steps: int | None = None,
        chunk_rows: int | None = None,
        quant_profile: str | None = None,
        exchange_version: str | None = None,
    ) -> "RenderConfig":
        raw = json.loads(json.dumps(self.raw))
        if denoise_steps is not None:
            raw["denoise_steps"] = denoise_steps
        if chunk_rows is not None:
            raw["chunk_rows"] = chunk_rows
        if quant_profile is not None:
            raw["quant_profile"] = quant_profile
        if exchange_version is not None:
            raw["exchange_version"] = exchange_version
        return validate_config(raw, source_path=self.source_path)


def _resolve_quant(
    raw: dict[str, Any],
) -> tuple[QuantProfile, dict[str, QuantProfile], tuple[str, ...], tuple[int, ...]]:
    profiles_raw = raw.get("quant_profiles")
    if not profiles_raw:
        fail(E_ENUM_MISSING, "quant_profiles enumeration is missing", field="quant_profiles")
    if not isinstance(profiles_raw, dict):
        fail(E_QUANT_PROFILE, "quant_profiles must be an object", field="quant_profiles")
    selected_name = raw.get("quant_profile")
    if not selected_name:
        fail(E_ENUM_MISSING, "quant_profile selection is missing", field="quant_profile")
    allowed = raw.get("allowed_quant_profiles")
    if not isinstance(allowed, list) or not allowed:
        fail(E_ENUM_MISSING, "allowed_quant_profiles enumeration is missing", field="allowed_quant_profiles")
    allowed_names = tuple(str(item) for item in allowed)
    if selected_name not in allowed_names:
        fail(
            E_ENUM_INVALID,
            "selected quant_profile is not allowed",
            selected=selected_name,
            allowed=list(allowed_names),
        )
    allowed_bits = raw.get("allowed_fraction_bits")
    if not isinstance(allowed_bits, list) or not allowed_bits:
        fail(
            E_ENUM_MISSING,
            "allowed_fraction_bits enumeration is missing",
            field="allowed_fraction_bits",
        )
    bits_tuple = tuple(int(item) for item in allowed_bits if isinstance(item, int) and not isinstance(item, bool))
    if len(bits_tuple) != len(allowed_bits):
        fail(E_ENUM_INVALID, "allowed_fraction_bits must contain integers", field="allowed_fraction_bits")

    profiles: dict[str, QuantProfile] = {}
    for name in allowed_names:
        item = profiles_raw.get(name)
        if not isinstance(item, dict):
            fail(E_QUANT_PROFILE, "allowed quant profile has no definition", profile=name)
        bits = item.get("fraction_bits")
        if isinstance(bits, bool) or not isinstance(bits, int):
            fail(E_ENUM_MISSING, "quant profile fraction_bits is missing", profile=name)
        if bits not in bits_tuple:
            fail(
                E_ENUM_INVALID,
                "quant profile fraction_bits is outside allowed_fraction_bits",
                profile=name,
                fraction_bits=bits,
                allowed=list(bits_tuple),
            )
        profile = make_profile(name, bits)
        if "scale" in item and item["scale"] != profile.scale:
            fail(
                E_QUANT_PROFILE,
                "quant profile scale does not equal 2**fraction_bits",
                profile=name,
                scale=item["scale"],
                expected=profile.scale,
            )
        profiles[name] = profile
    return profiles[str(selected_name)], profiles, allowed_names, bits_tuple


def validate_config(raw: Any, *, source_path: str = "<memory>") -> RenderConfig:
    data = require_mapping(raw, "render configuration")
    if data.get("schema") not in {None, "mmfm-render-config-v1"}:
        fail(E_CONFIG_INVALID, "unsupported render configuration schema", schema=data.get("schema"))

    seed = data.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        fail(E_CONFIG_INVALID, "seed must be a non-negative integer", field="seed", value=seed)
    frame_count = require_positive_int(data, "frame_count", E_FRAME_COUNT)
    condition_dim = require_positive_int(data, "condition_dim", E_DIMENSION)
    latent_dim = require_positive_int(data, "latent_dim", E_DIMENSION)
    denoise_steps = require_positive_int(data, "denoise_steps", E_DENOISE_STEPS)
    chunk_rows = require_positive_int(data, "chunk_rows", E_CHUNK_ROWS)

    image_raw = require_mapping(data.get("image"), "image")
    mode = image_raw.get("mode")
    if mode is None:
        fail(E_ENUM_MISSING, "image mode is missing", field="image.mode")
    if mode not in {"RGB", "RGBA"}:
        fail(E_ENUM_INVALID, "image mode must be RGB or RGBA", mode=mode)
    image = ImageConfig(
        width=require_positive_int(image_raw, "width", E_DIMENSION),
        height=require_positive_int(image_raw, "height", E_DIMENSION),
        mode=str(mode),
        tile_size=require_positive_int(image_raw, "tile_size", E_DIMENSION),
    )
    if image.tile_size > max(image.width, image.height):
        fail(E_DIMENSION, "image tile_size is larger than image dimensions", tile_size=image.tile_size)
    if chunk_rows > image.height:
        fail(
            E_CHUNK_ROWS,
            "chunk_rows cannot exceed image height because every row band must be non-empty",
            chunk_rows=chunk_rows,
            image_height=image.height,
        )

    motion_raw = require_mapping(data.get("motion"), "motion")
    motion = MotionConfig(block_size=require_positive_int(motion_raw, "block_size", E_DIMENSION))
    grid_rows, grid_cols = motion.grid_shape(image)
    if grid_rows * grid_cols <= 0:
        fail(E_DIMENSION, "motion grid has no blocks", rows=grid_rows, columns=grid_cols)

    weights = require_mapping(data.get("fusion_weights_permille"), "fusion_weights_permille")
    expected_types = {"image", "motion", "text"}
    if set(weights) != expected_types:
        fail(
            E_ENUM_MISSING,
            "fusion weight map must contain exactly image, motion, text",
            present=sorted(weights),
        )
    parsed_weights: dict[str, int] = {}
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            fail(E_CONFIG_INVALID, "fusion weight must be a non-negative integer permille", type=name)
        parsed_weights[name] = value

    quant_profile, quant_profiles, allowed_profiles, allowed_bits = _resolve_quant(data)

    fixed = require_mapping(data.get("fixed_point"), "fixed_point")
    allowed_scales = fixed.get("allowed_milli_scales")
    if not isinstance(allowed_scales, list) or not allowed_scales:
        fail(E_ENUM_MISSING, "allowed_milli_scales enumeration is missing")
    milli_scale = fixed.get("milli_scale")
    if milli_scale is None:
        fail(E_ENUM_MISSING, "fixed_point.milli_scale is missing")
    if milli_scale not in allowed_scales:
        fail(
            E_FIXED_SCALE,
            "fixed_point.milli_scale is outside allowed enumeration",
            value=milli_scale,
            allowed=allowed_scales,
        )

    noise_schemes_raw = data.get("noise_schemes")
    if not isinstance(noise_schemes_raw, dict) or not noise_schemes_raw:
        fail(E_ENUM_MISSING, "noise_schemes enumeration is missing")
    noise_schemes: dict[str, int] = {}
    for name, code in noise_schemes_raw.items():
        if isinstance(code, bool) or not isinstance(code, int):
            fail(E_ENUM_INVALID, "noise scheme code must be an integer", scheme=name)
        noise_schemes[str(name)] = code
    if len(set(noise_schemes.values())) != len(noise_schemes):
        fail(E_ENUM_INVALID, "noise scheme integer codes must be unique", codes=noise_schemes)
    noise_scheme = data.get("noise_scheme")
    if noise_scheme is None:
        fail(E_ENUM_MISSING, "noise_scheme selection is missing")
    if noise_scheme not in noise_schemes:
        fail(
            E_ENUM_INVALID,
            "selected noise_scheme is not enumerated",
            selected=noise_scheme,
            allowed=sorted(noise_schemes),
        )

    exchange_version = data.get("exchange_version")
    if exchange_version is None:
        fail(E_EXCHANGE_VERSION, "exchange_version is missing")
    if exchange_version not in {"v1", "v2"}:
        fail(E_EXCHANGE_VERSION, "exchange_version must be v1 or v2", value=exchange_version)

    projection = require_mapping(data.get("projection", {}), "projection")
    projection_version = projection.get("version", "linear-v1")
    if not isinstance(projection_version, str) or not projection_version:
        fail(E_ENUM_MISSING, "projection.version is missing")
    coefficient_radius = projection.get("coefficient_radius", 32)
    denominator_scale = projection.get("denominator_scale", 32)
    if isinstance(coefficient_radius, bool) or not isinstance(coefficient_radius, int) or coefficient_radius <= 0:
        fail(E_CONFIG_INVALID, "projection.coefficient_radius must be a positive integer")
    if isinstance(denominator_scale, bool) or not isinstance(denominator_scale, int) or denominator_scale <= 0:
        fail(E_CONFIG_INVALID, "projection.denominator_scale must be a positive integer")

    guidance = data.get("guidance_permille", 1000)
    if isinstance(guidance, bool) or not isinstance(guidance, int) or guidance < 0:
        fail(E_CONFIG_INVALID, "guidance_permille must be a non-negative integer", value=guidance)

    vocab_value = data.get("vocab")
    if not isinstance(vocab_value, str) or not vocab_value:
        fail(E_ENUM_MISSING, "vocab path is missing from render configuration")
    vocab_path = Path(vocab_value)
    if not vocab_path.is_absolute() and source_path != "<memory>":
        vocab_path = Path(source_path).parent / vocab_path

    max_workers = require_positive_int(data, "max_workers", E_CONFIG_INVALID)
    max_budget = data.get("max_denoise_points")
    if max_budget is not None:
        if isinstance(max_budget, bool) or not isinstance(max_budget, int) or max_budget < 0:
            fail(E_CONFIG_INVALID, "max_denoise_points must be non-negative", value=max_budget)

    return RenderConfig(
        raw=data,
        source_path=source_path,
        seed=seed,
        frame_count=frame_count,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        image=image,
        motion=motion,
        denoise_steps=denoise_steps,
        chunk_rows=chunk_rows,
        fusion_weights_permille=parsed_weights,
        quant_profile=quant_profile,
        quant_profiles=quant_profiles,
        allowed_quant_profiles=allowed_profiles,
        allowed_fraction_bits=allowed_bits,
        fixed_milli_scale=int(milli_scale),
        noise_scheme=str(noise_scheme),
        noise_schemes=noise_schemes,
        guidance_permille_default=guidance,
        projection_version=str(projection_version),
        projection_coefficient_radius=int(coefficient_radius),
        projection_denominator_scale=int(denominator_scale),
        exchange_version=str(exchange_version),
        max_workers=max_workers,
        max_denoise_points=max_budget,
        vocab_path=vocab_path,
    )


def load_render_config(path: str | Path) -> RenderConfig:
    return validate_config(load_structured(path), source_path=str(Path(path)))


@dataclass(frozen=True)
class VocabTerm:
    term: str
    kind: str
    index: int
    weight: int


@dataclass(frozen=True)
class Vocabulary:
    version: str
    terms: tuple[VocabTerm, ...]
    digest: str

    @property
    def by_name(self) -> dict[str, VocabTerm]:
        return {term.term: term for term in self.terms}


def load_vocabulary(path: str | Path, condition_dim: int) -> Vocabulary:
    raw = load_structured(path)
    data = require_mapping(raw, "vocabulary")
    if data.get("schema") not in {None, "mmfm-vocab-v1"}:
        fail(E_CONFIG_INVALID, "unsupported vocabulary schema", schema=data.get("schema"))
    terms_raw = data.get("terms")
    if not isinstance(terms_raw, list) or not terms_raw:
        fail(E_CONFIG_INVALID, "vocabulary terms must be a non-empty list")
    terms: list[VocabTerm] = []
    seen: set[str] = set()
    for item in terms_raw:
        if not isinstance(item, dict):
            fail(E_CONFIG_INVALID, "vocabulary term must be an object")
        term = item.get("term")
        kind = item.get("kind")
        index = item.get("index")
        weight = item.get("weight", 1)
        if not isinstance(term, str) or not term:
            fail(E_CONFIG_INVALID, "vocabulary term name is missing")
        if term in seen:
            fail(E_CONFIG_INVALID, "duplicate vocabulary term", term=term)
        seen.add(term)
        if kind not in {"entity", "attribute"}:
            fail(E_ENUM_INVALID, "vocabulary kind must be entity or attribute", term=term, kind=kind)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < condition_dim:
            fail(
                E_DIMENSION,
                "vocabulary term index is outside the configured condition dimension",
                term=term,
                index=index,
                condition_dim=condition_dim,
            )
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            fail(E_CONFIG_INVALID, "vocabulary term weight must be a positive integer", term=term)
        terms.append(VocabTerm(term=term, kind=kind, index=index, weight=weight))
    return Vocabulary(
        version=str(data.get("version", "1")),
        terms=tuple(terms),
        digest=content_digest(raw),
    )


def validate_prompt(prompt: Any, vocabulary: Vocabulary) -> dict[str, list[str]]:
    data = require_mapping(prompt, "prompt")
    entities = data.get("entities", [])
    attributes = data.get("attributes", [])
    if not isinstance(entities, list) or not isinstance(attributes, list):
        fail(E_CONFIG_INVALID, "prompt entities and attributes must be lists")
    if not entities and not attributes:
        fail(E_CONFIG_INVALID, "prompt must contain at least one entity or attribute")
    by_name = vocabulary.by_name
    unknown: list[str] = []
    wrong_kind: list[dict[str, str]] = []
    for term in entities:
        if not isinstance(term, str):
            fail(E_CONFIG_INVALID, "prompt terms must be strings", value=term)
        if term not in by_name:
            unknown.append(term)
        elif by_name[term].kind != "entity":
            wrong_kind.append({"term": term, "expected": "entity", "actual": by_name[term].kind})
    for term in attributes:
        if not isinstance(term, str):
            fail(E_CONFIG_INVALID, "prompt terms must be strings", value=term)
        if term not in by_name:
            unknown.append(term)
        elif by_name[term].kind != "attribute":
            wrong_kind.append({"term": term, "expected": "attribute", "actual": by_name[term].kind})
    if unknown:
        fail(
            E_UNKNOWN_TERM,
            "prompt contains terms outside the controlled vocabulary",
            terms=sorted(set(unknown)),
        )
    if wrong_kind:
        fail(E_UNKNOWN_TERM, "prompt term used with the wrong vocabulary kind", terms=wrong_kind)
    return {"entities": list(entities), "attributes": list(attributes)}
