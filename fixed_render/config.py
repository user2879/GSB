from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import InputInvalid
from .fixedpoint import ALLOWED_FRAC_BITS, ALLOWED_QUANT_MODES
from .hashing import fingerprint_json
from .miniyaml import load_yaml


@dataclass(frozen=True)
class ConditionsConfig:
    vocab_path: str = "samples/vocab/controlled_vocab.txt"
    prompt_tokens: tuple[str, ...] = ("entity_cube", "attr_red", "attr_bright")
    image_path: str | None = "samples/images/condition_a.png"
    motion_path: str = "samples/motion/vectors.csv"


@dataclass(frozen=True)
class RenderConfig:
    seed: int = 20261126
    frame_width: int = 64
    frame_height: int = 64
    num_frames: int = 2
    latent_scale: int = 4
    latent_channels: int = 4
    text_dim: int = 24
    image_width: int = 64
    image_height: int = 64
    image_grid: int = 4
    image_dim: int = 80
    motion_block_h: int = 8
    motion_block_w: int = 8
    motion_dim: int = 128
    fusion_dim: int = 32
    denoise_steps: int = 4
    tile_height: int = 8
    guidance_q8: int = 1536
    noise_amplitude: int = 96
    fusion_weights: dict[str, int] = field(
        default_factory=lambda: {"text": 3, "image": 2, "motion": 1}
    )
    fixed_frac_bits: int = 8
    quant_mode: str = "round_half_up"
    preview_budget_ms: int = 250
    lease_ttl_ms: int = 5000
    executor_host: str = "127.0.0.1"
    executor_port: int = 0
    max_slots: int = 2
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)
    base_dir: Path = Path(".")

    @classmethod
    def load(cls, path: str | Path) -> "RenderConfig":
        path = Path(path).resolve()
        try:
            data = load_yaml(path)
        except Exception as exc:  # YAML/IO failure is a whole-file rejection.
            raise InputInvalid(
                f"cannot parse YAML config: {exc}",
                field_name="config_file",
                details={"path": str(path)},
            ) from exc
        if not isinstance(data, dict):
            raise InputInvalid("config root must be a mapping", field_name="config_file")
        return cls.from_mapping(data, base_dir=path.parent)

    @classmethod
    def from_mapping(
        cls, data: dict[str, Any], base_dir: str | Path = "."
    ) -> "RenderConfig":
        errors: dict[str, str] = {}
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"base_dir"}
        unknown = sorted(set(data) - known)
        if unknown:
            errors["unknown_fields"] = f"unknown config fields: {','.join(unknown)}"

        conditions_data = data.get("conditions", {}) or {}
        if not isinstance(conditions_data, dict):
            errors["conditions"] = "must be a mapping"
            conditions_data = {}

        def get(name: str, default: Any) -> Any:
            return data[name] if name in data and data[name] is not None else default

        def integer(name: str, default: int, minimum: int = 1, maximum: int = 1_000_000) -> int:
            value = get(name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                errors[name] = f"must be an integer in [{minimum},{maximum}]"
                return default
            if not minimum <= value <= maximum:
                errors[name] = f"must be in [{minimum},{maximum}], got {value}"
            return value

        seed = integer("seed", 20261126, 0, 2**31 - 1)
        width = integer("frame_width", 64, 8, 1024)
        height = integer("frame_height", 64, 8, 1024)
        frames = integer("num_frames", 2, 1, 64)
        latent_scale = integer("latent_scale", 4, 1, 16)
        latent_channels = integer("latent_channels", 4, 1, 32)
        text_dim = integer("text_dim", 24, 1, 4096)
        image_width = integer("image_width", width, 1, 4096)
        image_height = integer("image_height", height, 1, 4096)
        image_grid = integer("image_grid", 4, 1, 64)
        image_dim = integer("image_dim", 80, 1, 16384)
        motion_bh = integer("motion_block_h", 8, 1, 512)
        motion_bw = integer("motion_block_w", 8, 1, 512)
        motion_dim = integer("motion_dim", 128, 1, 65536)
        fusion_dim = integer("fusion_dim", 32, 1, 4096)
        steps = integer("denoise_steps", 4, 1, 64)
        tile_height = integer("tile_height", 8, 1, 1024)
        guidance = integer("guidance_q8", 1536, 0, 2**31 - 1)
        noise_amp = integer("noise_amplitude", 96, 0, 2**31 - 1)
        preview_ms = integer("preview_budget_ms", 250, 0, 600_000)
        lease_ttl = integer("lease_ttl_ms", 5000, 20, 3_600_000)
        port = integer("executor_port", 0, 0, 65535)
        max_slots = integer("max_slots", 2, 1, 64)

        frac_bits = get("fixed_frac_bits", 8)
        if frac_bits not in ALLOWED_FRAC_BITS:
            errors["fixed_frac_bits"] = f"must be one of {ALLOWED_FRAC_BITS}"
            frac_bits = 8
        quant_mode = get("quant_mode", "round_half_up")
        if quant_mode not in ALLOWED_QUANT_MODES:
            errors["quant_mode"] = f"must be one of {ALLOWED_QUANT_MODES}"
            quant_mode = "round_half_up"

        host = get("executor_host", "127.0.0.1")
        if host not in {"127.0.0.1", "localhost"}:
            errors["executor_host"] = "loopback executor host must be 127.0.0.1"

        weights = get("fusion_weights", {"text": 3, "image": 2, "motion": 1})
        if not isinstance(weights, dict) or set(weights) != {"text", "image", "motion"}:
            errors["fusion_weights"] = "must contain exactly text, image and motion"
            weights = {"text": 3, "image": 2, "motion": 1}
        else:
            for key, value in weights.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 10000:
                    errors[f"fusion_weights.{key}"] = "must be a non-negative integer <= 10000"

        if height % tile_height != 0:
            errors["tile_height"] = f"tile_height {tile_height} must divide frame_height {height}"
        if tile_height % latent_scale != 0:
            errors["tile_height"] = (
                f"tile_height {tile_height} must be a multiple of latent_scale {latent_scale}"
            )
        if width % latent_scale or height % latent_scale:
            errors["latent_scale"] = "frame_width and frame_height must be divisible by latent_scale"
        expected_image_dim = image_grid * image_grid * 5
        if image_dim != expected_image_dim:
            errors["image_dim"] = (
                f"image_dim must be image_grid^2*5={expected_image_dim}, got {image_dim}"
            )
        if image_width <= 0 or image_height <= 0:
            errors["image_resolution"] = "image width/height must be positive"
        block_count = (height // motion_bh) * (width // motion_bw)
        if height % motion_bh or width % motion_bw:
            errors["motion_block"] = "frame dimensions must be divisible by motion block dimensions"
        expected_motion_dim = 2 * block_count
        if motion_dim != expected_motion_dim:
            errors["motion_dim"] = (
                f"motion_dim must be 2*block_count={expected_motion_dim}, got {motion_dim}"
            )

        try:
            conditions = ConditionsConfig(
                vocab_path=str(conditions_data.get("vocab_path", ConditionsConfig.vocab_path)),
                prompt_tokens=_parse_tokens(conditions_data.get("prompt_tokens", ["entity_cube"])),
                image_path=conditions_data.get("image_path", ConditionsConfig.image_path),
                motion_path=str(conditions_data.get("motion_path", ConditionsConfig.motion_path)),
            )
        except InputInvalid as exc:
            errors.update(exc.fields or {"conditions.prompt_tokens": exc.message})
            conditions = ConditionsConfig()

        if errors:
            raise InputInvalid(
                "configuration validation rejected the whole file",
                fields=errors,
                details={"path": str(base_dir)},
            )

        return cls(
            seed=seed,
            frame_width=width,
            frame_height=height,
            num_frames=frames,
            latent_scale=latent_scale,
            latent_channels=latent_channels,
            text_dim=text_dim,
            image_width=image_width,
            image_height=image_height,
            image_grid=image_grid,
            image_dim=image_dim,
            motion_block_h=motion_bh,
            motion_block_w=motion_bw,
            motion_dim=motion_dim,
            fusion_dim=fusion_dim,
            denoise_steps=steps,
            tile_height=tile_height,
            guidance_q8=guidance,
            noise_amplitude=noise_amp,
            fusion_weights=dict(weights),
            fixed_frac_bits=int(frac_bits),
            quant_mode=str(quant_mode),
            preview_budget_ms=preview_ms,
            lease_ttl_ms=lease_ttl,
            executor_host=str(host),
            executor_port=port,
            max_slots=max_slots,
            conditions=conditions,
            base_dir=Path(base_dir),
        )

    def resolve(self, maybe_path: str | Path | None) -> Path | None:
        if maybe_path is None:
            return None
        path = Path(maybe_path)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def with_changes(self, changes: dict[str, Any]) -> "RenderConfig":
        data = self.to_mapping()
        conditions = dict(data.get("conditions", {}))
        for key, value in changes.items():
            if key in {"vocab_path", "prompt_tokens", "image_path", "motion_path"}:
                conditions[key] = value
            elif key == "conditions":
                conditions.update(value or {})
            else:
                data[key] = value
        data["conditions"] = conditions
        return RenderConfig.from_mapping(data, base_dir=self.base_dir)

    def to_mapping(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("base_dir", None)
        data["conditions"] = asdict(self.conditions)
        return data

    def parameter_fingerprint(self) -> str:
        payload = self.to_mapping()
        # Input contents are represented separately by condition fingerprints.
        payload["conditions"].pop("vocab_path", None)
        payload["conditions"].pop("image_path", None)
        payload["conditions"].pop("motion_path", None)
        payload["conditions"].pop("prompt_tokens", None)
        return fingerprint_json(payload)

    def config_fingerprint(self) -> str:
        return fingerprint_json(self.to_mapping())


def _parse_tokens(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = tuple(value.split())
    elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        tokens = tuple(value)
    else:
        raise InputInvalid(
            "prompt_tokens must be a string or list of strings",
            field_name="conditions.prompt_tokens",
        )
    if not tokens:
        raise InputInvalid("prompt must not be empty", field_name="conditions.prompt_tokens")
    if any(not token or any(ch.isspace() for ch in token) for token in tokens):
        raise InputInvalid(
            "prompt tokens must be non-empty vocabulary items",
            field_name="conditions.prompt_tokens",
        )
    return tokens
