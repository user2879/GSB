"""Strict local YAML loading and validation for the render workbench."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import (
    E_CONFIG_INVALID,
    E_IMG2IMG_STEP_INVALID,
    E_NOISE_TABLE_MISMATCH,
    Failure,
)
from .util import WORKBENCH_FORMAT_VERSION, canonical_json, sha256_hex, value_fingerprint


BUILTIN_STRATEGY_VERSION = "strategy-table-v1"


@dataclass
class ProjectConfig:
    name: str = "deterministic-render-diagnostics"


@dataclass
class PromptFileConfig:
    samples_path: str
    embedding_dim: int


@dataclass
class ResolutionTier:
    name: str
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "width": self.width, "height": self.height}


@dataclass
class TableSpec:
    path: str | None
    values: tuple[float, ...] | None
    expected_length: int
    source: str


@dataclass
class RenderConfig:
    denoise_steps: int
    guidance_scale: float
    resolution_tiers: list[ResolutionTier]
    batch_sizes: list[int]
    latent_block_size: int
    cache_budget_blocks: int
    seed_max: int
    noise_table: TableSpec
    guidance_table: TableSpec


@dataclass
class Img2ImgConfig:
    enabled: bool
    replace_early_steps: int
    initial_image_path: str | None


@dataclass
class RuntimeConfig:
    temp_dir: str
    concurrency: int
    preview_budget_ms: int
    snapshot_interval: int


@dataclass
class WorkbenchConfig:
    schema_version: int
    source_path: str
    project: ProjectConfig
    prompt: PromptFileConfig
    render: RenderConfig
    img2img: Img2ImgConfig
    runtime: RuntimeConfig
    strategy_version: str

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "format_version": WORKBENCH_FORMAT_VERSION,
            "schema_version": self.schema_version,
            "project": {"name": self.project.name},
            "prompt": {"embedding_dim": self.prompt.embedding_dim},
            "render": {
                "denoise_steps": self.render.denoise_steps,
                "guidance_scale": self.render.guidance_scale,
                "resolution_tiers": [tier.to_dict() for tier in self.render.resolution_tiers],
                "batch_sizes": list(self.render.batch_sizes),
                "latent_block_size": self.render.latent_block_size,
                "cache_budget_blocks": self.render.cache_budget_blocks,
                "seed_max": self.render.seed_max,
            },
            "img2img": {
                "enabled": self.img2img.enabled,
                "replace_early_steps": self.img2img.replace_early_steps,
                "initial_image_sha256": (
                    sha256_hex(Path(self.img2img.initial_image_path).read_bytes())
                    if self.img2img.enabled and self.img2img.initial_image_path
                    else None
                ),
            },
            "runtime": {
                "concurrency": self.runtime.concurrency,
                "preview_budget_ms": self.runtime.preview_budget_ms,
                "snapshot_interval": self.runtime.snapshot_interval,
            },
            "strategy_version": self.strategy_version,
        }


@dataclass
class Prompt:
    prompt_id: str
    tokens: tuple[str, ...]
    index: int = 0

    def canonical_tokens(self) -> tuple[str, ...]:
        return tuple(sorted(self.tokens))

    def content_fingerprint(self, vocab: Iterable[str], embedding_dim: int) -> str:
        return value_fingerprint(
            "prompt-content-v1",
            {
                "id": self.prompt_id,
                "tokens": list(self.canonical_tokens()),
                "vocab": list(vocab),
                "embedding_dim": embedding_dim,
            },
        )


@dataclass
class PromptSamples:
    schema_version: int
    source_path: str
    vocab: tuple[str, ...]
    embedding_dim: int
    seeds: list[int]
    prompts: list[Prompt]

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "vocab": list(self.vocab),
            "embedding_dim": self.embedding_dim,
            "seeds": list(self.seeds),
            "prompts": [
                {"id": prompt.prompt_id, "tokens": list(prompt.tokens)}
                for prompt in self.prompts
            ],
        }


@dataclass
class LoadResult:
    config: WorkbenchConfig | None
    samples: PromptSamples | None
    noise_table: tuple[float, ...]
    guidance_table: tuple[float, ...]
    failures: list[Failure]
    fatal: bool
    valid_seeds: list[int] = field(default_factory=list)
    valid_prompts: list[Prompt] = field(default_factory=list)

    @property
    def can_render(self) -> bool:
        return (
            not self.fatal
            and self.config is not None
            and self.samples is not None
            and bool(self.valid_seeds)
            and bool(self.valid_prompts)
            and len(self.noise_table) == self.config.render.denoise_steps
            and len(self.guidance_table) == self.config.render.denoise_steps
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok" if self.can_render else "failed",
            "business_code": None if self.can_render else (self.failures[0].code if self.failures else E_CONFIG_INVALID),
            "failures": [failure.to_dict() for failure in self.failures],
            "fatal": self.fatal,
            "valid_seeds": list(self.valid_seeds),
            "valid_prompts": [
                {"id": prompt.prompt_id, "tokens": list(prompt.tokens), "index": prompt.index}
                for prompt in self.valid_prompts
            ],
            "noise_table_length": len(self.noise_table),
            "guidance_table_length": len(self.guidance_table),
        }


def _failure(
    failures: list[Failure],
    code: str,
    component: str,
    item: str,
    message: str,
    **details: Any,
) -> None:
    failures.append(
        Failure(
            code=code,
            component=component,
            item=item,
            message=message,
            details={key: value for key, value in details.items()},
        )
    )


def _unknown_keys(value: dict[str, Any], allowed: set[str]) -> list[str]:
    return sorted(key for key in value if key not in allowed)


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any) -> bool:
    return _is_int(value) and value > 0


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _parse_table(
    raw: Any,
    *,
    name: str,
    base_dir: Path,
    denoise_steps: int,
    failures: list[Failure],
    fatal_flags: list[bool],
) -> tuple[TableSpec | None, tuple[float, ...]]:
    if not isinstance(raw, dict):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            f"render.{name}",
            "table specification must be a mapping",
        )
        fatal_flags.append(True)
        return None, ()
    unknown = _unknown_keys(raw, {"path", "values", "expected_length"})
    if unknown:
        for key in unknown:
            _failure(
                failures,
                E_CONFIG_INVALID,
                "config",
                f"render.{name}.{key}",
                "unknown table key",
            )
        fatal_flags.append(True)

    expected = raw.get("expected_length")
    if not _positive_int(expected):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            f"render.{name}.expected_length",
            "expected_length must be a positive integer",
        )
        fatal_flags.append(True)
        expected = denoise_steps
    if expected != denoise_steps:
        _failure(
            failures,
            E_NOISE_TABLE_MISMATCH,
            "table",
            f"render.{name}.expected_length",
            "declared table length differs from denoise_steps",
            expected_length=expected,
            denoise_steps=denoise_steps,
        )
        fatal_flags.append(True)

    values: list[Any] | None = None
    path_text: str | None = None
    source = ""
    if "values" in raw:
        if "path" in raw:
            _failure(
                failures,
                E_CONFIG_INVALID,
                "config",
                f"render.{name}",
                "table must use either path or values, not both",
            )
            fatal_flags.append(True)
        values = raw.get("values") if isinstance(raw.get("values"), list) else None
        source = "inline"
        if values is None:
            _failure(
                failures,
                E_CONFIG_INVALID,
                "config",
                f"render.{name}.values",
                "values must be a list",
            )
            fatal_flags.append(True)
    elif "path" in raw:
        if not isinstance(raw.get("path"), str) or not raw.get("path"):
            _failure(
                failures,
                E_CONFIG_INVALID,
                "config",
                f"render.{name}.path",
                "path must be a non-empty string",
            )
            fatal_flags.append(True)
        else:
            path_text = str(raw["path"])
            resolved = (base_dir / path_text).resolve()
            source = str(resolved)
            try:
                loaded = _read_yaml(resolved)
            except FileNotFoundError:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "config",
                    f"render.{name}.path",
                    "table file does not exist",
                    path=str(resolved),
                )
                fatal_flags.append(True)
                loaded = None
            except Exception as exc:  # YAML/OSError diagnostics are part of the input contract.
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "config",
                    f"render.{name}.path",
                    "table file could not be read",
                    path=str(resolved),
                    error=type(exc).__name__,
                )
                fatal_flags.append(True)
                loaded = None
            if isinstance(loaded, dict) and "values" in loaded:
                loaded = loaded["values"]
            if isinstance(loaded, list):
                values = loaded
            elif loaded is not None:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "config",
                    f"render.{name}.path",
                    "table file must contain a list or a mapping with values",
                    path=str(resolved),
                )
                fatal_flags.append(True)
    else:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            f"render.{name}",
            "table must specify path or values",
        )
        fatal_flags.append(True)

    if values is None:
        return None, ()
    numeric_values: list[float] = []
    for index, value in enumerate(values):
        if not _finite_number(value):
            _failure(
                failures,
                E_CONFIG_INVALID,
                "table",
                f"render.{name}.values[{index}]",
                "table entry must be a finite number",
            )
            fatal_flags.append(True)
        else:
            numeric_values.append(float(value))
    if len(numeric_values) != len(values):
        return None, ()
    if len(values) != expected or len(values) != denoise_steps:
        _failure(
            failures,
            E_NOISE_TABLE_MISMATCH,
            "table",
            f"render.{name}",
            "actual table length differs from declared length or denoise_steps",
            actual_length=len(values),
            expected_length=expected,
            denoise_steps=denoise_steps,
            source=source,
        )
        fatal_flags.append(True)
    spec = TableSpec(
        path=path_text,
        values=tuple(numeric_values),
        expected_length=expected,
        source=source,
    )
    return spec, tuple(numeric_values)


def _parse_tier(raw: Any, *, index: int, failures: list[Failure], fatal_flags: list[bool]) -> ResolutionTier | None:
    if not isinstance(raw, dict):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            f"render.resolution_tiers[{index}]",
            "tier must be a mapping",
        )
        fatal_flags.append(True)
        return None
    unknown = _unknown_keys(raw, {"name", "width", "height"})
    for key in unknown:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            f"render.resolution_tiers[{index}].{key}",
            "unknown tier key",
        )
        fatal_flags.append(True)
    name = raw.get("name")
    width = raw.get("width")
    height = raw.get("height")
    if not isinstance(name, str) or not name:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            f"render.resolution_tiers[{index}].name",
            "tier name must be a non-empty string",
        )
        fatal_flags.append(True)
    if not _positive_int(width) or not _positive_int(height):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            f"render.resolution_tiers[{index}]",
            "width and height must be positive integers",
        )
        fatal_flags.append(True)
    if not isinstance(name, str) or not name or not _positive_int(width) or not _positive_int(height):
        return None
    return ResolutionTier(name=name, width=int(width), height=int(height))


def _load_samples(
    path: Path,
    *,
    expected_embedding_dim: int,
    seed_max: int,
    failures: list[Failure],
    fatal_flags: list[bool],
) -> tuple[PromptSamples | None, list[int], list[Prompt]]:
    try:
        raw = _read_yaml(path)
    except FileNotFoundError:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "sample",
            "prompt.samples_path",
            "sample file does not exist",
            path=str(path),
        )
        fatal_flags.append(True)
        return None, [], []
    except Exception as exc:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "sample",
            "prompt.samples_path",
            "sample file could not be read",
            path=str(path),
            error=type(exc).__name__,
        )
        fatal_flags.append(True)
        return None, [], []

    if not isinstance(raw, dict):
        _failure(failures, E_CONFIG_INVALID, "sample", "root", "sample root must be a mapping")
        fatal_flags.append(True)
        return None, [], []

    unknown = _unknown_keys(raw, {"schema_version", "vocab", "embedding_dim", "seeds", "prompts"})
    for key in unknown:
        _failure(failures, E_CONFIG_INVALID, "sample", key, "unknown sample key")
        fatal_flags.append(True)

    if raw.get("schema_version") != 1:
        _failure(failures, E_CONFIG_INVALID, "sample", "schema_version", "schema_version must be 1")
        fatal_flags.append(True)

    sample_dim = raw.get("embedding_dim")
    if not _positive_int(sample_dim):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "sample",
            "embedding_dim",
            "embedding_dim must be a positive integer",
        )
        fatal_flags.append(True)
        sample_dim_int = 0
    else:
        sample_dim_int = int(sample_dim)
        if sample_dim_int != expected_embedding_dim:
            _failure(
                failures,
                E_CONFIG_INVALID,
                "sample",
                "embedding_dim",
                "sample embedding dimension does not match config prompt.embedding_dim",
                sample_embedding_dim=sample_dim_int,
                config_embedding_dim=expected_embedding_dim,
            )
            fatal_flags.append(True)

    vocab_raw = raw.get("vocab")
    vocab: tuple[str, ...] = ()
    if not isinstance(vocab_raw, list) or not vocab_raw:
        _failure(failures, E_CONFIG_INVALID, "sample", "vocab", "vocab must be a non-empty list")
        fatal_flags.append(True)
    elif not all(isinstance(token, str) and token for token in vocab_raw):
        _failure(failures, E_CONFIG_INVALID, "sample", "vocab", "vocab entries must be non-empty strings")
        fatal_flags.append(True)
    else:
        vocab = tuple(vocab_raw)
        if len(set(vocab)) != len(vocab):
            _failure(failures, E_CONFIG_INVALID, "sample", "vocab", "vocab entries must be unique")
            fatal_flags.append(True)

    seeds_raw = raw.get("seeds")
    valid_seeds: list[int] = []
    if not isinstance(seeds_raw, list) or not seeds_raw:
        _failure(failures, E_CONFIG_INVALID, "sample", "seeds", "seeds must be a non-empty list")
        fatal_flags.append(True)
    else:
        for index, seed in enumerate(seeds_raw):
            if not _is_int(seed) or seed < 0 or seed > seed_max:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "seed",
                    f"seeds[{index}]",
                    "seed is outside the configured integer range",
                    seed=seed,
                    seed_max=seed_max,
                )
                continue
            valid_seeds.append(int(seed))
        if not valid_seeds:
            fatal_flags.append(True)

    prompts_raw = raw.get("prompts")
    valid_prompts: list[Prompt] = []
    if not isinstance(prompts_raw, list) or not prompts_raw:
        _failure(failures, E_CONFIG_INVALID, "sample", "prompts", "prompts must be a non-empty list")
        fatal_flags.append(True)
    else:
        seen_ids: set[str] = set()
        for index, prompt_raw in enumerate(prompts_raw):
            item = f"prompts[{index}]"
            if not isinstance(prompt_raw, dict):
                _failure(failures, E_CONFIG_INVALID, "prompt", item, "prompt must be a mapping")
                continue
            prompt_unknown = _unknown_keys(prompt_raw, {"id", "tokens"})
            prompt_ok = True
            for key in prompt_unknown:
                _failure(failures, E_CONFIG_INVALID, "prompt", f"{item}.{key}", "unknown prompt key")
                prompt_ok = False
            prompt_id = prompt_raw.get("id")
            if not isinstance(prompt_id, str) or not prompt_id:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "prompt",
                    f"{item}.id",
                    "prompt id must be a non-empty string",
                )
                prompt_ok = False
            elif prompt_id in seen_ids:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "prompt",
                    f"{item}.id",
                    "prompt id must be unique",
                    prompt_id=prompt_id,
                )
                prompt_ok = False
            else:
                seen_ids.add(prompt_id)
            tokens_raw = prompt_raw.get("tokens")
            if not isinstance(tokens_raw, list) or not tokens_raw or not all(
                isinstance(token, str) and token for token in tokens_raw
            ):
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "prompt",
                    f"{item}.tokens",
                    "tokens must be a non-empty list of non-empty strings",
                )
                prompt_ok = False
            elif any(token not in vocab for token in tokens_raw):
                unknown_tokens = sorted(set(str(token) for token in tokens_raw if token not in vocab))
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "prompt",
                    f"{item}.tokens",
                    "prompt contains tokens outside the restricted vocabulary",
                    unknown_tokens=unknown_tokens,
                )
                prompt_ok = False
            if prompt_ok:
                valid_prompts.append(Prompt(prompt_id=str(prompt_id), tokens=tuple(tokens_raw)))
            else:
                continue

    if not valid_prompts:
        fatal_flags.append(True)

    for index, prompt in enumerate(valid_prompts):
        prompt.index = index

    samples = PromptSamples(
        schema_version=1 if raw.get("schema_version") == 1 else 0,
        source_path=str(path),
        vocab=vocab,
        embedding_dim=sample_dim_int,
        seeds=list(valid_seeds),
        prompts=list(valid_prompts),
    )
    return samples, valid_seeds, valid_prompts


def load_workbench(config_path: str | Path, *, temp_dir_override: str | None = None) -> LoadResult:
    """Load and validate a local config/sample pair.

    Validation is intentionally aggregate: independent bad seeds or prompts
    become failure-list entries and are excluded, while structural errors make
    the whole run non-renderable.
    """

    failures: list[Failure] = []
    fatal_flags: list[bool] = []
    config_file = Path(config_path).resolve()
    try:
        raw = _read_yaml(config_file)
    except FileNotFoundError:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "root",
            "config file does not exist",
            path=str(config_file),
        )
        fatal_flags.append(True)
        return LoadResult(None, None, (), (), failures, True)
    except Exception as exc:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "root",
            "config file could not be read",
            path=str(config_file),
            error=type(exc).__name__,
        )
        fatal_flags.append(True)
        return LoadResult(None, None, (), (), failures, True)

    if not isinstance(raw, dict):
        _failure(failures, E_CONFIG_INVALID, "config", "root", "config root must be a mapping")
        return LoadResult(None, None, (), (), failures, True)

    for key in _unknown_keys(
        raw,
        {"schema_version", "project", "prompt", "render", "img2img", "runtime", "strategy"},
    ):
        _failure(failures, E_CONFIG_INVALID, "config", key, "unknown top-level config key")
        fatal_flags.append(True)

    if raw.get("schema_version") != 1:
        _failure(failures, E_CONFIG_INVALID, "config", "schema_version", "schema_version must be 1")
        fatal_flags.append(True)

    project_raw = raw.get("project", {})
    if not isinstance(project_raw, dict):
        _failure(failures, E_CONFIG_INVALID, "config", "project", "project must be a mapping")
        fatal_flags.append(True)
        project_raw = {}
    for key in _unknown_keys(project_raw, {"name"}):
        _failure(failures, E_CONFIG_INVALID, "config", f"project.{key}", "unknown project key")
        fatal_flags.append(True)
    project_name = project_raw.get("name", "deterministic-render-diagnostics")
    if not isinstance(project_name, str) or not project_name:
        _failure(failures, E_CONFIG_INVALID, "config", "project.name", "name must be a non-empty string")
        project_name = "deterministic-render-diagnostics"

    prompt_raw = raw.get("prompt")
    if not isinstance(prompt_raw, dict):
        _failure(failures, E_CONFIG_INVALID, "config", "prompt", "prompt must be a mapping")
        fatal_flags.append(True)
        prompt_raw = {}
    for key in _unknown_keys(prompt_raw, {"samples_path", "embedding_dim"}):
        _failure(failures, E_CONFIG_INVALID, "config", f"prompt.{key}", "unknown prompt key")
        fatal_flags.append(True)
    samples_path_raw = prompt_raw.get("samples_path")
    if not isinstance(samples_path_raw, str) or not samples_path_raw:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "prompt.samples_path",
            "samples_path must be a non-empty string",
        )
        fatal_flags.append(True)
        samples_path = config_file.parent / "__missing__.yaml"
    else:
        samples_path = (config_file.parent / samples_path_raw).resolve()
    embedding_dim_raw = prompt_raw.get("embedding_dim")
    if not _positive_int(embedding_dim_raw):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "prompt.embedding_dim",
            "embedding_dim must be a positive integer",
        )
        fatal_flags.append(True)
        embedding_dim = 0
    else:
        embedding_dim = int(embedding_dim_raw)

    render_raw = raw.get("render")
    if not isinstance(render_raw, dict):
        _failure(failures, E_CONFIG_INVALID, "config", "render", "render must be a mapping")
        fatal_flags.append(True)
        render_raw = {}
    for key in _unknown_keys(
        render_raw,
        {
            "denoise_steps",
            "guidance_scale",
            "resolution_tiers",
            "batch_sizes",
            "latent_block_size",
            "cache_budget_blocks",
            "seed_max",
            "noise_table",
            "guidance_table",
        },
    ):
        _failure(failures, E_CONFIG_INVALID, "config", f"render.{key}", "unknown render key")
        fatal_flags.append(True)

    denoise_steps_raw = render_raw.get("denoise_steps")
    if not _positive_int(denoise_steps_raw):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "render.denoise_steps",
            "denoise_steps must be a positive integer",
        )
        fatal_flags.append(True)
        denoise_steps = 1
    else:
        denoise_steps = int(denoise_steps_raw)

    guidance_raw = render_raw.get("guidance_scale")
    if not _finite_number(guidance_raw) or float(guidance_raw) < 0:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "render.guidance_scale",
            "guidance_scale must be a finite non-negative number",
        )
        guidance_scale = 0.0
        fatal_flags.append(True)
    else:
        guidance_scale = float(guidance_raw)

    tiers_raw = render_raw.get("resolution_tiers")
    tiers: list[ResolutionTier] = []
    if not isinstance(tiers_raw, list) or not tiers_raw:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "render.resolution_tiers",
            "resolution_tiers must be a non-empty list",
        )
        fatal_flags.append(True)
    else:
        names: set[str] = set()
        for index, tier_raw in enumerate(tiers_raw):
            tier = _parse_tier(tier_raw, index=index, failures=failures, fatal_flags=fatal_flags)
            if tier is None:
                continue
            if tier.name in names:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "config",
                    f"render.resolution_tiers[{index}].name",
                    "tier names must be unique",
                    name=tier.name,
                )
                fatal_flags.append(True)
                continue
            names.add(tier.name)
            tiers.append(tier)
        if len(tiers) != len(tiers_raw):
            fatal_flags.append(True)
        for index in range(1, len(tiers)):
            if tiers[index].area <= tiers[index - 1].area:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "config",
                    "render.resolution_tiers",
                    "resolution tiers must have strictly increasing area",
                    previous=tiers[index - 1].to_dict(),
                    current=tiers[index].to_dict(),
                )
                fatal_flags.append(True)

    batch_raw = render_raw.get("batch_sizes")
    batch_sizes: list[int] = []
    if not isinstance(batch_raw, list) or not batch_raw:
        _failure(failures, E_CONFIG_INVALID, "config", "render.batch_sizes", "batch_sizes must be a non-empty list")
        fatal_flags.append(True)
    else:
        for index, value in enumerate(batch_raw):
            if not _positive_int(value):
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "config",
                    f"render.batch_sizes[{index}]",
                    "batch size must be a positive integer",
                )
                fatal_flags.append(True)
            elif int(value) in batch_sizes:
                _failure(
                    failures,
                    E_CONFIG_INVALID,
                    "config",
                    f"render.batch_sizes[{index}]",
                    "batch sizes must be unique",
                )
                fatal_flags.append(True)
            else:
                batch_sizes.append(int(value))
        if batch_sizes != sorted(batch_sizes):
            _failure(
                failures,
                E_CONFIG_INVALID,
                "config",
                "render.batch_sizes",
                "batch sizes must be listed in ascending order",
            )
            fatal_flags.append(True)

    block_size_raw = render_raw.get("latent_block_size")
    if not _positive_int(block_size_raw):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "render.latent_block_size",
            "latent_block_size must be a positive integer",
        )
        fatal_flags.append(True)
        block_size = 1
    else:
        block_size = int(block_size_raw)

    cache_budget_raw = render_raw.get("cache_budget_blocks")
    if not _positive_int(cache_budget_raw):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "render.cache_budget_blocks",
            "cache_budget_blocks must be a positive integer",
        )
        fatal_flags.append(True)
        cache_budget = 1
    else:
        cache_budget = int(cache_budget_raw)

    seed_max_raw = render_raw.get("seed_max")
    if not _is_int(seed_max_raw) or int(seed_max_raw) < 0:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "render.seed_max",
            "seed_max must be a non-negative integer",
        )
        fatal_flags.append(True)
        seed_max = 0
    else:
        seed_max = int(seed_max_raw)

    noise_spec, noise_values = _parse_table(
        render_raw.get("noise_table"),
        name="noise_table",
        base_dir=config_file.parent,
        denoise_steps=denoise_steps,
        failures=failures,
        fatal_flags=fatal_flags,
    )
    guidance_spec, guidance_values = _parse_table(
        render_raw.get("guidance_table"),
        name="guidance_table",
        base_dir=config_file.parent,
        denoise_steps=denoise_steps,
        failures=failures,
        fatal_flags=fatal_flags,
    )

    img_raw = raw.get("img2img", {})
    if not isinstance(img_raw, dict):
        _failure(failures, E_CONFIG_INVALID, "config", "img2img", "img2img must be a mapping")
        fatal_flags.append(True)
        img_raw = {}
    for key in _unknown_keys(img_raw, {"enabled", "replace_early_steps", "initial_image_path"}):
        _failure(failures, E_CONFIG_INVALID, "config", f"img2img.{key}", "unknown img2img key")
        fatal_flags.append(True)
    img_enabled_raw = img_raw.get("enabled", False)
    if not isinstance(img_enabled_raw, bool):
        _failure(failures, E_CONFIG_INVALID, "config", "img2img.enabled", "enabled must be boolean")
        img_enabled = False
        fatal_flags.append(True)
    else:
        img_enabled = img_enabled_raw
    img_steps_raw = img_raw.get("replace_early_steps", 0)
    if not _is_int(img_steps_raw) or int(img_steps_raw) < 0:
        _failure(
            failures,
            E_IMG2IMG_STEP_INVALID,
            "img2img",
            "img2img.replace_early_steps",
            "replace_early_steps must be a non-negative integer",
        )
        img_steps = 0
        fatal_flags.append(True)
    else:
        img_steps = int(img_steps_raw)
        if img_enabled and img_steps > denoise_steps:
            _failure(
                failures,
                E_IMG2IMG_STEP_INVALID,
                "img2img",
                "img2img.replace_early_steps",
                "replace_early_steps exceeds total denoise steps",
                replace_early_steps=img_steps,
                denoise_steps=denoise_steps,
            )
            fatal_flags.append(True)
    image_path_raw = img_raw.get("initial_image_path")
    image_path: str | None = None
    if image_path_raw is not None and (not isinstance(image_path_raw, str) or not image_path_raw):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "img2img.initial_image_path",
            "initial_image_path must be a non-empty string",
        )
        fatal_flags.append(True)
    elif isinstance(image_path_raw, str):
        image_path = str((config_file.parent / image_path_raw).resolve())
    if img_enabled:
        if image_path is None:
            _failure(
                failures,
                E_CONFIG_INVALID,
                "config",
                "img2img.initial_image_path",
                "initial_image_path is required when img2img is enabled",
            )
            fatal_flags.append(True)
        elif not Path(image_path).is_file():
            _failure(
                failures,
                E_CONFIG_INVALID,
                "config",
                "img2img.initial_image_path",
                "initial image does not exist",
                path=image_path,
            )
            fatal_flags.append(True)

    runtime_raw = raw.get("runtime", {})
    if not isinstance(runtime_raw, dict):
        _failure(failures, E_CONFIG_INVALID, "config", "runtime", "runtime must be a mapping")
        fatal_flags.append(True)
        runtime_raw = {}
    for key in _unknown_keys(runtime_raw, {"temp_dir", "concurrency", "preview_budget_ms", "snapshot_interval"}):
        _failure(failures, E_CONFIG_INVALID, "config", f"runtime.{key}", "unknown runtime key")
        fatal_flags.append(True)
    temp_dir_raw = runtime_raw.get("temp_dir", ".renderlab")
    if not isinstance(temp_dir_raw, str) or not temp_dir_raw:
        _failure(failures, E_CONFIG_INVALID, "config", "runtime.temp_dir", "temp_dir must be a non-empty string")
        fatal_flags.append(True)
        temp_dir_raw = ".renderlab"
    temp_dir = str((config_file.parent / temp_dir_raw).resolve())
    if temp_dir_override:
        temp_dir = str(Path(temp_dir_override).resolve())
    concurrency_raw = runtime_raw.get("concurrency", 4)
    if not _positive_int(concurrency_raw):
        _failure(failures, E_CONFIG_INVALID, "config", "runtime.concurrency", "concurrency must be a positive integer")
        fatal_flags.append(True)
        concurrency = 1
    else:
        concurrency = int(concurrency_raw)
    preview_raw = runtime_raw.get("preview_budget_ms", 50)
    if not _is_int(preview_raw) or int(preview_raw) < 0:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "runtime.preview_budget_ms",
            "preview_budget_ms must be a non-negative integer",
        )
        fatal_flags.append(True)
        preview_budget = 0
    else:
        preview_budget = int(preview_raw)
    snapshot_raw = runtime_raw.get("snapshot_interval", 1)
    if not _positive_int(snapshot_raw):
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "runtime.snapshot_interval",
            "snapshot_interval must be a positive integer",
        )
        fatal_flags.append(True)
        snapshot_interval = 1
    else:
        snapshot_interval = int(snapshot_raw)

    strategy_raw = raw.get("strategy", {})
    if not isinstance(strategy_raw, dict):
        _failure(failures, E_CONFIG_INVALID, "config", "strategy", "strategy must be a mapping")
        fatal_flags.append(True)
        strategy_raw = {}
    for key in _unknown_keys(strategy_raw, {"version"}):
        _failure(failures, E_CONFIG_INVALID, "config", f"strategy.{key}", "unknown strategy key")
        fatal_flags.append(True)
    strategy_version = strategy_raw.get("version", BUILTIN_STRATEGY_VERSION)
    if not isinstance(strategy_version, str) or not strategy_version:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "strategy.version",
            "strategy version must be a non-empty string",
        )
        fatal_flags.append(True)
        strategy_version = BUILTIN_STRATEGY_VERSION
    elif strategy_version != BUILTIN_STRATEGY_VERSION:
        _failure(
            failures,
            E_CONFIG_INVALID,
            "config",
            "strategy.version",
            "only the built-in strategy table version is supported",
            expected=BUILTIN_STRATEGY_VERSION,
            actual=strategy_version,
        )
        fatal_flags.append(True)

    samples, valid_seeds, valid_prompts = _load_samples(
        samples_path,
        expected_embedding_dim=embedding_dim,
        seed_max=seed_max,
        failures=failures,
        fatal_flags=fatal_flags,
    )

    config = WorkbenchConfig(
        schema_version=1 if raw.get("schema_version") == 1 else 0,
        source_path=str(config_file),
        project=ProjectConfig(name=project_name),
        prompt=PromptFileConfig(samples_path=str(samples_path), embedding_dim=embedding_dim),
        render=RenderConfig(
            denoise_steps=denoise_steps,
            guidance_scale=guidance_scale,
            resolution_tiers=tiers,
            batch_sizes=batch_sizes,
            latent_block_size=block_size,
            cache_budget_blocks=cache_budget,
            seed_max=seed_max,
            noise_table=noise_spec
            or TableSpec(None, tuple(noise_values), denoise_steps, "invalid"),
            guidance_table=guidance_spec
            or TableSpec(None, tuple(guidance_values), denoise_steps, "invalid"),
        ),
        img2img=Img2ImgConfig(
            enabled=img_enabled,
            replace_early_steps=img_steps,
            initial_image_path=image_path,
        ),
        runtime=RuntimeConfig(
            temp_dir=temp_dir,
            concurrency=concurrency,
            preview_budget_ms=preview_budget,
            snapshot_interval=snapshot_interval,
        ),
        strategy_version=strategy_version,
    )
    return LoadResult(
        config=config,
        samples=samples,
        noise_table=tuple(noise_values),
        guidance_table=tuple(guidance_values),
        failures=failures,
        fatal=any(fatal_flags),
        valid_seeds=valid_seeds,
        valid_prompts=valid_prompts,
    )


def clone_with_runtime(config: WorkbenchConfig, *, temp_dir: str | None = None) -> WorkbenchConfig:
    """Return a config copy with an overridden runtime temp directory."""

    if temp_dir is None:
        return replace(config)
    return replace(config, runtime=replace(config.runtime, temp_dir=str(Path(temp_dir).resolve())))


def state_fingerprint(result: LoadResult, *, batch_size: int, patches: list[dict[str, Any]] | None = None) -> str:
    """A semantic fingerprint independent of absolute source paths."""

    if result.config is None or result.samples is None:
        return value_fingerprint("invalid-state", result.to_dict())
    return value_fingerprint(
        "workbench-state-v1",
        {
            "config": result.config.semantic_dict(),
            "samples": result.samples.semantic_dict(),
            "noise_table": list(result.noise_table),
            "guidance_table": list(result.guidance_table),
            "batch_size": batch_size,
            "patches": patches or [],
        },
    )


def debug_state_json(result: LoadResult, *, batch_size: int, patches: list[dict[str, Any]] | None = None) -> str:
    """Canonical state JSON used by tests and the CLI decision record."""

    if result.config is None or result.samples is None:
        return canonical_json(result.to_dict())
    return canonical_json(
        {
            "config": result.config.semantic_dict(),
            "samples": result.samples.semantic_dict(),
            "noise_table": list(result.noise_table),
            "guidance_table": list(result.guidance_table),
            "batch_size": batch_size,
            "patches": patches or [],
        }
    )
