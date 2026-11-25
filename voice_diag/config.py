"""Registration configuration and command-flow validation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigFailure
from .fingerprint import sha256_json
from .jsonutil import LineLocator, parse_json_text

ALLOWED_QUANT_PROFILES = ("q8", "q12", "q16", "q20", "q24")
ALLOWED_FRACTIONAL_BITS = (2, 4, 6, 8, 10, 12)
ALLOWED_REPRESENTATIVE_RULES = ("composite", "lexicographic", "most_frames")


def profile_bits(profile: str) -> int:
    return int(profile[1:])


@dataclass
class RegistrationConfig:
    path: Path
    workspace: Path
    data: dict[str, Any]
    line_locator: LineLocator

    @property
    def version(self) -> str:
        return self.data["version"]

    @property
    def words(self) -> list[dict[str, Any]]:
        return self.data["words"]

    @property
    def template_dir(self) -> Path:
        return self.workspace / self.data["paths"]["templates"]

    @property
    def clip_dir(self) -> Path:
        return self.workspace / self.data["paths"]["clips"]

    @property
    def labels_path(self) -> Path:
        return self.workspace / self.data["paths"]["labels"]

    @property
    def features(self) -> dict[str, Any]:
        return self.data["features"]

    @property
    def projection(self) -> dict[str, Any]:
        return self.data["projection"]

    @property
    def quantization(self) -> dict[str, Any]:
        return self.data["quantization"]

    @property
    def thresholds(self) -> dict[str, Any]:
        return self.data["thresholds"]

    @property
    def registration(self) -> dict[str, Any]:
        return self.data["registration"]

    @property
    def audio(self) -> dict[str, Any]:
        return self.data["audio"]

    @property
    def profile(self) -> str:
        return self.quantization["profile"]

    @property
    def fractional_bits(self) -> int:
        return self.quantization["fractional_bits"]

    @property
    def reference_profile(self) -> str:
        return self.quantization["reference_profile"]

    @property
    def reject_threshold(self) -> int:
        return self.thresholds["reject_permille"]

    @property
    def confusion_threshold(self) -> int:
        return self.thresholds["confusion_candidate_permille"]

    @property
    def cluster_threshold(self) -> int:
        return self.thresholds["cluster_merge_permille"]

    @property
    def max_workers(self) -> int:
        return self.registration["max_workers"]

    @property
    def budget(self) -> int:
        return self.registration["comparison_budget"]

    @property
    def overlap_candidates(self) -> int:
        return self.registration["overlap_candidates"]

    @property
    def representative_rule(self) -> str:
        return self.registration["representative_rule"]

    @property
    def template_min(self) -> int:
        return self.registration["template_min"]

    @property
    def feature_fingerprint(self) -> str:
        return sha256_json(
            {
                "audio": self.audio,
                "features": self.features,
                "projection": self.projection,
            }
        )

    @property
    def runtime_fingerprint(self) -> str:
        return sha256_json(
            {
                "feature_fingerprint": self.feature_fingerprint,
                "quantization": self.quantization,
                "thresholds": self.thresholds,
                "registration": self.registration,
            }
        )

    @property
    def structural_fingerprint(self) -> str:
        return sha256_json(
            {
                "schema_version": self.data["schema_version"],
                "audio": self.audio,
                "features": self.features,
                "projection": self.projection,
            }
        )

    @property
    def config_fingerprint(self) -> str:
        return sha256_json(self.data)

    def template_map(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for word in self.words:
            for template in word["templates"]:
                key = f'{word["command_word"]}:{template["template_id"]}'
                result[key] = {
                    "command_word": word["command_word"],
                    "template_id": template["template_id"],
                    "path": self.workspace / template["path"],
                    "relative_path": template["path"],
                }
        return result


def _expect_dict(value: Any, source: str, line: int | None, field: str, sink: list) -> dict | None:
    if isinstance(value, dict):
        return value
    sink.append(ConfigFailure(source, line, field, "expected object").failure)
    return None


def _expect_list(value: Any, source: str, line: int | None, field: str, sink: list) -> list | None:
    if isinstance(value, list):
        return value
    sink.append(ConfigFailure(source, line, field, "expected array").failure)
    return None


def _expect_string(value: Any, source: str, line: int | None, field: str, sink: list) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    sink.append(ConfigFailure(source, line, field, "expected non-empty string").failure)
    return None


def _expect_int(
    value: Any,
    source: str,
    line: int | None,
    field: str,
    sink: list,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        sink.append(ConfigFailure(source, line, field, "expected integer").failure)
        return None
    if minimum is not None and value < minimum:
        sink.append(ConfigFailure(source, line, field, f"must be >= {minimum}").failure)
        return None
    if maximum is not None and value > maximum:
        sink.append(ConfigFailure(source, line, field, f"must be <= {maximum}").failure)
        return None
    return value


def _safe_workspace_relative(workspace: Path, raw_value: str) -> bool:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return False
    try:
        (workspace / candidate).resolve().relative_to(workspace.resolve())
    except ValueError:
        return False
    return True


def _next_reference_profile(profile: str, allowed: list[str]) -> str:
    candidates = [candidate for candidate in ALLOWED_QUANT_PROFILES if profile_bits(candidate) > profile_bits(profile)]
    for candidate in candidates:
        if candidate == "q24":
            return candidate
    if candidates:
        return candidates[0]
    return profile


def load_registration_config(path: Path, failure_sink: list) -> RegistrationConfig | None:
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failure_sink.append(ConfigFailure(source, None, "file", f"cannot read config: {exc}").failure)
        return None
    locator = LineLocator(text)
    root = parse_json_text(path, text, failure_sink)
    if root is None:
        return None
    if _expect_dict(root, source, None, "$", failure_sink) is None:
        return None
    failure_count = len(failure_sink)

    version = _expect_string(root.get("version"), source, locator.line_for('"version"'), "version", failure_sink)
    schema_version = _expect_int(
        root.get("schema_version", 1), source, locator.line_for('"schema_version"'), "schema_version", failure_sink, minimum=1
    )

    paths = _expect_dict(root.get("paths"), source, locator.line_for('"paths"'), "paths", failure_sink) or {}
    normalized_paths = {}
    for field, default in (("templates", "templates"), ("clips", "clips"), ("labels", "labels.jsonl")):
        raw_value = paths.get(field, default)
        value = _expect_string(raw_value, source, locator.line_for(f'"{field}"'), f"paths.{field}", failure_sink)
        if value is not None:
            if _safe_workspace_relative(path.parent, value):
                normalized_paths[field] = value
            else:
                failure_sink.append(
                    ConfigFailure(
                        source,
                        locator.line_for(f'"{field}"'),
                        f"paths.{field}",
                        "path must remain inside the config workspace",
                    ).failure
                )

    audio = _expect_dict(root.get("audio"), source, locator.line_for('"audio"'), "audio", failure_sink) or {}
    sample_rate = _expect_int(
        audio.get("sample_rate", 16000), source, locator.line_for('"sample_rate"'), "audio.sample_rate", failure_sink, minimum=1
    )
    channels = _expect_int(
        audio.get("channels", 1), source, locator.line_for('"channels"'), "audio.channels", failure_sink, minimum=1
    )
    bit_depth = _expect_int(
        audio.get("bit_depth", 16), source, locator.line_for('"bit_depth"'), "audio.bit_depth", failure_sink, minimum=1
    )
    if sample_rate != 16000:
        failure_sink.append(
            ConfigFailure(source, locator.line_for('"sample_rate"'), "audio.sample_rate", "must be 16000").failure
        )
    if channels != 1:
        failure_sink.append(ConfigFailure(source, locator.line_for('"channels"'), "audio.channels", "must be 1").failure)
    if bit_depth != 16:
        failure_sink.append(ConfigFailure(source, locator.line_for('"bit_depth"'), "audio.bit_depth", "must be 16").failure)

    features = _expect_dict(root.get("features"), source, locator.line_for('"features"'), "features", failure_sink) or {}
    frame_length = _expect_int(
        features.get("frame_length"), source, locator.line_for('"frame_length"'), "features.frame_length", failure_sink, minimum=1
    )
    frame_shift = _expect_int(
        features.get("frame_shift"), source, locator.line_for('"frame_shift"'), "features.frame_shift", failure_sink, minimum=1
    )
    dimension = _expect_int(
        features.get("dimension"), source, locator.line_for('"dimension"'), "features.dimension", failure_sink, minimum=2
    )
    if dimension is not None and dimension % 2:
        failure_sink.append(
            ConfigFailure(source, locator.line_for('"dimension"'), "features.dimension", "must be a positive even integer").failure
        )
    cepstral = _expect_int(
        features.get("cepstral_coefficients", 4),
        source,
        locator.line_for('"cepstral_coefficients"'),
        "features.cepstral_coefficients",
        failure_sink,
        minimum=1,
    )

    projection = _expect_dict(root.get("projection"), source, locator.line_for('"projection"'), "projection", failure_sink) or {}
    projection_seed = _expect_int(
        projection.get("seed"), source, locator.line_for('"seed"'), "projection.seed", failure_sink, minimum=0
    )
    projection_version = _expect_int(
        projection.get("seed_version", 1),
        source,
        locator.line_for('"seed_version"'),
        "projection.seed_version",
        failure_sink,
        minimum=1,
    )

    quantization = _expect_dict(
        root.get("quantization"), source, locator.line_for('"quantization"'), "quantization", failure_sink
    ) or {}
    profile = _expect_string(
        quantization.get("profile"), source, locator.line_for('"profile"'), "quantization.profile", failure_sink
    )
    allowed_profiles_raw = _expect_list(
        quantization.get("allowed_profiles"),
        source,
        locator.line_for('"allowed_profiles"'),
        "quantization.allowed_profiles",
        failure_sink,
    ) or []
    allowed_profiles: list[str] = []
    for index, candidate in enumerate(allowed_profiles_raw):
        if not isinstance(candidate, str) or candidate not in ALLOWED_QUANT_PROFILES:
            failure_sink.append(
                ConfigFailure(
                    source,
                    locator.line_for('"allowed_profiles"'),
                    f"quantization.allowed_profiles[{index}]",
                    f"must be one of {list(ALLOWED_QUANT_PROFILES)}",
                ).failure
            )
        else:
            allowed_profiles.append(candidate)
    if not allowed_profiles:
        failure_sink.append(
            ConfigFailure(
                source,
                locator.line_for('"allowed_profiles"'),
                "quantization.allowed_profiles",
                "must be a non-empty array",
            ).failure
        )
    if profile is not None and allowed_profiles and profile not in allowed_profiles:
        failure_sink.append(
            ConfigFailure(
                source,
                locator.line_for('"profile"'),
                "quantization.profile",
                f"profile {profile!r} is not in allowed_profiles",
            ).failure
        )
    fractional_bits = _expect_int(
        quantization.get("fractional_bits"),
        source,
        locator.line_for('"fractional_bits"'),
        "quantization.fractional_bits",
        failure_sink,
    )
    allowed_bits_raw = _expect_list(
        quantization.get("allowed_fractional_bits"),
        source,
        locator.line_for('"allowed_fractional_bits"'),
        "quantization.allowed_fractional_bits",
        failure_sink,
    ) or []
    allowed_bits: list[int] = []
    for index, candidate in enumerate(allowed_bits_raw):
        if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate not in ALLOWED_FRACTIONAL_BITS:
            failure_sink.append(
                ConfigFailure(
                    source,
                    locator.line_for('"allowed_fractional_bits"'),
                    f"quantization.allowed_fractional_bits[{index}]",
                    f"must be one of {list(ALLOWED_FRACTIONAL_BITS)}",
                ).failure
            )
        else:
            allowed_bits.append(candidate)
    if not allowed_bits:
        failure_sink.append(
            ConfigFailure(
                source,
                locator.line_for('"allowed_fractional_bits"'),
                "quantization.allowed_fractional_bits",
                "must be a non-empty array",
            ).failure
        )
    if fractional_bits is not None:
        if fractional_bits not in ALLOWED_FRACTIONAL_BITS:
            failure_sink.append(
                ConfigFailure(
                    source,
                    locator.line_for('"fractional_bits"'),
                    "quantization.fractional_bits",
                    f"must be one of {list(ALLOWED_FRACTIONAL_BITS)}",
                ).failure
            )
        elif allowed_bits and fractional_bits not in allowed_bits:
            failure_sink.append(
                ConfigFailure(
                    source,
                    locator.line_for('"fractional_bits"'),
                    "quantization.fractional_bits",
                    "fractional_bits is not in allowed_fractional_bits",
                ).failure
            )
    reference_raw = quantization.get("reference_profile")
    if reference_raw is None and profile is not None:
        reference_profile = _next_reference_profile(profile, allowed_profiles)
    else:
        reference_profile = _expect_string(
            reference_raw,
            source,
            locator.line_for('"reference_profile"'),
            "quantization.reference_profile",
            failure_sink,
        )
    if reference_profile is not None and reference_profile not in ALLOWED_QUANT_PROFILES:
        failure_sink.append(
            ConfigFailure(
                source,
                locator.line_for('"reference_profile"'),
                "quantization.reference_profile",
                f"must be one of {list(ALLOWED_QUANT_PROFILES)}",
            ).failure
        )
    if (
        profile is not None
        and profile in ALLOWED_QUANT_PROFILES
        and reference_profile is not None
        and reference_profile in ALLOWED_QUANT_PROFILES
        and profile_bits(reference_profile) <= profile_bits(profile)
    ):
        reference_profile = next(
            (
                candidate
                for candidate in ALLOWED_QUANT_PROFILES
                if profile_bits(candidate) > profile_bits(profile)
            ),
            profile,
        )

    thresholds = _expect_dict(root.get("thresholds"), source, locator.line_for('"thresholds"'), "thresholds", failure_sink) or {}
    reject = _expect_int(
        thresholds.get("reject_permille"),
        source,
        locator.line_for('"reject_permille"'),
        "thresholds.reject_permille",
        failure_sink,
        minimum=0,
        maximum=1000,
    )
    confusion_threshold = _expect_int(
        thresholds.get("confusion_candidate_permille"),
        source,
        locator.line_for('"confusion_candidate_permille"'),
        "thresholds.confusion_candidate_permille",
        failure_sink,
        minimum=0,
        maximum=1000,
    )
    cluster_threshold = _expect_int(
        thresholds.get("cluster_merge_permille"),
        source,
        locator.line_for('"cluster_merge_permille"'),
        "thresholds.cluster_merge_permille",
        failure_sink,
        minimum=0,
        maximum=1000,
    )

    registration = _expect_dict(
        root.get("registration"), source, locator.line_for('"registration"'), "registration", failure_sink
    ) or {}
    template_min = _expect_int(
        registration.get("template_min", 1),
        source,
        locator.line_for('"template_min"'),
        "registration.template_min",
        failure_sink,
        minimum=1,
    )
    max_workers = _expect_int(
        registration.get("max_workers", 1),
        source,
        locator.line_for('"max_workers"'),
        "registration.max_workers",
        failure_sink,
        minimum=1,
        maximum=64,
    )
    budget = _expect_int(
        registration.get("comparison_budget", 1_000_000),
        source,
        locator.line_for('"comparison_budget"'),
        "registration.comparison_budget",
        failure_sink,
        minimum=0,
    )
    overlap = _expect_int(
        registration.get("overlap_candidates", 3),
        source,
        locator.line_for('"overlap_candidates"'),
        "registration.overlap_candidates",
        failure_sink,
        minimum=1,
    )
    representative_rule = _expect_string(
        registration.get("representative_rule", "composite"),
        source,
        locator.line_for('"representative_rule"'),
        "registration.representative_rule",
        failure_sink,
    )
    if representative_rule is not None and representative_rule not in ALLOWED_REPRESENTATIVE_RULES:
        failure_sink.append(
            ConfigFailure(
                source,
                locator.line_for('"representative_rule"'),
                "registration.representative_rule",
                f"must be one of {list(ALLOWED_REPRESENTATIVE_RULES)}",
            ).failure
        )

    words_raw = _expect_list(root.get("words"), source, locator.line_for('"words"'), "words", failure_sink) or []
    normalized_words: list[dict[str, Any]] = []
    seen_words: set[str] = set()
    for word_index, raw_word in enumerate(words_raw):
        word_line = locator.line_for('"command_word"')
        word = _expect_dict(raw_word, source, word_line, f"words[{word_index}]", failure_sink)
        if word is None:
            continue
        command_word = _expect_string(
            word.get("command_word"), source, word_line, f"words[{word_index}].command_word", failure_sink
        )
        if command_word is None:
            continue
        if command_word in seen_words:
            failure_sink.append(
                ConfigFailure(
                    source, word_line, f"words[{word_index}].command_word", f"duplicate command word {command_word!r}"
                ).failure
            )
        seen_words.add(command_word)
        templates_raw = _expect_list(
            word.get("templates"), source, locator.line_for('"templates"'), f"words[{word_index}].templates", failure_sink
        ) or []
        if template_min is not None and len(templates_raw) < template_min:
            failure_sink.append(
                ConfigFailure(
                    source,
                    locator.line_for('"templates"'),
                    f"words[{word_index}].templates",
                    f"template count {len(templates_raw)} is below registration minimum {template_min}",
                ).failure
            )
        normalized_templates: list[dict[str, Any]] = []
        seen_templates: set[str] = set()
        for template_index, raw_template in enumerate(templates_raw):
            template = _expect_dict(
                raw_template,
                source,
                locator.line_for('"template_id"'),
                f"words[{word_index}].templates[{template_index}]",
                failure_sink,
            )
            if template is None:
                continue
            template_id = _expect_string(
                template.get("template_id"),
                source,
                locator.line_for('"template_id"'),
                f"words[{word_index}].templates[{template_index}].template_id",
                failure_sink,
            )
            if template_id is None:
                continue
            if template_id in seen_templates:
                failure_sink.append(
                    ConfigFailure(
                        source,
                        locator.line_for('"template_id"'),
                        f"words[{word_index}].templates[{template_index}].template_id",
                        f"duplicate template id {template_id!r} within command word {command_word!r}",
                    ).failure
                )
            seen_templates.add(template_id)
            relative_path = _expect_string(
                template.get("path"),
                source,
                locator.line_for('"path"'),
                f"words[{word_index}].templates[{template_index}].path",
                failure_sink,
            )
            if relative_path is None:
                continue
            if not _safe_workspace_relative(path.parent, relative_path):
                failure_sink.append(
                    ConfigFailure(
                        source,
                        locator.line_for('"path"'),
                        f"words[{word_index}].templates[{template_index}].path",
                        "template path must remain inside the config workspace",
                    ).failure
                )
                continue
            normalized_templates.append({"template_id": template_id, "path": relative_path})
        normalized_words.append({"command_word": command_word, "templates": normalized_templates})

    if len(failure_sink) != failure_count:
        return None

    normalized = {
        "schema_version": schema_version,
        "version": version,
        "paths": normalized_paths,
        "audio": {"sample_rate": sample_rate, "channels": channels, "bit_depth": bit_depth},
        "features": {
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "dimension": dimension,
            "cepstral_coefficients": cepstral,
        },
        "projection": {"seed": projection_seed, "seed_version": projection_version},
        "quantization": {
            "profile": profile,
            "fractional_bits": fractional_bits,
            "allowed_profiles": allowed_profiles,
            "allowed_fractional_bits": allowed_bits,
            "reference_profile": reference_profile,
        },
        "thresholds": {
            "reject_permille": reject,
            "confusion_candidate_permille": confusion_threshold,
            "cluster_merge_permille": cluster_threshold,
        },
        "registration": {
            "template_min": template_min,
            "max_workers": max_workers,
            "comparison_budget": budget,
            "overlap_candidates": overlap,
            "representative_rule": representative_rule,
        },
        "words": normalized_words,
    }
    config = RegistrationConfig(path=path.resolve(), workspace=path.resolve().parent, data=normalized, line_locator=locator)
    required_paths = [config.template_dir, config.clip_dir]
    for required in required_paths:
        if not required.is_dir():
            failure_sink.append(
                ConfigFailure(source, locator.line_for('"paths"'), "paths", f"directory does not exist: {required.name}").failure
            )
    if not config.labels_path.is_file():
        failure_sink.append(
            ConfigFailure(source, locator.line_for('"labels"'), "paths.labels", f"labels file does not exist: {config.labels_path.name}").failure
        )
    if len(failure_sink) != failure_count:
        return None
    return config


def clone_config_data(config: RegistrationConfig) -> dict[str, Any]:
    return copy.deepcopy(config.data)
