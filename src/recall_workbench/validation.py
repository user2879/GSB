"""Strict, non-fail-fast loading for JSONL inputs and experiment configuration."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    E_CONFIG_INVALID,
    E_INPUT_NOT_FOUND,
    E_RECORD_INVALID,
    ConfigInvalid,
    FailureRecord,
    InputNotFound,
    sorted_failures,
)
from .fixed import round_half_even
from .models import AppConfig, ChannelConfig, CHANNEL_ORDER, Dataset, FeatureConfig, Interaction, Item, TruthEntry


SUPPORTED_FEATURES = {
    "lexical_norm",
    "collaborative_norm",
    "popularity_norm",
    "history_overlap",
    "freshness",
    "popularity_prior",
}


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure(source: str, line: int | None, field: str, message: str, code: str = E_RECORD_INVALID) -> FailureRecord:
    return FailureRecord(code=code, source=source, line=line, field=field, message=message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_config(path: Path) -> tuple[AppConfig | None, list[FailureRecord]]:
    source = path.name
    if not path.exists():
        raise InputNotFound(f"configuration file not found: {path}", details={"path": str(path)})
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [_failure(source, None, "$", f"cannot parse configuration JSON: {exc}", E_CONFIG_INVALID)]
    if not isinstance(raw, dict):
        return None, [_failure(source, None, "$", "configuration root must be a JSON object", E_CONFIG_INVALID)]

    failures: list[FailureRecord] = []

    def reject(field: str, message: str) -> None:
        failures.append(_failure(source, None, field, message, E_CONFIG_INVALID))

    schema_version = raw.get("schema_version", 1)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version != 1:
        reject("schema_version", "must be integer 1")

    seed = raw.get("seed", 0)
    if not isinstance(seed, int) or isinstance(seed, bool):
        reject("seed", "must be an integer")

    event_types_raw = raw.get("event_types")
    event_types: tuple[str, ...] = ()
    if not isinstance(event_types_raw, list) or not event_types_raw:
        reject("event_types", "must be a non-empty list of strings")
    else:
        cleaned = []
        for index, event_type in enumerate(event_types_raw):
            if not isinstance(event_type, str) or not event_type.strip():
                reject(f"event_types[{index}]", "must be a non-empty string")
            elif event_type in cleaned:
                reject(f"event_types[{index}]", f"duplicate event type: {event_type}")
            else:
                cleaned.append(event_type)
        event_types = tuple(cleaned)

    event_weights_raw = raw.get("event_weights")
    event_weights: dict[str, float] = {}
    if not isinstance(event_weights_raw, dict):
        reject("event_weights", "must be an object mapping event types to positive weights")
    else:
        for name in event_types:
            value = event_weights_raw.get(name)
            if not _is_number(value) or float(value) <= 0:
                reject(f"event_weights.{name}", "must be a finite number greater than zero")
            else:
                event_weights[name] = float(value)
        for name in event_weights_raw:
            if name not in event_types:
                reject(f"event_weights.{name}", "event type is not declared in event_types")

    channels_raw = raw.get("channels")
    channels: dict[str, ChannelConfig] = {}
    if not isinstance(channels_raw, dict):
        reject("channels", "must be an object")
    else:
        for name in channels_raw:
            if name not in CHANNEL_ORDER:
                reject(f"channels.{name}", "unknown channel name")
        for name in CHANNEL_ORDER:
            channel = channels_raw.get(name)
            if not isinstance(channel, dict):
                reject(f"channels.{name}", "missing channel object")
                continue
            enabled = channel.get("enabled")
            weight = channel.get("weight")
            limit = channel.get("candidate_limit")
            if not isinstance(enabled, bool):
                reject(f"channels.{name}.enabled", "must be boolean")
            if not _is_number(weight) or float(weight) < 0:
                reject(f"channels.{name}.weight", "must be a finite number greater than or equal to zero")
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 10000:
                reject(f"channels.{name}.candidate_limit", "must be an integer between 1 and 10000")
            if isinstance(enabled, bool) and _is_number(weight) and isinstance(limit, int) and not isinstance(limit, bool):
                channels[name] = ChannelConfig(enabled=enabled, weight=float(weight), candidate_limit=limit)
        enabled_channels = [channel for channel in channels.values() if channel.enabled]
        if channels and not enabled_channels:
            reject("channels", "at least one channel must be enabled")
        if enabled_channels and sum(channel.weight for channel in enabled_channels) <= 0:
            reject("channels", "sum of enabled channel weights must be greater than zero")

    fusion_raw = raw.get("fusion")
    strategy = "weighted_sum"
    temperature = 1.0
    normalization = "max"
    zero_policy = "redistribute"
    if not isinstance(fusion_raw, dict):
        reject("fusion", "must be an object")
    else:
        strategy = fusion_raw.get("strategy", strategy)
        if strategy != "weighted_sum":
            reject("fusion.strategy", "only 'weighted_sum' is supported")
        temperature = fusion_raw.get("temperature", temperature)
        if not _is_number(temperature) or float(temperature) <= 0:
            reject("fusion.temperature", "must be a finite number greater than zero")
        normalization = fusion_raw.get("normalization", normalization)
        if normalization not in {"max", "min_max"}:
            reject("fusion.normalization", "must be 'max' or 'min_max'")
        zero_policy = fusion_raw.get("zero_channel_policy", zero_policy)
        if zero_policy not in {"redistribute", "none"}:
            reject("fusion.zero_channel_policy", "must be 'redistribute' or 'none'")

    features_raw = raw.get("ranking_features", [])
    features: list[FeatureConfig] = []
    if not isinstance(features_raw, list):
        reject("ranking_features", "must be a list")
    else:
        seen_features: set[str] = set()
        for index, feature in enumerate(features_raw):
            if not isinstance(feature, dict):
                reject(f"ranking_features[{index}]", "must be an object")
                continue
            name = feature.get("name")
            weight = feature.get("weight")
            if not isinstance(name, str) or name not in SUPPORTED_FEATURES:
                reject(f"ranking_features[{index}].name", f"unsupported feature name: {name!r}")
                continue
            if name in seen_features:
                reject(f"ranking_features[{index}].name", f"duplicate feature: {name}")
                continue
            if not _is_number(weight) or float(weight) < 0:
                reject(f"ranking_features[{index}].weight", "must be a finite number greater than or equal to zero")
                continue
            seen_features.add(name)
            features.append(FeatureConfig(name=name, weight=float(weight)))

    fixed_raw = raw.get("fixed_point")
    primary_bits = 8
    reference_bits = 12
    tolerance = 0.0
    rounding = "half_even"
    if not isinstance(fixed_raw, dict):
        reject("fixed_point", "must be an object")
    else:
        primary_bits = fixed_raw.get("primary_bits", primary_bits)
        reference_bits = fixed_raw.get("reference_bits", reference_bits)
        tolerance = fixed_raw.get("tolerance", tolerance)
        rounding = fixed_raw.get("rounding", rounding)
        if not isinstance(primary_bits, int) or isinstance(primary_bits, bool) or not 1 <= primary_bits <= 29:
            reject("fixed_point.primary_bits", "must be an integer between 1 and 29")
        if not isinstance(reference_bits, int) or isinstance(reference_bits, bool) or not 2 <= reference_bits <= 30:
            reject("fixed_point.reference_bits", "must be an integer between 2 and 30")
        if isinstance(primary_bits, int) and isinstance(reference_bits, int) and reference_bits <= primary_bits:
            reject("fixed_point.reference_bits", "must be greater than primary_bits")
        if not _is_number(tolerance) or float(tolerance) < 0:
            reject("fixed_point.tolerance", "must be a finite number greater than or equal to zero")
        if rounding != "half_even":
            reject("fixed_point.rounding", "only deterministic 'half_even' rounding is supported")

    clip_raw = raw.get("score_clip", {"min": 0.0, "max": 1.0})
    score_min = 0.0
    score_max = 1.0
    if not isinstance(clip_raw, dict):
        reject("score_clip", "must be an object")
    else:
        score_min = clip_raw.get("min", score_min)
        score_max = clip_raw.get("max", score_max)
        if not _is_number(score_min) or not _is_number(score_max) or float(score_min) >= float(score_max):
            reject("score_clip", "min and max must be finite numbers with min < max")

    retrieval_raw = raw.get("retrieval")
    half_life_ms = 86_400_000.0
    popularity_floor = 0.0
    if not isinstance(retrieval_raw, dict):
        reject("retrieval", "must be an object")
    else:
        half_life_ms = retrieval_raw.get("half_life_ms", half_life_ms)
        popularity_floor = retrieval_raw.get("popularity_floor", popularity_floor)
        if not _is_number(half_life_ms) or float(half_life_ms) <= 0:
            reject("retrieval.half_life_ms", "must be a finite number greater than zero")
        if not _is_number(popularity_floor) or float(popularity_floor) < 0:
            reject("retrieval.popularity_floor", "must be a finite number greater than or equal to zero")

    execution_raw = raw.get("execution")
    max_workers = 1
    commit_interval = 1
    top_k = 10
    if not isinstance(execution_raw, dict):
        reject("execution", "must be an object")
    else:
        max_workers = execution_raw.get("max_workers", max_workers)
        commit_interval = execution_raw.get("snapshot_commit_interval", commit_interval)
        top_k = execution_raw.get("top_k", top_k)
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or not 1 <= max_workers <= 256:
            reject("execution.max_workers", "must be an integer between 1 and 256")
        if not isinstance(commit_interval, int) or isinstance(commit_interval, bool) or commit_interval < 1:
            reject("execution.snapshot_commit_interval", "must be an integer greater than or equal to 1")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            reject("execution.top_k", "must be an integer greater than or equal to 1")

    if failures:
        return None, sorted_failures(failures)

    if round_half_even(float(temperature) * (1 << primary_bits)) < 1:
        failures.append(
            _failure(
                source,
                None,
                "fusion.temperature",
                "temperature is below the primary fixed-point resolution",
                E_CONFIG_INVALID,
            )
        )
        return None, sorted_failures(failures)

    config = AppConfig(
        raw=raw,
        config_hash=sha256_bytes(stable_json_bytes(raw)),
        schema_version=schema_version,
        seed=seed,
        event_types=event_types,
        event_weights=event_weights,
        channels=channels,
        fusion_strategy=strategy,
        temperature=float(temperature),
        normalization=normalization,
        zero_channel_policy=zero_policy,
        ranking_features=tuple(features),
        primary_bits=primary_bits,
        reference_bits=reference_bits,
        tolerance=float(tolerance),
        rounding=rounding,
        score_min=float(score_min),
        score_max=float(score_max),
        half_life_ms=float(half_life_ms),
        popularity_floor=float(popularity_floor),
        max_workers=max_workers,
        snapshot_commit_interval=commit_interval,
        top_k=top_k,
    )
    return config, []


def _parse_event_ms(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp() * 1000)
        if not math.isfinite(numeric) or numeric < 0:
            return None
        return int(numeric)
    return None


def _iter_jsonl(path: Path, source: str, failures: list[FailureRecord]) -> Iterable[tuple[int, Any]]:
    try:
        handle = path.open("rb")
    except OSError as exc:
        failures.append(_failure(source, None, "$", f"cannot open file: {exc}"))
        return
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number == 1:
                line = line.lstrip(b"\xef\xbb\xbf")
            if not line.strip():
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError as exc:
                failures.append(_failure(source, line_number, "$", f"invalid UTF-8: {exc.reason}"))
                continue
            try:
                yield line_number, json.loads(text)
            except json.JSONDecodeError as exc:
                failures.append(_failure(source, line_number, "$", f"invalid JSON: {exc.msg}"))


def _require_string(record: dict[str, Any], field: str) -> str | None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def load_catalog(path: Path, failures: list[FailureRecord]) -> dict[str, Item]:
    source = path.name
    items: dict[str, Item] = {}
    for line_number, raw in _iter_jsonl(path, source, failures):
        if not isinstance(raw, dict):
            failures.append(_failure(source, line_number, "$", "record must be a JSON object"))
            continue
        item_id = _require_string(raw, "item_id")
        if item_id is None:
            failures.append(_failure(source, line_number, "item_id", "must be a non-empty string"))
            continue
        terms_raw = raw.get("terms")
        if not isinstance(terms_raw, list) or not terms_raw:
            failures.append(_failure(source, line_number, "terms", "must be a non-empty list"))
            continue
        terms: list[str] = []
        invalid_term = False
        for index, term in enumerate(terms_raw):
            if not isinstance(term, str) or not term.strip():
                failures.append(_failure(source, line_number, f"terms[{index}]", "must be a non-empty string"))
                invalid_term = True
                break
            if term.strip() not in terms:
                terms.append(term.strip())
        if invalid_term:
            continue
        if item_id in items:
            failures.append(_failure(source, line_number, "item_id", f"duplicate item id: {item_id}"))
            continue
        items[item_id] = Item(item_id=item_id, terms=tuple(terms), line=line_number)
    return items


def load_interactions(path: Path, config: AppConfig, catalog: dict[str, Item], failures: list[FailureRecord]) -> list[Interaction]:
    source = path.name
    records: list[Interaction] = []
    for line_number, raw in _iter_jsonl(path, source, failures):
        if not isinstance(raw, dict):
            failures.append(_failure(source, line_number, "$", "record must be a JSON object"))
            continue
        user_id = _require_string(raw, "user_id")
        item_id = _require_string(raw, "item_id")
        event_type = _require_string(raw, "event_type")
        event_ms = _parse_event_ms(raw.get("event_ms"))
        invalid = False
        if user_id is None:
            failures.append(_failure(source, line_number, "user_id", "must be a non-empty string"))
            invalid = True
        if item_id is None:
            failures.append(_failure(source, line_number, "item_id", "must be a non-empty string"))
            invalid = True
        elif item_id not in catalog:
            failures.append(_failure(source, line_number, "item_id", f"item id not present in catalog: {item_id}"))
            invalid = True
        if event_type is None:
            failures.append(_failure(source, line_number, "event_type", "must be a non-empty string"))
            invalid = True
        elif event_type not in config.event_types:
            failures.append(_failure(source, line_number, "event_type", f"event type not configured: {event_type}"))
            invalid = True
        if event_ms is None:
            failures.append(_failure(source, line_number, "event_ms", "must be a parseable non-negative millisecond timestamp"))
            invalid = True
        if invalid:
            continue
        records.append(
            Interaction(user_id=user_id, item_id=item_id, event_ms=event_ms, event_type=event_type, line=line_number)
        )
    return records


def load_truth(path: Path, failures: list[FailureRecord]) -> dict[str, TruthEntry]:
    source = path.name
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        failures.append(_failure(source, None, "$", f"cannot parse truth JSON: {exc}"))
        return {}
    users_raw = raw.get("users") if isinstance(raw, dict) else None
    if not isinstance(users_raw, dict):
        failures.append(_failure(source, None, "users", "truth root must contain a users object"))
        return {}
    truth: dict[str, TruthEntry] = {}
    for index, (user_id, value) in enumerate(users_raw.items()):
        line = index + 1
        if not isinstance(user_id, str) or not user_id.strip():
            failures.append(_failure(source, line, "user_id", "must be a non-empty string"))
            continue
        if not isinstance(value, dict):
            failures.append(_failure(source, line, f"users.{user_id}", "must be an object"))
            continue
        candidates = value.get("candidates")
        ranking = value.get("ranking")
        if not isinstance(candidates, list) or not isinstance(ranking, list):
            failures.append(_failure(source, line, f"users.{user_id}", "candidates and ranking must be lists"))
            continue
        if not all(isinstance(item_id, str) and item_id.strip() for item_id in candidates + ranking):
            failures.append(_failure(source, line, f"users.{user_id}", "truth item ids must be non-empty strings"))
            continue
        if len(set(candidates)) != len(candidates) or len(set(ranking)) != len(ranking):
            failures.append(_failure(source, line, f"users.{user_id}", "truth lists must not contain duplicates"))
            continue
        truth[user_id.strip()] = TruthEntry(candidates=tuple(candidates), ranking=tuple(ranking))
    return truth


def discover_inputs(input_dir: Path) -> dict[str, Path]:
    return {
        "config": input_dir / "config.json",
        "interactions": input_dir / "interactions.jsonl",
        "catalog": input_dir / "items.jsonl",
        "truth": input_dir / "truth.json",
    }


def load_dataset(input_dir: Path, config: AppConfig) -> Dataset:
    paths = discover_inputs(input_dir)
    for name, path in paths.items():
        if not path.exists():
            raise InputNotFound(f"required {name} input not found: {path}", details={"path": str(path)})
    failures: list[FailureRecord] = []
    catalog = load_catalog(paths["catalog"], failures)
    interactions = load_interactions(paths["interactions"], config, catalog, failures)
    truth = load_truth(paths["truth"], failures)
    file_hashes = {path.name: sha256_file(path) for path in paths.values()}
    return Dataset(
        interactions=interactions,
        items=catalog,
        truth=truth,
        failures=sorted_failures(failures),
        file_hashes=file_hashes,
    )


def config_error_payload(failures: list[FailureRecord]) -> dict[str, Any]:
    return {"code": E_CONFIG_INVALID, "failures": [failure.to_dict() for failure in failures]}
