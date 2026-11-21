"""Fragment JSONL loading, validation and isolated fault injection."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .config import FusionConfig
from .diagnostics import failure, sort_failures
from .hashing import fragment_set_hash
from .io_utils import read_json, sha256_bytes


INJECTION_CODE_MAP = {
    "nonfinite_score": "E_NONFINITE_SCORE",
    "duplicate_candidate": "E_CONFIG_INVALID",
    "trunc_limit_out_of_range": "E_CONFIG_INVALID",
    "zero_capacity": "E_CONFIG_INVALID",
}

INJECTION_STAGE_MAP = {
    "nonfinite_score": "load",
    "duplicate_candidate": "load",
    "trunc_limit_out_of_range": "config",
    "zero_capacity": "config",
}


def injection_hit(injection: dict[str, Any], *, channel: str | None = None) -> dict[str, Any]:
    injection_type = str(injection.get("type"))
    return {
        "channel": channel if channel is not None else injection.get("channel"),
        "id": injection.get("id"),
        "mapped_code": INJECTION_CODE_MAP.get(injection_type, "E_CONFIG_INVALID"),
        "stage": injection.get("stage"),
        "type": injection_type,
    }


def load_injections(path: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if path is None:
        return [], []
    try:
        payload = read_json(path)
    except Exception as exc:
        return [], [
            failure("config", "E_CONFIG_INVALID", f"cannot read injection config: {exc}")
        ]
    if not isinstance(payload, dict) or not isinstance(payload.get("injections", []), list):
        return [], [
            failure("config", "E_CONFIG_INVALID", "injection config must contain an injections list")
        ]
    injections: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("injections", [])):
        if not isinstance(item, dict):
            failures.append(
                failure("config", "E_CONFIG_INVALID", f"injection #{index} must be an object")
            )
            continue
        injection_id = item.get("id")
        injection_type = item.get("type")
        stage = item.get("stage")
        if not isinstance(injection_id, str) or not injection_id:
            failures.append(
                failure(
                    "config",
                    "E_CONFIG_INVALID",
                    f"injection #{index} has no valid id",
                    injection_id=injection_id if isinstance(injection_id, str) else None,
                )
            )
            continue
        expected_stage = INJECTION_STAGE_MAP.get(injection_type)
        if expected_stage is None:
            failures.append(
                failure(
                    "config",
                    "E_CONFIG_INVALID",
                    f"unsupported injection type: {injection_type!r}",
                    injection_id=injection_id,
                )
            )
            continue
        if stage != expected_stage:
            failures.append(
                failure(
                    "config",
                    "E_CONFIG_INVALID",
                    f"injection {injection_id} type {injection_type} requires stage {expected_stage}",
                    injection_id=injection_id,
                )
            )
            continue
        normalized = dict(item)
        normalized["stage"] = stage
        injections.append(normalized)
    return injections, sort_failures(failures)


def apply_config_injections(
    raw_config: Any,
    injections: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply config-stage injections to a copy of the raw config."""

    if not injections:
        return raw_config, [], []
    config = copy.deepcopy(raw_config)
    failures: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for injection in injections:
        injection_type = injection.get("type")
        if injection_type not in {"trunc_limit_out_of_range", "zero_capacity"}:
            continue
        injection_id = str(injection.get("id"))
        if injection_type == "trunc_limit_out_of_range":
            if not isinstance(config, dict):
                failures.append(
                    failure(
                        "config",
                        "E_CONFIG_INVALID",
                        "cannot apply truncation-limit injection to non-object config",
                        injection_id=injection_id,
                    )
                )
                continue
            config["trunc_k"] = injection.get("value")
            hits.append(injection_hit(injection))
            continue

        channel = injection.get("channel")
        if (
            isinstance(config, dict)
            and isinstance(config.get("channels"), dict)
            and isinstance(channel, str)
            and isinstance(config["channels"].get(channel), dict)
        ):
            config["channels"][channel]["capacity"] = 0
            hits.append(injection_hit(injection, channel=channel))
        else:
            failures.append(
                failure(
                    "config",
                    "E_CONFIG_INVALID",
                    f"zero-capacity injection target not found: {channel!r}",
                    channel=channel if isinstance(channel, str) else None,
                    injection_id=injection_id,
                )
            )
    return config, hits, sort_failures(failures)


def _coerce_nonfinite(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"nan", "+nan", "-nan"}:
            return float("nan")
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if lowered in {"-inf", "-infinity"}:
            return float("-inf")
        return float(value)
    return float(value)


def _raw_channel(item: dict[str, Any], default_channel: str) -> str:
    channel = item.get("channel", default_channel)
    return channel if isinstance(channel, str) else ""


def load_fragments(
    workspace: Path,
    config: FusionConfig,
    injections: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all configured channel fragments and apply load-stage injections."""

    failures: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    fragments_root = workspace / "fragments"
    raw_entries_by_channel: dict[str, list[dict[str, Any]]] = {
        name: [] for name in config.channel_names
    }

    configured = set(config.channel_names)
    discovered: set[str] = set()
    if fragments_root.is_dir():
        for child in sorted(fragments_root.iterdir(), key=lambda item: item.name):
            if child.is_dir():
                discovered.add(child.name)

    for unexpected in sorted(discovered - configured):
        failures.append(
            failure(
                "load",
                "E_CONFIG_INVALID",
                f"fragment directory for unconfigured channel: {unexpected}",
                channel=unexpected,
            )
        )

    fragments_by_channel: dict[str, list[dict[str, Any]]] = {}
    for channel in config.channel_names:
        channel_dir = fragments_root / channel
        fragments: list[dict[str, Any]] = []
        fragment_files: list[Path] = []
        if channel_dir.is_dir():
            fragment_files = sorted(channel_dir.glob("*.jsonl"), key=lambda item: item.name)
        for fragment_path in fragment_files:
            fragment_id = f"{channel}/{fragment_path.name}"
            try:
                data = fragment_path.read_bytes()
            except OSError as exc:
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"cannot read fragment {fragment_id}: {exc}",
                        channel=channel,
                    )
                )
                continue
            fragment_sha = sha256_bytes(data)
            fragments.append(
                {
                    "id": fragment_id,
                    "line_count": len(data.splitlines()),
                    "path": str(fragment_path.relative_to(workspace)),
                    "sha256": fragment_sha,
                }
            )
            text = data.decode("utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                entry: dict[str, Any] = {
                    "channel_default": channel,
                    "fragment_id": fragment_id,
                    "fragment_sha256": fragment_sha,
                    "injection_id": None,
                    "line_number": line_number,
                }
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    failures.append(
                        failure(
                            "load",
                            "E_CONFIG_INVALID",
                            f"invalid JSON in {fragment_id}:{line_number}: {exc}",
                            channel=channel,
                        )
                    )
                    continue
                if not isinstance(value, dict):
                    failures.append(
                        failure(
                            "load",
                            "E_CONFIG_INVALID",
                            f"fragment line must be an object: {fragment_id}:{line_number}",
                            channel=channel,
                        )
                    )
                    continue
                entry["data"] = value
                raw_entries_by_channel[channel].append(entry)
        fragments_by_channel[channel] = fragments

    # Apply content injections to the parsed stream before validation. This
    # deliberately makes the loader observe duplicate/non-finite input rather
    # than merely reporting that an injection was requested.
    for injection in injections:
        injection_type = injection.get("type")
        if injection_type not in {"nonfinite_score", "duplicate_candidate"}:
            continue
        injection_id = str(injection.get("id"))
        channel = injection.get("channel")
        candidate_id = injection.get("candidate_id")
        if not isinstance(channel, str) or channel not in raw_entries_by_channel:
            failures.append(
                failure(
                    "load",
                    "E_CONFIG_INVALID",
                    f"load injection target channel not found: {channel!r}",
                    channel=channel if isinstance(channel, str) else None,
                    candidate_id=candidate_id if isinstance(candidate_id, str) else None,
                    injection_id=injection_id,
                )
            )
            continue
        entries = raw_entries_by_channel[channel]
        target_index = next(
            (
                index
                for index, entry in enumerate(entries)
                if entry["data"].get("candidate_id") == candidate_id
                and _raw_channel(entry["data"], channel) == channel
            ),
            None,
        )
        if target_index is None:
            failures.append(
                failure(
                    "load",
                    "E_CONFIG_INVALID",
                    f"load injection target not found: {channel}/{candidate_id}",
                    channel=channel,
                    candidate_id=candidate_id if isinstance(candidate_id, str) else None,
                    injection_id=injection_id,
                )
            )
            continue
        target = entries[target_index]
        if injection_type == "nonfinite_score":
            try:
                target["data"]["score"] = _coerce_nonfinite(injection.get("value", "NaN"))
            except (TypeError, ValueError) as exc:
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"invalid nonfinite injection value: {exc}",
                        channel=channel,
                        candidate_id=str(candidate_id),
                        injection_id=injection_id,
                    )
                )
                continue
            target["injection_id"] = injection_id
        else:
            duplicate = copy.deepcopy(target)
            duplicate["line_number"] = target["line_number"]
            duplicate["injection_id"] = injection_id
            entries.insert(target_index + 1, duplicate)
        hits.append(injection_hit(injection, channel=channel))

    channels_state: dict[str, Any] = {}
    for channel in config.channel_names:
        entry_list = raw_entries_by_channel[channel]
        candidates: dict[str, Any] = {}
        candidate_order: list[str] = []
        for entry in entry_list:
            data = entry["data"]
            line_number = entry["line_number"]
            fragment_id = entry["fragment_id"]
            injection_id = entry.get("injection_id")
            actual_channel = _raw_channel(data, channel)
            candidate_id = data.get("candidate_id")
            context = f"{fragment_id}:{line_number}"
            if actual_channel != channel:
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"channel mismatch at {context}: expected {channel!r}, got {actual_channel!r}",
                        channel=channel,
                        candidate_id=candidate_id if isinstance(candidate_id, str) else None,
                        injection_id=injection_id,
                    )
                )
                continue
            if not isinstance(candidate_id, str) or not candidate_id:
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"invalid candidate_id at {context}",
                        channel=channel,
                        injection_id=injection_id,
                    )
                )
                continue
            if candidate_id in candidates:
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"duplicate candidate_id {candidate_id!r} in channel {channel!r} at {context}",
                        channel=channel,
                        candidate_id=candidate_id,
                        injection_id=injection_id,
                    )
                )
                continue

            score = data.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"score is not numeric at {context}",
                        channel=channel,
                        candidate_id=candidate_id,
                        injection_id=injection_id,
                    )
                )
                continue
            score = float(score)
            if not math.isfinite(score):
                failures.append(
                    failure(
                        "load",
                        "E_NONFINITE_SCORE",
                        f"non-finite score {score!r} at {context}",
                        channel=channel,
                        candidate_id=candidate_id,
                        injection_id=injection_id,
                    )
                )
                continue

            features = data.get("features")
            if not isinstance(features, list):
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"features must be a list at {context}",
                        channel=channel,
                        candidate_id=candidate_id,
                        injection_id=injection_id,
                    )
                )
                continue
            if len(features) != config.feature_dim:
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"feature dimension mismatch at {context}: expected {config.feature_dim}, got {len(features)}",
                        channel=channel,
                        candidate_id=candidate_id,
                        injection_id=injection_id,
                    )
                )
                continue
            normalized_features: list[float] = []
            feature_invalid = False
            for feature_index, feature in enumerate(features):
                if not isinstance(feature, (int, float)) or isinstance(feature, bool):
                    failures.append(
                        failure(
                            "load",
                            "E_CONFIG_INVALID",
                            f"non-numeric feature #{feature_index} at {context}",
                            channel=channel,
                            candidate_id=candidate_id,
                            injection_id=injection_id,
                        )
                    )
                    feature_invalid = True
                    break
                feature_float = float(feature)
                if not math.isfinite(feature_float):
                    failures.append(
                        failure(
                            "load",
                            "E_CONFIG_INVALID",
                            f"non-finite feature #{feature_index} at {context}",
                            channel=channel,
                            candidate_id=candidate_id,
                            injection_id=injection_id,
                        )
                    )
                    feature_invalid = True
                    break
                normalized_features.append(feature_float)
            if feature_invalid:
                continue

            if len(candidates) >= config.channels[channel].capacity:
                failures.append(
                    failure(
                        "load",
                        "E_CONFIG_INVALID",
                        f"channel capacity exceeded for {channel!r} at {context}: {config.channels[channel].capacity}",
                        channel=channel,
                        candidate_id=candidate_id,
                        injection_id=injection_id,
                    )
                )
                continue

            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "channel": channel,
                "features": normalized_features,
                "fragment_id": fragment_id,
                "fragment_sha256": entry["fragment_sha256"],
                "score": score,
            }
            candidate_order.append(candidate_id)

        channels_state[channel] = {
            "candidate_ids": sorted(candidate_order),
            "candidates": {candidate_id: candidates[candidate_id] for candidate_id in sorted(candidates)},
            "fragment_set_hash": fragment_set_hash(fragments_by_channel[channel]),
            "fragments": sorted(fragments_by_channel[channel], key=lambda item: item["id"]),
        }

    state = {
        "channels": channels_state,
        "parameters": {"feature_dim": config.feature_dim},
    }
    return state, sort_failures(failures), sorted(hits, key=lambda item: str(item.get("id")))
