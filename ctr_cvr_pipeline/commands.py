from __future__ import annotations

import json
import math
from typing import Any

from .config import FeatureConfig
from .errors import ConfigError, Failure
from .fixed import ALLOWED_FRACTIONAL_BITS, ALLOWED_TIERS, TIER_BITS
from .samples import validate_label_update, validate_sample_record
from .state import PipelineState


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _command_failure(index: int, field: str, message: str, *, code: str = "E_COMMAND_INVALID") -> Failure:
    return Failure(
        code=code,
        message=message,
        field=field,
        command_index=index,
    )


def load_commands(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("E_CONFIG_INVALID", "commands", f"cannot load JSON: {exc}") from exc
    if isinstance(raw, dict):
        raw = raw.get("commands")
    if not isinstance(raw, list):
        raise ConfigError("E_CONFIG_INVALID", "commands", "must be an array or {commands: [...]} object")
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError("E_CONFIG_INVALID", f"commands[{index}]", "must be an object")
    return raw


def _command_event_time(state: PipelineState, command: dict[str, Any], index: int) -> tuple[int | None, Failure | None]:
    value = command.get("event_time_ms")
    if value is None:
        return state.current_event_time_ms, None
    if not _is_int(value) or value < 0:
        return None, _command_failure(index, "event_time_ms", "must be a non-negative integer")
    if value < state.current_event_time_ms:
        return None, _command_failure(
            index,
            "event_time_ms",
            f"training event time moved backwards from {state.current_event_time_ms} to {value}",
            code="E_EVENT_TIME_INVALID",
        )
    return int(value), None


def _validate_edges(feature: str, edges: Any, bins: int, index: int) -> tuple[tuple[float, ...] | None, Failure | None]:
    if not isinstance(edges, list):
        return None, _command_failure(index, "edges", "must be an array or null")
    if len(edges) != bins + 1:
        return None, _command_failure(
            index,
            "edges",
            f"feature {feature} requires exactly {bins + 1} edges",
        )
    parsed: list[float] = []
    for edge_index, value in enumerate(edges):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            return None, _command_failure(index, f"edges[{edge_index}]", "must be finite number")
        parsed.append(float(value))
    for left, right in zip(parsed, parsed[1:]):
        if right <= left:
            return None, _command_failure(index, "edges", "boundaries must be strictly increasing")
    return tuple(parsed), None


def apply_command(
    state: PipelineState,
    command: dict[str, Any],
    index: int,
    config: FeatureConfig,
) -> None:
    command_type = command.get("type")
    event_time, event_failure = _command_event_time(state, command, index)
    if event_failure is not None:
        state.failures.append(event_failure)
        state.applied_commands.append(
            {"command_index": index, "type": command_type, "status": "REJECTED"}
        )
        return
    assert event_time is not None

    if command_type == "append_batch":
        batch_id = command.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            state.failures.append(_command_failure(index, "batch_id", "must be a non-empty string"))
            return
        if batch_id in state.batch_event_times:
            state.failures.append(_command_failure(index, "batch_id", "batch_id already exists"))
            return
        raw_samples = command.get("samples")
        if not isinstance(raw_samples, list):
            state.failures.append(_command_failure(index, "samples", "must be an array"))
            return
        seen = set(state.samples)
        added = 0
        for sample_index, raw in enumerate(raw_samples):
            sample, failures = validate_sample_record(
                raw,
                config,
                event_time_ms=event_time,
                batch_id=batch_id,
                seen_keys=seen,
                command_index=index,
            )
            if failures:
                state.failures.extend(failures)
            if sample is not None:
                state.samples[sample.key] = sample
                seen.add(sample.key)
                added += 1
        state.batch_event_times[batch_id] = event_time
        state.current_event_time_ms = event_time
        state.applied_commands.append(
            {
                "command_index": index,
                "type": command_type,
                "status": "APPLIED",
                "batch_id": batch_id,
                "event_time_ms": event_time,
                "accepted_samples": added,
                "submitted_samples": len(raw_samples),
            }
        )
        return

    if command_type == "correct_label":
        request_id = command.get("request_id")
        item_id = command.get("item_id")
        key = (request_id, item_id)
        sample = state.samples.get(key)
        if sample is None:
            failure = _command_failure(
                index,
                "item_id",
                "request_id/item_id pair does not exist",
                code="E_SAMPLE_NOT_FOUND",
            )
            failure = Failure(
                code=failure.code,
                message=failure.message,
                request_id=str(request_id) if request_id is not None else None,
                item_id=str(item_id) if item_id is not None else None,
                sample_id=f"{request_id}:{item_id}",
                field=failure.field,
                command_index=index,
            )
            state.failures.append(failure)
            return
        candidates = command.get("labels")
        if isinstance(candidates, dict):
            click = candidates.get("click", command.get("click", sample.click))
            conversion = candidates.get("conversion", command.get("conversion", sample.conversion))
            delay = candidates.get(
                "conversion_delay_ms",
                command.get("conversion_delay_ms", sample.conversion_delay_ms),
            )
        else:
            click = command.get("click", sample.click)
            conversion = command.get("conversion", sample.conversion)
            delay = command.get("conversion_delay_ms", sample.conversion_delay_ms)
        failures = validate_label_update(
            request_id,
            item_id,
            click=click,
            conversion=conversion,
            conversion_delay_ms=delay,
            command_index=index,
        )
        if failures:
            state.failures.extend(failures)
            return
        state.samples[key] = sample.with_labels(
            click=int(click), conversion=int(conversion), conversion_delay_ms=delay
        )
        state.current_event_time_ms = event_time
        state.applied_commands.append(
            {
                "command_index": index,
                "type": command_type,
                "status": "APPLIED",
                "request_id": request_id,
                "item_id": item_id,
            }
        )
        return

    if command_type == "modify_bin_edges":
        feature_name = command.get("feature")
        feature = next((item for item in config.features if item.name == feature_name), None)
        if feature is None or feature.kind != "dense":
            state.failures.append(
                _command_failure(index, "feature", "must reference a configured dense feature")
            )
            return
        edges_value = command.get("edges")
        if edges_value is None:
            state.explicit_edges.pop(feature.name, None)
            edge_metadata: list[float] | str = "AUTO"
        else:
            assert feature.bins is not None
            edges, edge_failure = _validate_edges(feature.name, edges_value, feature.bins, index)
            if edge_failure is not None:
                state.failures.append(edge_failure)
                return
            assert edges is not None
            state.explicit_edges[feature.name] = edges
            edge_metadata = list(edges)
        state.current_event_time_ms = event_time
        state.applied_commands.append(
            {
                "command_index": index,
                "type": command_type,
                "status": "APPLIED",
                "feature": feature.name,
                "edges": edge_metadata,
            }
        )
        return

    if command_type in ("switch_quantization", "switch_quantization_tier"):
        tier = command.get("tier", state.quantization_tier)
        bits = command.get("fractional_bits", state.fixed_fractional_bits)
        if tier not in ALLOWED_TIERS:
            state.failures.append(
                _command_failure(index, "tier", f"must be one of {ALLOWED_TIERS}")
            )
            return
        if not _is_int(bits) or bits not in ALLOWED_FRACTIONAL_BITS:
            state.failures.append(
                _command_failure(index, "fractional_bits", f"must be one of {ALLOWED_FRACTIONAL_BITS}")
            )
            return
        if bits > TIER_BITS[tier] - 2:
            state.failures.append(
                _command_failure(index, "fractional_bits", "does not fit inside selected tier")
            )
            return
        state.quantization_tier = str(tier)
        state.fixed_fractional_bits = int(bits)
        state.current_event_time_ms = event_time
        state.applied_commands.append(
            {
                "command_index": index,
                "type": "switch_quantization",
                "status": "APPLIED",
                "tier": state.quantization_tier,
                "fractional_bits": state.fixed_fractional_bits,
            }
        )
        return

    if command_type in ("set_conversion_cutoff", "adjust_conversion_cutoff"):
        cutoff = command.get("cutoff_ms", command.get("conversion_delay_cutoff_ms"))
        if not _is_int(cutoff) or cutoff < 0:
            state.failures.append(
                _command_failure(index, "cutoff_ms", "must be a non-negative integer")
            )
            return
        state.conversion_delay_cutoff_ms = int(cutoff)
        state.current_event_time_ms = event_time
        state.applied_commands.append(
            {
                "command_index": index,
                "type": "set_conversion_cutoff",
                "status": "APPLIED",
                "cutoff_ms": state.conversion_delay_cutoff_ms,
            }
        )
        return

    state.failures.append(_command_failure(index, "type", f"unknown command type: {command_type!r}"))
