"""Incremental fixed-point renderer and deterministic batch scheduler."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .errors import BizError
from .fixed import CHANNEL_ORDER, quantize_fixed, scale_for
from .hashing import channel_key, composite_key, content_key, layer_key, layer_parameter_key
from .material import (
    base_channel_arrays,
    build_channel_artifact,
    channel_summary,
    compose_channels,
    mix_channel,
)
from .models import (
    ChannelArtifact,
    CompositeArtifact,
    LayerArtifact,
    LayerPartial,
    MaterialConfig,
    RenderConfig,
    RenderState,
    TextureData,
)
from .noise import compute_layer_chunk, layer_fingerprint
from .scheduler import BatchTaskResult, run_ordered_tasks


@dataclass
class RenderBatchOutcome:
    summary: dict[str, Any]
    max_workers_used: int
    max_active_tasks: int
    affected_layers: list[int] = field(default_factory=list)
    ledger_rows: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)


class InjectionRegistry:
    """Thread-safe registry that raises real failures at render stages."""

    def __init__(self, injections: list[dict[str, Any]]) -> None:
        self.injections = injections
        self.hits: list[dict[str, Any]] = []
        self._hit_indices: set[int] = set()
        self._lock = threading.Lock()

    def _matches(
        self,
        injection: dict[str, Any],
        stage: str,
        layer: int | None,
        channel: str | None,
        referenced_layers: set[int] | None,
    ) -> bool:
        if injection.get("stage") != stage:
            return False
        inj_layer = injection.get("layer")
        if inj_layer is not None:
            if layer is None or int(inj_layer) != int(layer):
                if referenced_layers is None or int(inj_layer) not in referenced_layers:
                    return False
        inj_channel = injection.get("channel")
        if inj_channel is not None and inj_channel != channel:
            return False
        return True

    def trigger(
        self,
        stage: str,
        *,
        layer: int | None = None,
        channel: str | None = None,
        referenced_layers: set[int] | None = None,
    ) -> None:
        for index, injection in enumerate(self.injections):
            if not self._matches(injection, stage, layer, channel, referenced_layers):
                continue
            with self._lock:
                if index not in self._hit_indices:
                    self._hit_indices.add(index)
                    self.hits.append(
                        {
                            "injection_index": index,
                            "stage": stage,
                            "layer": layer if layer is not None else injection.get("layer"),
                            "channel": channel if channel is not None else injection.get("channel"),
                            "code": injection["code"],
                        }
                    )
            if injection.get("delay_ms", 0):
                time.sleep(int(injection["delay_ms"]) / 1000.0)
            raise BizError(
                code=injection["code"],
                message=injection.get("message", f"injected failure at {stage}"),
                stage=stage,
                layer=layer,
                channel=channel,
                injected=True,
            )


def _failure_from_error(
    error: BaseException,
    *,
    product: str,
    layer: int | None = None,
    channel: str | None = None,
    fallback_stage: str,
) -> dict[str, Any]:
    if isinstance(error, BizError):
        result = error.to_dict()
        result.setdefault("product", product)
        if result.get("layer") is None and layer is not None:
            result["layer"] = layer
        if result.get("channel") is None and channel is not None:
            result["channel"] = channel
        return result
    return {
        "code": "E_RENDER",
        "message": f"{type(error).__name__}: {error}",
        "stage": fallback_stage,
        "product": product,
        "layer": layer,
        "channel": channel,
        "injected": False,
    }


def _failure_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    product_rank = {"layer": 0, "channel": 1, "composite": 2}
    stage_rank = {
        "noise_synthesis": 0,
        "layer_quantize": 1,
        "material_mix": 2,
        "channel_compose": 3,
    }
    channel_rank = {name: index for index, name in enumerate(CHANNEL_ORDER)}
    layer = item.get("layer")
    channel = item.get("channel")
    product = str(item.get("product"))
    stage = stage_rank.get(str(item.get("stage")), 9)
    if product == "layer":
        return (0, -1 if layer is None else int(layer), 0, stage, str(item.get("code", "")), str(item.get("message", "")))
    if product == "channel":
        return (
            1,
            channel_rank.get(str(channel), 99),
            -1 if layer is None else int(layer),
            stage,
            str(item.get("code", "")),
            str(item.get("message", "")),
        )
    return (
        product_rank.get(str(item.get("product")), 9),
        -1,
        0,
        stage,
        str(item.get("code", "")),
        str(item.get("message", "")),
    )


def _pending_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    product_rank = {"layer": 0, "channel": 1, "composite": 2}
    channel_rank = {name: index for index, name in enumerate(CHANNEL_ORDER)}
    return (
        product_rank.get(str(item.get("product")), 9),
        int(item.get("index", -1)) if item.get("index") is not None else -1,
        channel_rank.get(str(item.get("channel")), 99),
        str(item.get("code", "")),
    )


def _layer_invalidation_reason(old: LayerArtifact | LayerPartial, meta: dict[str, Any]) -> str:
    if old.texture_key != meta["texture_key"]:
        return "base_texture_changed"
    if old.parameter_key != meta["parameter_key"]:
        return "layer_parameter_changed"
    if old.quant_step != meta["quant_step"]:
        return "quant_step_changed"
    if old.frac_bits != meta["frac_bits"]:
        return "fraction_bits_changed"
    if old.key != meta["key"]:
        return "cache_key_changed"
    return "cache_miss"


def _channel_parameter_key(channel: Any, config: RenderConfig, material: MaterialConfig) -> str:
    return content_key(
        "channel-parameters",
        {
            "channel": channel.parameter_dict(),
            "tie_rule": material.tie_rule,
            "equal_weight_rule": material.equal_weight_rule,
            "blend_mode": config.blend_mode,
            "fraction_bits": config.fixed.fraction_bits,
            "quant_step": config.fixed.quant_step,
            "resolution": config.resolution,
        },
    )


def _channel_invalidation_reason(
    old: ChannelArtifact,
    expected_layer_keys: tuple[str, ...],
    quant_step: int,
    blend_mode: str,
    parameter_key: str,
    expected_key: str,
) -> str:
    if old.layer_keys != expected_layer_keys:
        return "upstream_layer_changed"
    if old.quant_step != quant_step:
        return "quant_step_changed"
    if old.blend_mode != blend_mode:
        return "blend_mode_changed"
    if old.parameter_key != parameter_key:
        return "channel_weight_changed"
    if old.key != expected_key:
        return "cache_key_changed"
    return "cache_miss"


def _composite_invalidation_reason(
    old: CompositeArtifact,
    expected_channels: tuple[str, ...],
    config: RenderConfig,
    expected_key: str,
) -> str:
    if old.channel_keys != expected_channels:
        return "upstream_channel_changed"
    if old.quant_step != config.fixed.quant_step:
        return "quant_step_changed"
    if old.frac_bits != config.fixed.fraction_bits:
        return "fraction_bits_changed"
    if old.key != expected_key:
        return "cache_key_changed"
    return "cache_miss"


def _ledger_row(
    *,
    sequence: int,
    product: str,
    action: str,
    key: str,
    reason: str,
    quant_step: int,
    fraction_bits: int,
    index: int | None = None,
    channel: str | None = None,
    references: dict[str, Any] | None = None,
    points: int | None = None,
    points_done: int | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sequence": sequence,
        "product": product,
        "action": action,
        "key": key,
        "reason": reason,
        "quant_step": quant_step,
        "fraction_bits": fraction_bits,
    }
    if index is not None:
        row["index"] = index
    if channel is not None:
        row["channel"] = channel
    if references is not None:
        row["references"] = references
    if points is not None:
        row["points"] = points
    if points_done is not None:
        row["points_done"] = points_done
    if code is not None:
        row["code"] = code
    return row


def _finalize_layers(
    state: RenderState,
    index: int,
    meta: dict[str, Any],
) -> LayerArtifact:
    partial = state.partials.pop(index)
    artifact = LayerArtifact(
        index=index,
        key=meta["key"],
        texture_key=meta["texture_key"],
        parameter_key=meta["parameter_key"],
        frac_bits=meta["frac_bits"],
        quant_step=meta["quant_step"],
        raw=partial.raw,
        quantized=partial.quantized,
        residual=partial.residual,
        decisions=partial.decisions,
        complete=True,
    )
    state.layers[index] = artifact
    return artifact


def render_batch(
    state: RenderState,
    config: RenderConfig,
    material: MaterialConfig,
    texture: TextureData,
    *,
    sequence: int,
    injections: list[dict[str, Any]] | None = None,
) -> RenderBatchOutcome:
    """Render at most one configured point-budget batch into ``state``."""

    registry = InjectionRegistry(injections or [])
    bits = config.fixed.fraction_bits
    quant_step = config.fixed.quant_step
    scale = scale_for(bits)
    width, height = texture.width, texture.height
    total_points = width * height
    ledger: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    affected: set[int] = set()

    texture_row = _ledger_row(
        sequence=sequence,
        product="texture",
        action="reuse_hit" if state.last_texture_key == texture.content_key else "load",
        key=texture.content_key,
        reason="content_key_match" if state.last_texture_key == texture.content_key else "texture_loaded",
        quant_step=quant_step,
        fraction_bits=bits,
        references={"file_sha256": texture.file_sha256, "path": texture.path},
    )
    ledger.append(texture_row)

    layer_meta: dict[int, dict[str, Any]] = {}
    for layer in sorted(config.layers, key=lambda item: item.index):
        fingerprint = layer_fingerprint(layer, config, texture)
        parameter_key = layer_parameter_key(layer, config, texture)
        layer_meta[layer.index] = {
            "key": layer_key(layer, config, texture),
            "parameter_key": parameter_key,
            "fingerprint": fingerprint,
            "texture_key": texture.content_key,
            "frac_bits": bits,
            "quant_step": quant_step,
            "spec": layer,
        }
    expected_layer_keys = {index: meta["key"] for index, meta in layer_meta.items()}

    for index in sorted(set(state.layers) - set(layer_meta)):
        del state.layers[index]
    for index in sorted(set(state.partials) - set(layer_meta)):
        del state.partials[index]

    plans: list[dict[str, Any]] = []
    batch_remaining = config.budgets.batch_points
    for index in sorted(layer_meta):
        meta = layer_meta[index]
        layer = meta["spec"]
        old_complete = state.layers.get(index)
        old_partial = state.partials.get(index)
        references = {
            "base_texture_key": texture.content_key,
            "base_texture_sha256": texture.file_sha256,
            "parameter_key": meta["parameter_key"],
            "fingerprint": meta["fingerprint"],
        }
        if old_complete is not None and old_complete.key == meta["key"]:
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="layer",
                    index=index,
                    action="reuse_hit",
                    key=meta["key"],
                    reason="cache_key_match",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=references,
                    points=0,
                    points_done=total_points,
                )
            )
            continue

        reason = "cache_miss"
        if old_complete is not None:
            reason = _layer_invalidation_reason(old_complete, meta)
            del state.layers[index]
            old_partial = None
        if old_partial is not None and old_partial.key != meta["key"]:
            reason = _layer_invalidation_reason(old_partial, meta)
            del state.partials[index]
            old_partial = None

        done = old_partial.points_done if old_partial is not None else 0
        remaining = total_points - done
        allocation = min(remaining, config.budgets.layer_points, batch_remaining)
        if allocation <= 0:
            pending.append(
                {
                    "product": "layer",
                    "index": index,
                    "points_done": done,
                    "points_total": total_points,
                    "code": "E_POINT_BUDGET_PENDING",
                    "reason": "per_layer_or_batch_budget_exhausted",
                }
            )
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="layer",
                    index=index,
                    action="pending",
                    key=meta["key"],
                    reason=reason,
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=references,
                    points=remaining,
                    points_done=done,
                    code="E_POINT_BUDGET_PENDING",
                )
            )
            affected.add(index)
            continue

        if old_partial is None:
            old_partial = LayerPartial(
                index=index,
                key=meta["key"],
                texture_key=texture.content_key,
                parameter_key=meta["parameter_key"],
                frac_bits=bits,
                quant_step=quant_step,
                raw=np.zeros(total_points, dtype=np.int64),
                quantized=np.zeros(total_points, dtype=np.int64),
                residual=np.zeros(total_points, dtype=np.int64),
                decisions=np.zeros(total_points, dtype=np.int8),
                points_done=0,
            )
            state.partials[index] = old_partial
            reason = reason if reason != "cache_miss" else "cache_miss"

        plan = {
            "index": index,
            "meta": meta,
            "layer": layer,
            "partial": old_partial,
            "start": done,
            "end": done + allocation,
            "reason": reason,
            "references": references,
            "resumed": done > 0,
        }
        plans.append(plan)
        batch_remaining -= allocation
        affected.add(index)
        if allocation < remaining:
            pending.append(
                {
                    "product": "layer",
                    "index": index,
                    "points_done": done + allocation,
                    "points_total": total_points,
                    "code": "E_POINT_BUDGET_PENDING",
                    "reason": "per_layer_or_batch_budget_exhausted",
                }
            )

    def layer_task(plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        layer = plan["layer"]
        meta = plan["meta"]
        if _matches_layer_injection(registry, "noise_synthesis", plan):
            registry.trigger("noise_synthesis", layer=layer.index)
        raw = compute_layer_chunk(
            layer,
            config,
            texture,
            plan["start"],
            plan["end"],
            frac_bits=bits,
            fingerprint=meta["fingerprint"],
        )
        if _matches_layer_injection(registry, "layer_quantize", plan):
            registry.trigger("layer_quantize", layer=layer.index)
        quantized, residual, decisions = quantize_fixed(raw, quant_step, config.fixed.tie_rule)
        return raw, quantized, residual, decisions

    task_result = run_ordered_tasks(
        [lambda plan=plan: layer_task(plan) for plan in plans],
        config.thread_pool_size,
    )
    max_workers_used = task_result.max_workers_used
    max_active_tasks = task_result.max_active_tasks

    for plan, result in zip(plans, task_result.results):
        index = plan["index"]
        meta = plan["meta"]
        references = plan["references"]
        done_before = plan["start"]
        allocation = plan["end"] - plan["start"]
        if result.error is not None:
            state.partials.pop(index, None)
            failure = _failure_from_error(
                result.error,
                product="layer",
                layer=index,
                fallback_stage="noise_synthesis",
            )
            failures.append(failure)
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="layer",
                    index=index,
                    action="failure",
                    key=meta["key"],
                    reason="stage_exception",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=references,
                    points=allocation,
                    points_done=done_before,
                    code=str(failure["code"]),
                )
            )
            continue

        raw, quantized, residual, decisions = result.value
        partial: LayerPartial = plan["partial"]
        start, end = plan["start"], plan["end"]
        partial.raw[start:end] = raw
        partial.quantized[start:end] = quantized
        partial.residual[start:end] = residual
        partial.decisions[start:end] = decisions
        partial.points_done = end
        action = "recompute"
        if plan["resumed"]:
            action = "resume_partial"
        if end >= total_points:
            _finalize_layers(state, index, meta)
            action = "recompute_complete"
        ledger.append(
            _ledger_row(
                sequence=sequence,
                product="layer",
                index=index,
                action=action,
                key=meta["key"],
                reason=plan["reason"],
                quant_step=quant_step,
                fraction_bits=bits,
                references=references,
                points=allocation,
                points_done=end,
            )
        )

    failed_layers = {int(item["layer"]) for item in failures if item.get("layer") is not None}
    incomplete_layers = {
        index for index in layer_meta if index not in state.layers and index not in failed_layers
    }
    for index in sorted(incomplete_layers):
        if not any(item.get("product") == "layer" and item.get("index") == index for item in pending):
            partial = state.partials.get(index)
            pending.append(
                {
                    "product": "layer",
                    "index": index,
                    "points_done": partial.points_done if partial is not None else 0,
                    "points_total": total_points,
                    "code": "E_POINT_BUDGET_PENDING",
                    "reason": "awaiting_next_batch",
                }
            )

    base_channels = base_channel_arrays(texture, config, bits)
    channel_meta: dict[str, dict[str, Any]] = {}
    for channel in material.ordered_channels():
        expected_ref_keys = tuple(
            expected_layer_keys[index]
            for index in channel.layer_indices
            if index in expected_layer_keys
        )
        expected_key = channel_key(channel, config, material, texture, expected_layer_keys)
        param_key = _channel_parameter_key(channel, config, material)
        channel_meta[channel.name] = {
            "spec": channel,
            "key": expected_key,
            "parameter_key": param_key,
            "layer_keys": expected_ref_keys,
        }

    channel_plans: list[dict[str, Any]] = []
    for channel in material.ordered_channels():
        meta = channel_meta[channel.name]
        old = state.channels.get(channel.name)
        references = {
            "base_texture_key": texture.content_key,
            "layer_indices": list(channel.layer_indices),
            "layer_keys": list(meta["layer_keys"]),
            "parameter_key": meta["parameter_key"],
        }
        if any(index in failed_layers for index in channel.layer_indices):
            state.channels.pop(channel.name, None)
            failure = {
                "code": "E_CHANNEL_DEPENDENCY",
                "message": "channel depends on a failed layer",
                "stage": "material_mix",
                "product": "channel",
                "channel": channel.name,
                "layer": next(index for index in channel.layer_indices if index in failed_layers),
                "injected": False,
            }
            failures.append(failure)
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="channel",
                    channel=channel.name,
                    action="failure",
                    key=meta["key"],
                    reason="upstream_layer_failed",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=references,
                    code="E_CHANNEL_DEPENDENCY",
                )
            )
            continue
        if any(index not in state.layers for index in channel.layer_indices):
            state.channels.pop(channel.name, None)
            pending.append(
                {
                    "product": "channel",
                    "channel": channel.name,
                    "points_done": 0,
                    "points_total": total_points,
                    "code": "E_POINT_BUDGET_PENDING",
                    "reason": "upstream_layers_pending",
                }
            )
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="channel",
                    channel=channel.name,
                    action="pending",
                    key=meta["key"],
                    reason="upstream_layers_pending",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=references,
                    code="E_POINT_BUDGET_PENDING",
                )
            )
            continue
        if old is not None and old.key == meta["key"]:
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="channel",
                    channel=channel.name,
                    action="reuse_hit",
                    key=meta["key"],
                    reason="cache_key_match",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=references,
                )
            )
            continue
        reason = "cache_miss"
        if old is not None:
            reason = _channel_invalidation_reason(
                old,
                meta["layer_keys"],
                quant_step,
                config.blend_mode,
                meta["parameter_key"],
                meta["key"],
            )
        channel_plans.append(
            {
                "channel": channel,
                "meta": meta,
                "reason": reason,
                "references": references,
            }
        )

    def channel_task(plan: dict[str, Any]) -> tuple[Any, str]:
        channel = plan["channel"]
        referenced = {
            index: state.layers[index].quantized.reshape(height, width)
            for index in channel.layer_indices
        }
        for injection in registry.injections:
            if injection.get("stage") != "material_mix":
                continue
            inj_layer = injection.get("layer")
            inj_channel = injection.get("channel")
            if inj_channel is not None and inj_channel != channel.name:
                continue
            if inj_layer is not None and int(inj_layer) not in channel.layer_indices:
                continue
            registry.trigger(
                "material_mix",
                layer=int(inj_layer) if inj_layer is not None else None,
                channel=channel.name,
                referenced_layers=set(channel.layer_indices),
            )
            break
        result = mix_channel(
            channel,
            base_channels[channel.name],
            referenced,
            config,
            material,
            bits,
        )
        return result, plan["meta"]["parameter_key"]

    channel_task_result = run_ordered_tasks(
        [lambda plan=plan: channel_task(plan) for plan in channel_plans],
        config.thread_pool_size,
    )
    max_workers_used = max(max_workers_used, channel_task_result.max_workers_used)
    max_active_tasks = max(max_active_tasks, channel_task_result.max_active_tasks)
    for plan, result in zip(channel_plans, channel_task_result.results):
        channel = plan["channel"]
        meta = plan["meta"]
        if result.error is not None:
            state.channels.pop(channel.name, None)
            failure = _failure_from_error(
                result.error,
                product="channel",
                channel=channel.name,
                fallback_stage="material_mix",
            )
            failures.append(failure)
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="channel",
                    channel=channel.name,
                    action="failure",
                    key=meta["key"],
                    reason="stage_exception",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=plan["references"],
                    code=str(failure["code"]),
                )
            )
            continue
        mix, parameter_key = result.value
        artifact = build_channel_artifact(
            channel,
            mix,
            meta["key"],
            expected_layer_keys,
            config,
            parameter_key=parameter_key,
        )
        state.channels[channel.name] = artifact
        ledger.append(
            _ledger_row(
                sequence=sequence,
                product="channel",
                channel=channel.name,
                action="recompute",
                key=meta["key"],
                reason=plan["reason"],
                quant_step=quant_step,
                fraction_bits=bits,
                references=plan["references"],
            )
        )

    expected_channel_keys = {
        name: meta["key"] for name, meta in channel_meta.items() if name in material.channels
    }
    ordered_channel_names = [channel.name for channel in material.ordered_channels()]
    ordered_channel_keys = tuple(expected_channel_keys.get(name, "") for name in ordered_channel_names)
    expected_composite_key = composite_key(config, material, expected_channel_keys)
    channels_ready = all(
        name in state.channels and state.channels[name].key == expected_channel_keys.get(name)
        for name in ordered_channel_names
    )
    final_references = {
        "channel_keys": {name: expected_channel_keys.get(name, "") for name in ordered_channel_names},
        "parameter_key": content_key(
            "composite-parameters",
            {"quant_step": quant_step, "fraction_bits": bits, "resolution": config.resolution},
        ),
    }
    if not channels_ready:
        state.composite = None
        dependency_failed = any(
            item.get("product") == "channel" and item.get("stage") == "material_mix"
            for item in failures
        )
        if dependency_failed:
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="composite",
                    action="failure",
                    key=expected_composite_key,
                    reason="upstream_channel_failed",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=final_references,
                    code="E_CHANNEL_DEPENDENCY",
                )
            )
        else:
            pending.append(
                {
                    "product": "composite",
                    "points_done": 0,
                    "points_total": total_points,
                    "code": "E_POINT_BUDGET_PENDING",
                    "reason": "upstream_channels_pending",
                }
            )
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="composite",
                    action="pending",
                    key=expected_composite_key,
                    reason="upstream_channels_pending",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=final_references,
                    code="E_POINT_BUDGET_PENDING",
                )
            )
    elif state.composite is not None and state.composite.key == expected_composite_key:
        ledger.append(
            _ledger_row(
                sequence=sequence,
                product="composite",
                action="reuse_hit",
                key=expected_composite_key,
                reason="cache_key_match",
                quant_step=quant_step,
                fraction_bits=bits,
                references=final_references,
            )
        )
    else:
        reason = "cache_miss"
        if state.composite is not None:
            reason = _composite_invalidation_reason(
                state.composite,
                ordered_channel_keys,
                config,
                expected_composite_key,
            )

        def compose_task() -> tuple[np.ndarray, str]:
            for injection in registry.injections:
                if injection.get("stage") != "channel_compose":
                    continue
                inj_channel = injection.get("channel")
                inj_layer = injection.get("layer")
                if inj_channel is not None and inj_channel not in state.channels:
                    continue
                if inj_layer is not None and not any(
                    int(inj_layer) in state.channels[name].layer_indices for name in state.channels
                ):
                    continue
                registry.trigger(
                    "channel_compose",
                    layer=int(inj_layer) if inj_layer is not None else None,
                    channel=str(inj_channel) if inj_channel is not None else None,
                )
                break
            arrays = {name: state.channels[name].array for name in ordered_channel_names}
            result = compose_channels(arrays, config, bits)
            param_key = content_key(
                "composite-parameters",
                {"quant_step": quant_step, "fraction_bits": bits, "resolution": config.resolution},
            )
            return result, param_key

        compose_result = run_ordered_tasks([compose_task], config.thread_pool_size)
        max_workers_used = max(max_workers_used, compose_result.max_workers_used)
        max_active_tasks = max(max_active_tasks, compose_result.max_active_tasks)
        result = compose_result.results[0]
        state.composite = None
        if result.error is not None:
            failure = _failure_from_error(
                result.error,
                product="composite",
                fallback_stage="channel_compose",
            )
            failures.append(failure)
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="composite",
                    action="failure",
                    key=expected_composite_key,
                    reason="stage_exception",
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=final_references,
                    code=str(failure["code"]),
                )
            )
        else:
            composite_array, parameter_key = result.value
            state.composite = CompositeArtifact(
                key=expected_composite_key,
                channel_keys=ordered_channel_keys,
                array=composite_array,
                frac_bits=bits,
                quant_step=quant_step,
                parameter_key=parameter_key,
            )
            ledger.append(
                _ledger_row(
                    sequence=sequence,
                    product="composite",
                    action="recompute",
                    key=expected_composite_key,
                    reason=reason,
                    quant_step=quant_step,
                    fraction_bits=bits,
                    references=final_references,
                )
            )

    pending.sort(key=_pending_sort_key)
    failures.sort(key=_failure_sort_key)
    state.pending = pending
    state.failures = failures
    state.ledger.extend(ledger)
    if registry.hits:
        registry.hits.sort(
            key=lambda item: (
                int(item.get("injection_index", 0)),
                str(item.get("stage", "")),
                -1 if item.get("layer") is None else int(item["layer"]),
                str(item.get("channel", "")),
            )
        )
        existing = {item.get("injection_index") for item in state.injection_hits}
        for hit in registry.hits:
            if hit["injection_index"] not in existing:
                state.injection_hits.append(hit)
        state.injection_hits.sort(
            key=lambda item: (
                int(item.get("injection_index", 0)),
                str(item.get("stage", "")),
                -1 if item.get("layer") is None else int(item["layer"]),
                str(item.get("channel", "")),
            )
        )
    state.last_texture_key = texture.content_key
    state.last_sequence = sequence
    state.effective_config_hash = content_key("effective-config", config.to_dict())

    completed_layers = sorted(state.layers)
    completed_points = total_points * len(completed_layers)
    for index, partial in state.partials.items():
        completed_points += partial.points_done
    total_layer_points = max(1, total_points * len(config.layers))
    channels_complete = sum(
        1
        for name in ordered_channel_names
        if name in state.channels and state.channels[name].key == expected_channel_keys.get(name)
    )
    progressive_units = (
        (completed_points * 700) // total_layer_points
        + (channels_complete * 200) // max(1, len(ordered_channel_names))
        + (100 if state.composite is not None else 0)
    )
    codes = sorted(
        {str(item.get("code")) for item in pending + failures if item.get("code") is not None}
    )
    summary: dict[str, Any] = {
        "sequence": sequence,
        "stale": False,
        "complete": state.composite is not None,
        "completed_layers": completed_layers,
        "completed_layer_count": len(completed_layers),
        "total_layers": len(config.layers),
        "completed_points": completed_points,
        "total_points": total_layer_points,
        "progress_ppm": max(0, min(1_000_000, progressive_units * 1000)),
        "progress_permille": max(0, min(1000, progressive_units)),
        "affected_layers": sorted(affected),
        "channels": [
            channel_summary(state.channels[name])
            for name in ordered_channel_names
            if name in state.channels
        ],
        "pending": pending,
        "failures": failures,
        "injection_hits": list(state.injection_hits),
        "codes": codes,
        "max_workers_used": max_workers_used,
        "max_active_tasks": max_active_tasks,
        "thread_pool_size": config.thread_pool_size,
        "quant_step": quant_step,
        "fraction_bits": bits,
        "texture_key": texture.content_key,
    }
    state.last_preview = summary
    return RenderBatchOutcome(
        summary=summary,
        max_workers_used=max_workers_used,
        max_active_tasks=max_active_tasks,
        affected_layers=sorted(affected),
        ledger_rows=ledger,
        failures=failures,
        pending=pending,
    )


def _matches_layer_injection(registry: InjectionRegistry, stage: str, plan: dict[str, Any]) -> bool:
    for injection in registry.injections:
        if injection.get("stage") != stage:
            continue
        inj_layer = injection.get("layer")
        if inj_layer is None or int(inj_layer) == int(plan["index"]):
            return True
    return False


def full_render_serial(
    config: RenderConfig,
    material: MaterialConfig,
    texture: TextureData,
) -> tuple[dict[int, LayerArtifact], dict[str, ChannelArtifact], CompositeArtifact]:
    """Reference full rebuild used by equality checks and diagnostics."""

    bits = config.fixed.fraction_bits
    height, width = texture.height, texture.width
    layer_keys: dict[int, str] = {}
    layers: dict[int, LayerArtifact] = {}
    for layer in sorted(config.layers, key=lambda item: item.index):
        key = layer_key(layer, config, texture)
        parameter_key = layer_parameter_key(layer, config, texture)
        raw = compute_layer_chunk(
            layer,
            config,
            texture,
            0,
            width * height,
            frac_bits=bits,
            fingerprint=layer_fingerprint(layer, config, texture),
        )
        quantized, residual, decisions = quantize_fixed(raw, config.fixed.quant_step, config.fixed.tie_rule)
        layers[layer.index] = LayerArtifact(
            index=layer.index,
            key=key,
            texture_key=texture.content_key,
            parameter_key=parameter_key,
            frac_bits=bits,
            quant_step=config.fixed.quant_step,
            raw=raw,
            quantized=quantized,
            residual=residual,
            decisions=decisions,
        )
        layer_keys[layer.index] = key

    base_channels = base_channel_arrays(texture, config, bits)
    channels: dict[str, ChannelArtifact] = {}
    for channel in material.ordered_channels():
        referenced = {
            index: layers[index].quantized.reshape(height, width)
            for index in channel.layer_indices
        }
        mix = mix_channel(channel, base_channels[channel.name], referenced, config, material, bits)
        key = channel_key(channel, config, material, texture, layer_keys)
        channels[channel.name] = build_channel_artifact(
            channel,
            mix,
            key,
            layer_keys,
            config,
            parameter_key=_channel_parameter_key(channel, config, material),
        )
    ordered_names = [channel.name for channel in material.ordered_channels()]
    composite_array = compose_channels(
        {name: channels[name].array for name in ordered_names}, config, bits
    )
    channel_keys = {name: channels[name].key for name in ordered_names}
    composite = CompositeArtifact(
        key=composite_key(config, material, channel_keys),
        channel_keys=tuple(channel_keys[name] for name in ordered_names),
        array=composite_array,
        frac_bits=bits,
        quant_step=config.fixed.quant_step,
        parameter_key=content_key(
            "composite-parameters",
            {
                "quant_step": config.fixed.quant_step,
                "fraction_bits": bits,
                "resolution": config.resolution,
            },
        ),
    )
    return layers, channels, composite
