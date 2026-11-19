"""修复工程的参数化模型、schema 识别与 v1->v2 迁移。

阶段固定顺序：denoise -> gate -> normalize -> limit。
v1 只记录降噪参数；迁移到 v2 时其余阶段补“恒等默认值”，
保证迁移前后对同一素材的修复输出逐样本一致（哈希一致）。
"""
from __future__ import annotations

from typing import Any

from . import config
from .errors import (
    E_FIELD_MISSING,
    E_PARAM_RANGE,
    E_SCHEMA_VERSION,
    RepairError,
)
from .hashing import sha256_json

# ---- 参数范围契约 ----
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "denoise.over_subtraction": (0.0, 8.0),
    "denoise.floor": (0.0, 1.0),
    "denoise.noise_seconds": (0.01, 60.0),
    "gate.threshold_db": (-120.0, 0.0),
    "gate.ratio": (1.0, 100.0),
    "normalize.target_peak": (0.0, 1.0),
    "normalize.max_gain_db": (0.0, 60.0),
    "limit.ceiling": (0.0, 1.0),
}

# ---- v1 允许的降噪字段（schema_version=1 的全部契约）----
V1_DENOISE_FIELDS = {"alpha", "floor", "noise_seed", "noise_seconds"}

# ---- 固定迁移表：v1 缺字段 -> v2 默认值 ----
MIGRATION_DEFAULTS: dict[str, Any] = {
    "gate": {"threshold_db": -120.0, "ratio": 1.0},
    "normalize": {"target_peak": 1.0, "max_gain_db": 0.0},
    "limit": {"ceiling": 1.0},
    "channel_order": config.ORDER_MONO_FIRST,
    "bit_depth_policy": {"output_bit_depth": config.DEFAULT_OUTPUT_BIT_DEPTH, "mode": "force16"},
}


def default_v2_project(project_id: str = "project") -> dict[str, Any]:
    return {
        "schema_version": config.SCHEMA_V2,
        "project_id": project_id,
        "engine_version": config.ENGINE_VERSION,
        "stft": {"window": config.STFT_WINDOW, "nfft": config.STFT_NFFT, "hop": config.STFT_HOP},
        "denoise": {
            "over_subtraction": 2.0,
            "floor": 0.02,
            "noise_seed": 7,
            "noise_seconds": 0.5,
        },
        "gate": {"threshold_db": -45.0, "ratio": 10.0},
        "normalize": {"target_peak": 0.89, "max_gain_db": 12.0},
        "limit": {"ceiling": 0.95},
        "channel_order": config.ORDER_MONO_FIRST,
        "bit_depth_policy": {
            "output_bit_depth": config.DEFAULT_OUTPUT_BIT_DEPTH,
            "mode": "force16",
        },
    }


def legacy_v1_project(project_id: str = "legacy") -> dict[str, Any]:
    """构造一个典型的旧版工程（仅降噪参数）。"""
    return {
        "schema_version": config.SCHEMA_V1,
        "project_id": project_id,
        "denoise": {
            "alpha": 2.0,
            "floor": 0.02,
            "noise_seed": 7,
            "noise_seconds": 0.5,
        },
    }


def detect_schema_version(data: dict[str, Any]) -> int:
    if "schema_version" not in data:
        raise RepairError(
            E_FIELD_MISSING,
            "project JSON missing schema_version",
            field="schema_version",
        )
    version = data["schema_version"]
    if version not in (config.SCHEMA_V1, config.SCHEMA_V2):
        raise RepairError(
            E_SCHEMA_VERSION,
            f"unsupported schema_version={version}",
            field="schema_version",
            context={"schema_version": version},
        )
    return int(version)


def _check_ranges(params: dict[str, Any]) -> None:
    flat: dict[str, float] = {}
    for stage in ("denoise", "gate", "normalize", "limit"):
        stage_params = params.get(stage, {})
        for key, value in stage_params.items():
            flat[f"{stage}.{key}"] = value
    for name, value in flat.items():
        if name in PARAM_RANGES:
            low, high = PARAM_RANGES[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise RepairError(
                    E_PARAM_RANGE,
                    f"{name} must be numeric, got {type(value).__name__}",
                    field=name,
                )
            if not (low <= float(value) <= high):
                raise RepairError(
                    E_PARAM_RANGE,
                    f"{name}={value} out of range [{low}, {high}]",
                    field=name,
                    context={"value": value, "range": [low, high]},
                )


def _validate_v2(data: dict[str, Any]) -> None:
    required = ("denoise", "gate", "normalize", "limit", "channel_order", "bit_depth_policy")
    for field in required:
        if field not in data:
            raise RepairError(
                E_FIELD_MISSING,
                f"v2 project missing required field: {field}",
                field=field,
            )
    if data["channel_order"] not in config.CHANNEL_ORDERS:
        raise RepairError(
            E_PARAM_RANGE,
            f"channel_order must be one of {config.CHANNEL_ORDERS}",
            field="channel_order",
            context={"value": data["channel_order"]},
        )
    policy = data["bit_depth_policy"]
    if policy.get("output_bit_depth") not in config.SUPPORTED_BIT_DEPTHS:
        raise RepairError(
            E_PARAM_RANGE,
            "bit_depth_policy.output_bit_depth must be 16 or 24",
            field="bit_depth_policy.output_bit_depth",
            context={"value": policy.get("output_bit_depth")},
        )
    denoise = data["denoise"]
    for key in ("over_subtraction", "floor", "noise_seed", "noise_seconds"):
        if key not in denoise:
            raise RepairError(
                E_FIELD_MISSING, f"denoise missing field: {key}", field=f"denoise.{key}"
            )
    _check_ranges(data)


def migrate_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """按固定迁移表迁移；幂等；不可迁移字段抛 E_SCHEMA_VERSION 并指明字段名。"""
    version = detect_schema_version(data)
    if version == config.SCHEMA_V2:
        _validate_v2(data)
        return dict(data)

    if "denoise" not in data:
        raise RepairError(
            E_FIELD_MISSING,
            "v1 project must contain denoise parameters",
            field="denoise",
        )
    legacy = data["denoise"]
    unknown = set(legacy) - V1_DENOISE_FIELDS
    if unknown:
        bad = sorted(unknown)[0]
        raise RepairError(
            E_SCHEMA_VERSION,
            f"v1 field 'denoise.{bad}' has no migration rule and cannot be migrated",
            field=f"denoise.{bad}",
            context={"unknown_fields": sorted(unknown)},
        )
    for key in V1_DENOISE_FIELDS:
        if key not in legacy:
            raise RepairError(
                E_FIELD_MISSING,
                f"v1 denoise missing field: {key}",
                field=f"denoise.{key}",
            )

    # v1 的 alpha 在 v2 中命名为 over_subtraction，语义与数值不变。
    denoise_v2 = {
        "over_subtraction": legacy["alpha"],
        "floor": legacy["floor"],
        "noise_seed": legacy["noise_seed"],
        "noise_seconds": legacy["noise_seconds"],
    }
    migrated = {
        "schema_version": config.SCHEMA_V2,
        "project_id": data.get("project_id", "migrated_project"),
        "engine_version": config.ENGINE_VERSION,
        "stft": {
            "window": config.STFT_WINDOW,
            "nfft": config.STFT_NFFT,
            "hop": config.STFT_HOP,
        },
        "denoise": denoise_v2,
        "gate": dict(MIGRATION_DEFAULTS["gate"]),
        "normalize": dict(MIGRATION_DEFAULTS["normalize"]),
        "limit": dict(MIGRATION_DEFAULTS["limit"]),
        "channel_order": MIGRATION_DEFAULTS["channel_order"],
        "bit_depth_policy": dict(MIGRATION_DEFAULTS["bit_depth_policy"]),
    }
    _validate_v2(migrated)
    return migrated


def load_and_normalize(data: dict[str, Any]) -> dict[str, Any]:
    """识别版本并返回校验过的 v2 工程（不修改入参）。"""
    version = detect_schema_version(data)
    if version == config.SCHEMA_V1:
        return migrate_v1_to_v2(data)
    normalized = dict(data)
    _validate_v2(normalized)
    return normalized


def project_fingerprint(project_v2: dict[str, Any]) -> str:
    """工程参数指纹（不含易变 id），用于缓存键与失效判定。"""
    payload = {
        "schema_version": project_v2["schema_version"],
        "engine_version": config.ENGINE_VERSION,
        "stft": project_v2["stft"],
        "denoise": project_v2["denoise"],
        "gate": project_v2["gate"],
        "normalize": project_v2["normalize"],
        "limit": project_v2["limit"],
        "channel_order": project_v2["channel_order"],
        "bit_depth_policy": project_v2["bit_depth_policy"],
    }
    return sha256_json(payload)
