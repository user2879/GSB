"""配置加载、参数/结构注入校验与交互命令应用。"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Mapping

import numpy as np

from .errors import (
    E_CONFIG_INVALID,
    E_PATCH_OUT_OF_RANGE,
    E_GUIDANCE_INVALID,
    E_INIT_IMAGE_INVALID,
    E_NOISE_TABLE_MISMATCH,
    E_SEED_OUT_OF_RANGE,
    E_STEP_ZERO,
    WorkbenchError,
)
from .stages import LATENT_CHANNELS, LATENT_SCALE, RESOLUTION_TIERS

SEED_MIN = 0
SEED_MAX = 2**32 - 1
REQUIRED_FIELDS = (
    "prompt",
    "seed",
    "steps",
    "guidance",
    "frames",
    "resolution",
    "mode",
    "max_workers",
    "noise_table",
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(value: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)


def apply_command(config: Mapping[str, Any], command: Mapping[str, Any]) -> dict[str, Any]:
    """把单条交互命令应用到配置，返回新配置（不修改入参）。"""
    updated = copy.deepcopy(dict(config))
    operation = command.get("op")
    if operation == "set_prompt":
        updated["prompt"] = _require_str(command, "value")
    elif operation == "set_seed":
        updated["seed"] = command["value"]
    elif operation == "set_steps":
        updated["steps"] = command["value"]
    elif operation == "set_guidance":
        updated["guidance"] = command["value"]
    elif operation == "set_frames":
        updated["frames"] = command["value"]
    elif operation == "set_init_image":
        updated.setdefault("i2v", {})["init_image"] = _require_str(command, "value")
    elif operation == "set_replace_steps":
        updated.setdefault("i2v", {})["replace_steps"] = command["value"]
    elif operation == "set_mode":
        updated["mode"] = _require_str(command, "value")
    elif operation == "set_patch":
        patches = list(updated.get("patches", []))
        target_step = int(command["step"])
        replacement = None
        for index, existing in enumerate(patches):
            if int(existing["step"]) == target_step:
                replacement = index
                break
        payload = {"step": target_step, "patch": command["patch"]}
        if replacement is None:
            patches.append(payload)
        else:
            patches[replacement] = payload
        updated["patches"] = sorted(patches, key=lambda item: int(item["step"]))
    elif operation == "clear_patch":
        target_step = int(command["step"])
        updated["patches"] = [
            item
            for item in updated.get("patches", [])
            if int(item["step"]) != target_step
        ]
    else:
        raise WorkbenchError(
            E_CONFIG_INVALID, f"未知交互命令: {operation!r}"
        )
    return updated


def _require_str(command: Mapping[str, Any], key: str) -> str:
    value = command.get(key)
    if not isinstance(value, str):
        raise WorkbenchError(
            E_CONFIG_INVALID, f"命令 {command.get('op')} 的 {key} 必须是字符串"
        )
    return value


def validate_config(raw: Mapping[str, Any], base_dir: str) -> dict[str, Any]:
    """完整参数与结构校验，返回归一化后的运行配置。

    校验顺序固定：结构/类型 -> 种子 -> 引导 -> 步数 -> 噪声表 -> 图生视频。
    """
    if not isinstance(raw, Mapping):
        raise WorkbenchError(E_CONFIG_INVALID, "配置根节点必须是 JSON 对象")

    for field in REQUIRED_FIELDS:
        if field not in raw:
            raise WorkbenchError(E_CONFIG_INVALID, f"配置缺少必填字段: {field}")

    if not isinstance(raw["prompt"], str) or not raw["prompt"].strip():
        raise WorkbenchError(E_CONFIG_INVALID, "prompt 必须是非空字符串")

    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise WorkbenchError(E_CONFIG_INVALID, "seed 必须是整数")
    if not (SEED_MIN <= seed <= SEED_MAX):
        raise WorkbenchError(
            E_SEED_OUT_OF_RANGE, f"seed={seed} 越界，允许区间 [{SEED_MIN}, {SEED_MAX}]"
        )

    steps = raw["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise WorkbenchError(E_CONFIG_INVALID, "steps 必须是整数")
    if steps == 0:
        raise WorkbenchError(E_STEP_ZERO, "steps 不能为 0")
    if steps < 0:
        raise WorkbenchError(E_CONFIG_INVALID, "steps 必须为正整数")

    guidance = raw["guidance"]
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)):
        raise WorkbenchError(E_CONFIG_INVALID, "guidance 必须是数值")
    if not np.isfinite(float(guidance)):
        raise WorkbenchError(E_GUIDANCE_INVALID, f"guidance 非有限数: {guidance}")

    frames = raw["frames"]
    if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
        raise WorkbenchError(E_CONFIG_INVALID, "frames 必须是正整数")

    resolution = raw["resolution"]
    if not isinstance(resolution, str) or resolution not in RESOLUTION_TIERS:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            f"resolution 必须是 {sorted(RESOLUTION_TIERS)} 之一",
        )
    height, width = RESOLUTION_TIERS[resolution]

    mode = raw["mode"]
    if mode not in ("t2v", "i2v"):
        raise WorkbenchError(E_CONFIG_INVALID, "mode 必须是 t2v 或 i2v")

    max_workers = raw["max_workers"]
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise WorkbenchError(E_CONFIG_INVALID, "max_workers 必须是 >=1 的整数")

    noise_path = _resolve(base_dir, raw["noise_table"])
    if not isinstance(raw["noise_table"], str):
        raise WorkbenchError(E_CONFIG_INVALID, "noise_table 必须是路径字符串")
    if not os.path.isfile(noise_path):
        raise WorkbenchError(
            E_NOISE_TABLE_MISMATCH, f"噪声表文件不存在: {noise_path}"
        )
    noise_table = np.load(noise_path).astype(np.float64)
    if noise_table.ndim != 1 or noise_table.shape[0] != steps:
        raise WorkbenchError(
            E_NOISE_TABLE_MISMATCH,
            f"噪声表长度 {noise_table.shape[0]} 与步数 {steps} 不符",
        )

    i2v = raw.get("i2v", {})
    if not isinstance(i2v, Mapping):
        raise WorkbenchError(E_CONFIG_INVALID, "i2v 必须是对象")
    replace_steps = int(i2v.get("replace_steps", 0))
    if isinstance(i2v.get("replace_steps", 0), bool) or replace_steps < 0:
        raise WorkbenchError(E_CONFIG_INVALID, "i2v.replace_steps 必须是非负整数")
    if replace_steps > steps:
        raise WorkbenchError(
            E_CONFIG_INVALID,
            f"i2v.replace_steps={replace_steps} 不能超过 steps={steps}",
        )

    init_image = None
    if mode == "i2v":
        image_path = i2v.get("init_image")
        if not isinstance(image_path, str) or not image_path:
            raise WorkbenchError(
                E_INIT_IMAGE_INVALID, "i2v 模式必须提供 init_image 路径"
            )
        resolved = _resolve(base_dir, image_path)
        if not os.path.isfile(resolved):
            raise WorkbenchError(
                E_INIT_IMAGE_INVALID, f"初始图缺失: {resolved}", stage="image_encode"
            )
        init_image = np.load(resolved)
        if init_image.ndim != 3 or init_image.shape[2] != 3:
            raise WorkbenchError(
                E_INIT_IMAGE_INVALID,
                f"初始图必须是 HxWx3 数组，实际 shape={list(init_image.shape)}",
                stage="image_encode",
            )
        if (init_image.shape[0], init_image.shape[1]) != (height, width):
            raise WorkbenchError(
                E_INIT_IMAGE_INVALID,
                (
                    f"初始图尺寸 {list(init_image.shape[:2])} 与分辨率档位 "
                    f"{resolution} 要求的 [{height}, {width}] 不符"
                ),
                stage="image_encode",
            )
        finite = np.isfinite(init_image.astype(np.float64)).all()
        if not finite:
            raise WorkbenchError(
                E_INIT_IMAGE_INVALID, "初始图包含非有限数值", stage="image_encode"
            )

    patches = _validate_patches(raw.get("patches", []))
    _validate_patch_bounds(patches, height, width)

    return {
        "prompt": raw["prompt"],
        "seed": int(seed),
        "steps": int(steps),
        "guidance": float(guidance),
        "frames": int(frames),
        "resolution": resolution,
        "height": height,
        "width": width,
        "mode": mode,
        "max_workers": int(max_workers),
        "noise_table_path": noise_path,
        "noise_table": noise_table,
        "replace_steps": replace_steps,
        "init_image": init_image,
        "init_image_path": i2v.get("init_image") if mode == "i2v" else None,
        "patches": patches,
    }


def _validate_patches(raw_patches: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_patches, list):
        raise WorkbenchError(E_CONFIG_INVALID, "patches 必须是数组")
    normalized: list[dict[str, Any]] = []
    for item in raw_patches:
        if not isinstance(item, Mapping) or "step" not in item or "patch" not in item:
            raise WorkbenchError(
                E_CONFIG_INVALID, "每个补丁需包含 step 与 patch 字段"
            )
        step = item["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise WorkbenchError(E_CONFIG_INVALID, "补丁 step 必须是非负整数")
        patch = item["patch"]
        if not isinstance(patch, Mapping):
            raise WorkbenchError(E_CONFIG_INVALID, "补丁 patch 必须是对象")
        for field in ("top", "left", "values"):
            if field not in patch:
                raise WorkbenchError(E_CONFIG_INVALID, f"补丁缺少字段 {field}")
        if not isinstance(patch["top"], int) or not isinstance(patch["left"], int):
            raise WorkbenchError(E_CONFIG_INVALID, "补丁 top/left 必须是整数")
        values = np.asarray(patch["values"], dtype=np.float64)
        if values.ndim != 3:
            raise WorkbenchError(E_CONFIG_INVALID, "补丁 values 必须是三维数组")
        normalized.append(
            {
                "step": int(step),
                "patch": {
                    "top": int(patch["top"]),
                    "left": int(patch["left"]),
                    "values": values,
                },
            }
        )
    normalized.sort(key=lambda entry: entry["step"])
    return normalized


def _validate_patch_bounds(
    patches: list[dict[str, Any]], height: int, width: int
) -> None:
    """写入前的补丁边界预校验：越界直接失败且不产生任何新状态。"""
    latent_h = height // LATENT_SCALE
    latent_w = width // LATENT_SCALE
    channels = LATENT_CHANNELS
    for item in patches:
        patch = item["patch"]
        top = patch["top"]
        left = patch["left"]
        values = np.asarray(patch["values"], dtype=np.float64)
        ph, pw, pc = values.shape
        step = item["step"]
        if (
            top < 0
            or left < 0
            or top + ph > latent_h
            or left + pw > latent_w
            or pc != channels
        ):
            raise WorkbenchError(
                E_PATCH_OUT_OF_RANGE,
                (
                    f"step {step} 的补丁区域 (top={top}, left={left}, "
                    f"size={ph}x{pw}x{pc}) 越界，潜变量尺寸为 "
                    f"{latent_h}x{latent_w}x{channels}"
                ),
                stage=f"denoise_{step:03d}",
                step=step,
            )


def _resolve(base_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def param_fingerprint_inputs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """用于参数指纹的 JSON 兼容视图（路径与数组内容都参与）。"""
    patches_view = [
        {
            "step": item["step"],
            "patch": {
                "top": item["patch"]["top"],
                "left": item["patch"]["left"],
                "values": np.asarray(item["patch"]["values"], dtype=np.float64),
            },
        }
        for item in cfg.get("patches", [])
    ]
    view = {
        "prompt": cfg["prompt"],
        "seed": cfg["seed"],
        "steps": cfg["steps"],
        "guidance": cfg["guidance"],
        "frames": cfg["frames"],
        "resolution": cfg["resolution"],
        "mode": cfg["mode"],
        "replace_steps": cfg.get("replace_steps", 0),
        "noise_table": cfg.get("noise_table"),
        "patches": patches_view,
    }
    if cfg.get("mode") == "i2v":
        view["init_image"] = cfg.get("init_image")
        view["init_image_path"] = cfg.get("init_image_path")
    return view
