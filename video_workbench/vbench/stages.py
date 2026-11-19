"""多阶段渲染管线的纯函数阶段实现。

设计原则
--------
1. 所有阶段都是纯函数：相同输入字节级产生相同输出，不依赖全局可变 RNG。
2. 所有浮点运算使用 float64；随机数通过由 (seed, 阶段, 步/帧) 派生的
   独立 ``numpy.random.Generator`` 获取，线程之间不共享 RNG 状态。
3. 文生视频与图生视频在替代步之后调用完全相同的 ``denoise_step``，
   保证去噪计算与噪声引导语义一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import (
    E_GUIDANCE_INVALID,
    E_PATCH_OUT_OF_RANGE,
    WorkbenchError,
)
from .hashing import fingerprint

# ---- 固定模型常量（本地确定性，无需训练） ---------------------------------
VOCAB_HASH_SIZE = 128
EMBED_DIM = 16
LATENT_CHANNELS = 4
# 潜变量空间相对像素空间的下采样倍率（每轴）。
LATENT_SCALE = 2
RGB_CHANNELS = 3
RESOLUTION_TIERS = {
    "low": (16, 16),
    "mid": (24, 24),
}


def _rng(seed: int, *labels: Any) -> np.random.Generator:
    tag = "|".join(str(label) for label in labels)
    tag_int = int(fingerprint(tag)[:16], 16)
    derived = np.random.SeedSequence([int(seed), tag_int])
    return np.random.default_rng(derived)


# ---- 提示词编码：受限词表 + 哈希词袋嵌入 -----------------------------------
# 受限词表的词向量是模型常量，只由词表索引决定，与用户种子解耦，
# 保证相同提示词在任意种子/编辑序列下编码字节级一致。
EMBED_MODEL_SEED = 730121


def encode_prompt(prompt: str, seed: int) -> dict[str, Any]:
    tokens = prompt.lower().split()
    embedding = np.zeros(EMBED_DIM, dtype=np.float64)
    hit_tokens: list[str] = []
    oov_tokens: list[str] = []
    for token in tokens:
        index = _vocab_index(token)
        vector_rng = _rng(EMBED_MODEL_SEED, "embed", index)
        vector = vector_rng.standard_normal(EMBED_DIM)
        embedding += vector
        if index < VOCAB_HASH_SIZE // 2:
            hit_tokens.append(token)
        else:
            oov_tokens.append(token)
    if tokens:
        embedding = embedding / math.sqrt(len(tokens))
    return {
        "tokens": tokens,
        "in_vocab": hit_tokens,
        "out_of_vocab": oov_tokens,
        "embedding": embedding.astype(np.float64),
    }


def _vocab_index(token: str) -> int:
    digest = fingerprint(token)
    return int(digest[:8], 16) % VOCAB_HASH_SIZE


# ---- 初始图编码 -------------------------------------------------------------
def encode_init_image(image: np.ndarray, seed: int) -> dict[str, Any]:
    """把 RGB 像素图（H,W,3, 0-255）编码为潜变量形状的图像潜变量。"""
    normalized = image.astype(np.float64) / 255.0
    height, width = normalized.shape[0], normalized.shape[1]
    latent_h = height // LATENT_SCALE
    latent_w = width // LATENT_SCALE
    cropped = normalized[: latent_h * LATENT_SCALE, : latent_w * LATENT_SCALE, :]
    pooled = cropped.reshape(latent_h, LATENT_SCALE, latent_w, LATENT_SCALE, RGB_CHANNELS)
    pooled = pooled.mean(axis=(1, 3))  # (lh, lw, 3)
    latent = np.zeros((latent_h, latent_w, LATENT_CHANNELS), dtype=np.float64)
    latent[:, :, :RGB_CHANNELS] = pooled - 0.5
    mix_rng = _rng(seed, "image-encode")
    mix = mix_rng.standard_normal((RGB_CHANNELS, LATENT_CHANNELS - RGB_CHANNELS))
    latent[:, :, RGB_CHANNELS:] = pooled @ mix
    return {"image_latent": latent}


# ---- 初始潜变量生成（文生视频） ---------------------------------------------
def init_latent(prompt_fp: str, seed: int, height: int, width: int) -> dict[str, Any]:
    latent_h = height // LATENT_SCALE
    latent_w = width // LATENT_SCALE
    rng = _rng(seed, "latent-init", prompt_fp[:16])
    latent = rng.standard_normal((latent_h, latent_w, LATENT_CHANNELS))
    return {"latent": latent.astype(np.float64)}


# ---- 去噪步（两条路径共享的唯一实现） ---------------------------------------
def denoise_step(
    previous: np.ndarray,
    prompt_embedding: np.ndarray,
    step_index: int,
    total_steps: int,
    guidance: float,
    noise_entry: np.ndarray,
    seed: int,
) -> np.ndarray:
    """单步去噪：读取上一步潜变量，叠加预置噪声表与条件引导。"""
    if not np.isfinite(guidance):
        raise WorkbenchError(E_GUIDANCE_INVALID, f"引导强度非有限数: {guidance}")
    step_rng = _rng(seed, "denoise", step_index)
    jitter = step_rng.standard_normal(previous.shape) * float(noise_entry)
    scale = 1.0 / (1.0 + step_index)
    # 条件引导：提示词嵌入沿通道方向广播到整张潜变量图。
    condition = guidance * prompt_embedding.reshape(1, 1, EMBED_DIM)[
        :, :, : LATENT_CHANNELS
    ]
    condition = np.broadcast_to(condition, previous.shape).astype(np.float64)
    updated = previous + scale * (0.1 * previous + condition) + jitter
    # 固定步长衰减，保证 total_steps 变化时既有步序结果不变。
    updated = updated * (1.0 - 0.01 * step_index)
    return updated.astype(np.float64)


def apply_patch(
    latent: np.ndarray,
    patch: Mapping[str, Any],
    *,
    stage: str = "denoise",
    step: int | None = None,
) -> np.ndarray:
    """对某步去噪输出施加局部矩形区域潜变量替换。

    越界时抛出 ``E_PATCH_OUT_OF_RANGE``，调用方负责不写入任何状态。
    """
    top = int(patch["top"])
    left = int(patch["left"])
    values = np.asarray(patch["values"], dtype=np.float64)
    ph, pw, pc = values.shape
    height, width, channels = latent.shape
    if (
        top < 0
        or left < 0
        or top + ph > height
        or left + pw > width
        or pc != channels
    ):
        raise WorkbenchError(
            E_PATCH_OUT_OF_RANGE,
            (
                f"补丁区域 (top={top}, left={left}, size={ph}x{pw}x{pc}) "
                f"越界，潜变量尺寸为 {height}x{width}x{channels}"
            ),
            stage=stage,
            step=step,
        )
    patched = latent.copy()
    patched[top : top + ph, left : left + pw, :] = values
    return patched


# ---- 逐帧解码：线性投影 + 逐像素裁剪到 0-255 -------------------------------
def decode_frame(
    latent: np.ndarray,
    frame_index: int,
    frames: int,
    seed: int,
    height: int,
    width: int,
) -> np.ndarray:
    latent_h, latent_w, _ = latent.shape
    upscaled = np.repeat(np.repeat(latent, LATENT_SCALE, axis=0), LATENT_SCALE, axis=1)
    weight_rng = _rng(seed, "decode-weight")
    weight = weight_rng.standard_normal((LATENT_CHANNELS, RGB_CHANNELS))
    bias_rng = _rng(seed, "decode-bias", frame_index)
    bias = bias_rng.standard_normal(RGB_CHANNELS) * 0.05
    projected = upscaled @ weight + bias
    # 帧间确定性时序偏移：仅由帧序号派生，保证可复现与帧序稳定。
    temporal = (frame_index - (frames - 1) / 2.0) * 0.02
    pixels = 128.0 + 64.0 * (projected + temporal)
    pixels = np.clip(np.rint(pixels), 0.0, 255.0).astype(np.uint8)
    result = np.zeros((height, width, RGB_CHANNELS), dtype=np.uint8)
    result[: pixels.shape[0], : pixels.shape[1], :] = pixels[
        :height, :width, :
    ]
    return result


def array_summary(array: np.ndarray) -> dict[str, Any]:
    """潜变量/像素帧摘要：形状、dtype 与统计量，用于产物元数据。"""
    cast = array.astype(np.float64)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(cast.min()) if array.size else 0.0,
        "max": float(cast.max()) if array.size else 0.0,
        "mean": float(cast.mean()) if array.size else 0.0,
    }


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str  # encode | image_encode | latent_init | denoise | decode
    step: int | None = None
    frame: int | None = None

    @property
    def sort_key(self) -> tuple[str, int, int]:
        kind_order = {
            "encode": 0,
            "image_encode": 1,
            "latent_init": 2,
            "denoise": 3,
            "decode": 4,
        }[self.kind]
        return (
            f"{kind_order:02d}:{self.name}",
            self.step if self.step is not None else -1,
            self.frame if self.frame is not None else -1,
        )


def build_stage_specs(
    steps: int, frames: int, mode: str, replace_steps: int
) -> list[StageSpec]:
    """按固定执行顺序构造阶段清单，阶段间不得跳过或合并。"""
    specs: list[StageSpec] = [StageSpec("encode", "encode")]
    if mode == "i2v":
        specs.append(StageSpec("image_encode", "image_encode"))
    else:
        specs.append(StageSpec("latent_init", "latent_init"))
    for index in range(steps):
        specs.append(StageSpec(f"denoise_{index:03d}", "denoise", step=index))
    for index in range(frames):
        specs.append(StageSpec(f"decode_{index:03d}", "decode", frame=index))
    return specs


def replaced_denoise_steps(mode: str, replace_steps: int, steps: int) -> set[int]:
    if mode != "i2v":
        return set()
    return set(range(min(replace_steps, steps)))
