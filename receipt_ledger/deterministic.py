"""确定性参数矩阵、量化口径与 NumPy 前向推理。

显式规则（不依赖全局随机状态，因此与加载顺序、并发线程无关）：
  1. 每个 (worker_id, param_version) 使用独立 ``numpy.random.default_rng``，
     种子由 blake2b(worker_id|model_dim|hidden_dim|seed|param_version) 派生；
  2. 权重矩阵 W: hidden_dim x model_dim、偏置 b: hidden_dim，标准正态；
  3. 按量化口径执行显式量化，再转回 fp64 参与前向；
  4. 前向 h = quant(W) @ x + quant(b)，输出摘要为排序后字节的 sha256。
"""

import hashlib

import numpy as np

from .digest import sha256_hex


def _rng_seed(worker_id: str, model_dim: int, hidden_dim: int,
              declared_seed: int, param_version: int) -> int:
    material = "|".join(
        [worker_id, str(model_dim), str(hidden_dim),
         str(declared_seed), str(param_version)]
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")


def build_parameters(worker, param_version: int):
    """返回 (量化后权重, 量化后偏置, 参数摘要)。同声明逐位一致。"""
    seed = _rng_seed(
        worker.worker_id, worker.model_dim, worker.hidden_dim,
        worker.seed, param_version,
    )
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((worker.hidden_dim, worker.model_dim))
    bias = rng.standard_normal(worker.hidden_dim)
    q_weight, q_bias = quantize(weight, worker.quant), quantize(bias, worker.quant)
    summary = parameter_summary(worker, param_version, q_weight, q_bias)
    return q_weight, q_bias, summary


def quantize(array: np.ndarray, quant: str) -> np.ndarray:
    """显式量化规则。fp32 与 fp16 走 IEEE 舍入，int8 走对称整数量化。"""
    if quant == "fp32":
        return array.astype(np.float32).astype(np.float64)
    if quant == "fp16":
        return array.astype(np.float16).astype(np.float64)
    if quant == "int8":
        peak = float(np.max(np.abs(array)))
        if peak == 0.0:
            return np.zeros_like(array, dtype=np.float64)
        scale = peak / 127.0
        levels = np.rint(array / scale).clip(-127, 127).astype(np.int8)
        return levels.astype(np.float64) * scale
    raise ValueError(f"不支持的量化口径: {quant}")


def parameter_summary(worker, param_version: int,
                      q_weight: np.ndarray, q_bias: np.ndarray) -> str:
    """参数摘要绑定工作器标识、模型维度与参数版本序号。"""
    payload = {
        "worker_id": worker.worker_id,
        "model_dim": worker.model_dim,
        "hidden_dim": worker.hidden_dim,
        "param_version": param_version,
        "quant": worker.quant,
        "param_format": worker.param_format,
        "weight_sha256": sha256_hex(np.ascontiguousarray(q_weight).tobytes()),
        "bias_sha256": sha256_hex(np.ascontiguousarray(q_bias).tobytes()),
    }
    return sha256_hex(
        (";".join(f"{k}={payload[k]}" for k in sorted(payload))).encode("utf-8")
    )


def forward(weight: np.ndarray, bias: np.ndarray, vector) -> np.ndarray:
    """NumPy 前向：h = W x + b，全程 fp64 确定运算。"""
    x = np.asarray(vector, dtype=np.float64)
    return weight @ x + bias


def output_digest(hidden: np.ndarray) -> str:
    """输出摘要：对隐藏向量的连续字节求 sha256。"""
    contiguous = np.ascontiguousarray(hidden, dtype=np.float64)
    return sha256_hex(contiguous.tobytes())


def request_destination(client_key: str, worker_ids) -> str:
    """确定性去向：客户端键哈希对规则工作器集合取模，与字典顺序无关。"""
    bucket = int.from_bytes(
        hashlib.blake2b(client_key.encode("utf-8"), digest_size=8).digest(), "big"
    )
    ordered = tuple(sorted(worker_ids))
    return ordered[bucket % len(ordered)]

