"""确定性整数定点前向计算（NumPy）。

所有参与输出摘要的运算都是整数运算：
  * 输入向量由固定种子 + 请求标识哈希派生的 :class:`numpy.random.RandomState`
    生成，元素为 uint32，并把请求定点值写入首元素；
  * 权重为 int8，形状由种子与参数版本号派生，参数版本变化会改变权重；
  * 线性层做矩阵乘后定点右移，激活用整数 clip + ReLU；
  * 累积用 int64，平台相关的有符号溢出先转 uint64 再回 int64，
    保证不同机器上逐字节一致。
"""

import hashlib
import json

import numpy as np

SHIFT1 = 8
SHIFT2 = 6


def _wrap_i64(array):
    """把 int64 运算结果按模 2^64 回绕，消除有符号溢出 UB 差异。"""
    return array.astype(np.uint64).view(np.int64)


def build_input_vector(seed, request_id, request_value, input_dim):
    rng_seed = (int(seed) * 1_000_003) ^ int.from_bytes(
        hashlib.blake2b(request_id.encode("utf-8"), digest_size=8).digest(), "little"
    )
    rng = np.random.RandomState(rng_seed & 0xFFFFFFFF)
    vector = rng.randint(0, 2**32, size=int(input_dim), dtype=np.uint32).copy()
    vector[0] = np.uint32(int(request_value) & 0xFFFFFFFF)
    return vector


def build_weights(seed, param_version, input_dim, hidden_dim):
    """参数版本不同 -> 权重不同；同版本任意进程重建结果一致。"""
    weight_seed = (int(seed) * 7919 + int(param_version) * 104729) & 0xFFFFFFFF
    rng = np.random.RandomState(weight_seed)
    w1 = rng.randint(-128, 128, size=(int(input_dim), int(hidden_dim)), dtype=np.int16)
    w2 = rng.randint(-128, 128, size=int(hidden_dim),),
    w2 = w2[0].astype(np.int16)
    return w1.astype(np.int64), w2.astype(np.int64)


def _fixed_relu(value, bits):
    clipped = np.clip(value, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return np.maximum(clipped, 0).astype(np.int64)


def forward_batch(batch, seed, param_version, input_dim, hidden_dim):
    """对批内请求做定点前向，返回 request_id -> 输出向量(list[int])。"""
    w1, w2 = build_weights(seed, param_version, input_dim, hidden_dim)
    inputs = np.stack(
        [
            build_input_vector(seed, request_id, value, input_dim)
            for request_id, value in batch
        ]
    ).astype(np.int64)

    hidden = _wrap_i64(inputs @ w1) >> SHIFT1
    hidden = _fixed_relu(hidden, 32)

    output = _wrap_i64(hidden @ w2.reshape(-1, 1)).reshape(-1) >> SHIFT2
    output = _fixed_relu(output, 32).astype(np.uint32)

    result = {}
    for (request_id, _), vector in zip(batch, output):
        result[request_id] = int(vector)
    return result


def output_digest(seed, param_version, request_id, request_value, output_scalar):
    """单个请求输出摘要：规范化 JSON 的 BLAKE2b 十六进制。"""
    payload = {
        "seed": int(seed),
        "param_version": int(param_version),
        "request_id": request_id,
        "value": int(request_value),
        "output": int(output_scalar),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=16).hexdigest()


def compute_request_digests(batch, seed, param_version, input_dim, hidden_dim):
    """逐请求计算输出标量与输出摘要，返回 request_id -> dict。"""
    outputs = forward_batch(batch, seed, param_version, input_dim, hidden_dim)
    result = {}
    for request_id, value in batch:
        scalar = outputs[request_id]
        result[request_id] = {
            "output": scalar,
            "digest": output_digest(seed, param_version, request_id, value, scalar),
        }
    return result


def batch_summary(request_digests):
    """批级输出摘要：请求标识排序后的逐请求摘要串联做 BLAKE2b。"""
    joined = "|".join(
        f"{request_id}:{request_digests[request_id]['digest']}"
        for request_id in sorted(request_digests)
    )
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=16).hexdigest()
