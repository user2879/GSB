"""共享编码器前向表示与逐层贡献追踪。"""

import numpy as np

from .model import Encoder


def layer_arrays(encoder: Encoder):
    """按层序返回 [(W, b)]，W 形状 (out, in)。"""
    return [
        (
            np.asarray(layer.weight, dtype=np.float64),
            np.asarray(layer.bias, dtype=np.float64),
        )
        for layer in encoder.layers
    ]


def forward_all(encoder: Encoder, X: np.ndarray):
    """返回 [H0..HL]，H0 为标准化输入，Hi 为第 i 层输出。"""
    mean = np.asarray(encoder.norm_mean, dtype=np.float64)
    std = np.asarray(encoder.norm_std, dtype=np.float64)
    h = (np.asarray(X, dtype=np.float64) - mean) / std
    outs = [h]
    for W, b in layer_arrays(encoder):
        h = h @ W.T + b
        outs.append(h)
    return outs


def target_matrix(encoder: Encoder, X: np.ndarray, target_layers):
    """拼接目标层输出，返回 (Z, slices)。slices: {层名: (start, end)}。"""
    outs = forward_all(encoder, X)
    index = encoder.layer_index()
    ordered = sorted(target_layers, key=lambda name: index[name])
    blocks, slices, pos = [], {}, 0
    for name in ordered:
        block = outs[index[name] + 1]
        blocks.append(block)
        slices[name] = (pos, pos + block.shape[1])
        pos += block.shape[1]
    return np.concatenate(blocks, axis=1), slices


def contribution_shares(encoder, task, adapter_params, X, Y, sensitivities, loss_and_dz):
    """计算每层对任务适配器损失的贡献份额。

    固定运算顺序：链式梯度自上而下回传，逐层 Taylor 一阶项与贡献清单
    敏感度系数合成并归一化到总损失。仅返回每层每任务份额，不保留中间激活。
    """
    outs = forward_all(encoder, X)
    index = encoder.layer_index()
    ordered = sorted(task.target_layers, key=lambda name: index[name])
    blocks, slices, pos = [], {}, 0
    for name in ordered:
        block = outs[index[name] + 1]
        blocks.append(block)
        slices[name] = (pos, pos + block.shape[1])
        pos += block.shape[1]
    Z = np.concatenate(blocks, axis=1)
    loss, dZ = loss_and_dz(adapter_params, Z, Y)

    n_layers = len(encoder.layers)
    grads = [None] * (n_layers + 1)
    for name in ordered:
        li = index[name] + 1
        start, end = slices[name]
        grads[li] = dZ[:, start:end].copy()
    arrays = layer_arrays(encoder)
    for i in range(n_layers, 0, -1):
        if grads[i] is None:
            continue
        W, _b = arrays[i - 1]
        back = grads[i] @ W
        grads[i - 1] = back if grads[i - 1] is None else grads[i - 1] + back

    n = X.shape[0]
    taylor = []
    for i in range(1, n_layers + 1):
        if grads[i] is None:
            taylor.append(0.0)
        else:
            taylor.append(float(np.sum(grads[i] * outs[i]) / n))
    coeffs = []
    for i, layer in enumerate(encoder.layers):
        sens = sensitivities.get((layer.name, task.name), {"sensitivity": 0.0})
        coeffs.append(sens["sensitivity"] * taylor[i])
    denom = 0.0
    for c in coeffs:
        denom += c
    shares = {}
    if denom == 0.0:
        for layer in encoder.layers:
            shares[layer.name] = 0.0
    else:
        for layer, c in zip(encoder.layers, coeffs):
            shares[layer.name] = float(loss * c / denom)
    return shares, float(loss)


def normalized_shares(shares):
    total = 0.0
    for key in sorted(shares):
        total += shares[key]
    if total == 0.0:
        return {key: 0.0 for key in sorted(shares)}
    return {key: shares[key] / total for key in sorted(shares)}
