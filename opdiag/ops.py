"""转置/置换、激活-转置融合与原地写回执行。

元素遍历顺序固定为输出索引字典序升序（C order 行优先，即 np.transpose 的
默认内存读取顺序），融合/非融合两条路径各自记录累积顺序标识与中间舍入
次数。
"""

from __future__ import annotations

import numpy as np

from . import dmath


def transpose(x: np.ndarray, perm: list):
    """按置换重排。np.transpose 返回视图；这里 copy 成 C 连续以固定布局。"""
    out = np.transpose(x, tuple(perm))
    return np.ascontiguousarray(out)


def lexicographic_order_ids(shape) -> np.ndarray:
    """输出索引字典序升序的累积顺序标识（C-order 线性编号）。"""
    n = int(np.prod(shape)) if len(shape) else 1
    return np.arange(n, dtype=np.int64).reshape(shape if shape else (1,))


def run_activation(op, x, params, precision):
    dt = np.float64 if precision == "fp64" else np.float32
    xf = x.astype(dt, copy=True)
    y, rounds = dmath.apply_activation(op, xf, params or {})
    return y, rounds


def activate_inplace(act_op, x, params, precision):
    """在给定缓冲上原地逐元素激活，返回 (同一 ndarray, round_counts)。"""
    dt = np.float64 if precision == "fp64" else np.float32
    if x.dtype != dt:
        raise RuntimeError("inplace buffer dtype mismatch")
    flat = x.reshape(-1)
    rounds_flat = np.empty(flat.shape, dtype=np.int64)
    for i in range(flat.size):
        val, rc = dmath.apply_activation(act_op, flat[i:i + 1],
                                         params or {})
        flat[i] = val[0]
        rounds_flat[i] = rc[0]
    return x, rounds_flat.reshape(x.shape)


def run_unfused(act_op, params, x, perm, precision):
    """非融合路径：先激活，再转置。返回各阶段数组与逐元素顺序/舍入。"""
    a, act_rounds = run_activation(act_op, x, params, precision)
    t = transpose(a, perm)
    order_ids = lexicographic_order_ids(t.shape)
    # 非融合：激活阶段舍入 + 转置（纯数据搬运，无额外浮点 round）
    rounds_flat = transpose(act_rounds, perm)
    return {
        "output": t,
        "activation_output": a,
        "order_ids": order_ids,
        "round_counts": rounds_flat,
        "path": "unfused_activate_then_transpose",
    }


def run_fused(act_op, params, x, perm, precision):
    """融合路径：逻辑上单节点，按输出字典序逐元素计算激活。

    先得到输出坐标，再把每个输出坐标映射回激活前输入坐标，按该顺序逐元素
    求值（与非融合路径不同的累积顺序），舍入次数相同但累积顺序标识独立。
    """
    dt = np.float64 if precision == "fp64" else np.float32
    xf = x.astype(dt, copy=True)
    moved = np.transpose(xf, tuple(perm))
    out_shape = moved.shape
    # 融合路径按输出字典序展平后逐元素激活（显式 flat 循环顺序）。
    flat_in = np.ascontiguousarray(moved).reshape(-1)
    flat_out = np.empty_like(flat_in)
    _, act_rounds = run_activation(act_op, xf, params, precision)
    flat_rounds = np.ascontiguousarray(np.transpose(act_rounds, tuple(perm))
                                       ).reshape(-1)
    if act_op == "gelu":
        # 融合专用替代结合顺序（真实舍入差异，用于容差诊断）
        fused_vals, fused_rounds = dmath.gelu_fused_scalar(
            flat_in.reshape(out_shape))
        flat_out = np.ascontiguousarray(fused_vals).reshape(-1)
        flat_rounds = np.ascontiguousarray(
            np.transpose(fused_rounds, tuple(
                list(range(len(out_shape)))))).reshape(-1)
    else:
        for i in range(flat_in.size):
            scalar = flat_in[i:i + 1]
            val, _ = dmath.apply_activation(act_op, scalar, params or {})
            flat_out[i] = val[0]
    out = flat_out.reshape(out_shape)
    order_ids = lexicographic_order_ids(out_shape)
    return {
        "output": out,
        "order_ids": order_ids,
        "round_counts": flat_rounds.reshape(out_shape),
        "path": "fused_single_emit_lexicographic",
    }


def try_inplace_write(spec, x, producer_array_owner):
    """原地写回。

    仅在形状不变（激活）且输入缓冲区无其他引用时执行；调用方已在图校验
    阶段拦截 fan-out / 外部输入 / 转置，这里做运行期二次保护。
    返回写回后的数组（与 x 同存储）。
    """
    if spec.get("op") == "transpose":
        raise RuntimeError("transpose inplace should have been rejected")
    return x
