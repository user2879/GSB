"""float64 参考路径。

以 float64 无截断卷积作为真值：激活与权重在 int64 可精确表示范围内，
乘加用 float64 计算整数精确乘积之和；反量化真值直接对原始整数和
乘以缩放因子（不做缓冲饱和、量化截断与定点舍入），但仍裁剪到输出区间。
"""

from __future__ import annotations

from .kernel import split_position
from .models import OpDesc
from .tensorpack import Tensor, TensorPack


def reference_position(
    desc: OpDesc,
    pack: TensorPack,
    position: int,
    *,
    activation_override: Tensor | None = None,
) -> tuple[int, float, float]:
    """返回 (精确整数和, 未裁剪反量化真值, 裁剪到输出区间后的真值)。"""
    activation = activation_override if activation_override is not None else pack.get("activation")
    weight = pack.get("weight")
    batch_index, out_channel, out_row, out_col = split_position(desc, position)

    exact_sum = 0
    for in_channel in range(desc.in_channels):
        for kernel_row in range(desc.k_h):
            for kernel_col in range(desc.k_w):
                in_h = out_row * desc.stride_h + kernel_row - desc.pad_h
                in_w = out_col * desc.stride_w + kernel_col - desc.pad_w
                if in_h < 0 or in_w < 0 or in_h >= desc.in_h or in_w >= desc.in_w:
                    act_value = 0
                else:
                    act_value = activation.get(batch_index, in_channel, in_h, in_w)
                weight_value = weight.get(out_channel, in_channel, kernel_row, kernel_col)
                exact_sum += act_value * weight_value

    scale = desc.effective_scale(out_channel)
    raw_float = float(exact_sum) * scale
    clipped = min(max(raw_float, desc.out_low), desc.out_high)
    return exact_sum, raw_float, clipped
