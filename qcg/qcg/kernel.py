"""Integer-domain fixed-point 2D single-channel convolution."""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Config
from .errors import QCGError, E_ACC_OVERFLOW
from .models import Node

TRAVERSAL_ID = "row_major:oh,ow,kh,kw"
KERNEL_IMPL_VERSION = "conv-int-1.0.0"


@dataclass
class ConvResult:
    output: np.ndarray
    macs: int
    overflows: list
    acc_bits: int
    traversal: str
    inplace: bool


def conv_output_shape(in_h: int, in_w: int, node: Node) -> Tuple[int, int]:
    out_h = (in_h + 2 * node.padding - node.kernel) // node.stride + 1
    out_w = (in_w + 2 * node.padding - node.kernel) // node.stride + 1
    return out_h, out_w


def integer_conv(activation: np.ndarray, weight: np.ndarray,
                 node: Node, cfg: Config) -> ConvResult:
    if activation.ndim != 2 or weight.ndim != 2:
        raise ValueError("integer_conv expects 2D activation and weight")
    if weight.shape != (node.kernel, node.kernel):
        raise ValueError("weight shape %s != kernel %d"
                         % (weight.shape, node.kernel))
    lo, hi = cfg.int_limits()
    in_h, in_w = activation.shape
    out_h, out_w = conv_output_shape(in_h, in_w, node)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("conv output shape non-positive for node %s"
                         % node.node_id)
    padded = np.zeros((in_h + 2 * node.padding,
                       in_w + 2 * node.padding), dtype=np.int64)
    padded[node.padding:node.padding + in_h,
           node.padding:node.padding + in_w] = activation.astype(np.int64)
    output = np.zeros((out_h, out_w), dtype=np.int64)
    overflows: List[dict] = []
    macs = 0
    for oh in range(out_h):
        for ow in range(out_w):
            acc = 0
            for kh in range(node.kernel):
                for kw in range(node.kernel):
                    ah = oh * node.stride + kh
                    aw = ow * node.stride + kw
                    acc += int(padded[ah, aw]) * int(weight[kh, kw])
                    macs += 1
                    if acc < lo or acc > hi:
                        record = {"node_id": node.node_id,
                                  "position": [oh, ow, kh, kw],
                                  "value": acc, "limits": [lo, hi]}
                        if cfg.overflow_policy == "ERROR":
                            raise QCGError(E_ACC_OVERFLOW,
                                           "accumulator out of range at %s"
                                           % record["position"],
                                           node_id=node.node_id,
                                           field="accumulator")
                        overflows.append(record)
                        if cfg.overflow_policy == "SATURATE":
                            acc = min(max(acc, lo), hi)
                        else:  # WRAP
                            span = hi - lo + 1
                            acc = ((acc - lo) % span) + lo
            output[oh, ow] = acc
    return ConvResult(output=output.astype(np.int64), macs=macs,
                      overflows=overflows, acc_bits=cfg.accumulator_bits,
                      traversal=TRAVERSAL_ID, inplace=node.inplace)
