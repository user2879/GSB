"""Deterministic fixed-point LoRA training and export.

Training runs entirely on int64 fixed-point values.  Batch order is the
sample_id-sorted order supplied by the caller; initialization derives from a
hash of (task, layer, rank) so no runtime global random state is involved.
A bit-parallel float64 reference path is provided for diagnostics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .fixedpoint import (
    Ledger,
    matmul_fixed,
    mul_fixed,
    quantize,
    round_div,
)


@dataclass
class TrainedAdapter:
    task_name: str
    target_layer: str
    rank: int
    alpha: float
    frac_bits: int
    A: np.ndarray  # (rank, in) int64 fixed-point
    B: np.ndarray  # (out, rank) int64 fixed-point
    losses_fixed: list  # per-epoch int64 loss (fixed-point)
    ledger: list  # residual ledger entries
    base_fingerprint: str = ""
    layer_fingerprint: str = ""
    key: str = ""

    @property
    def losses(self):
        return [float(v) / float(1 << self.frac_bits) for v in self.losses_fixed]

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


def adapter_key(task_name, layer, rank, alpha, layer_fingerprint, train_hash) -> str:
    material = f"{task_name}|{layer}|{rank}|{alpha!r}|{layer_fingerprint}|{train_hash}"
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def init_adapter_float(task_name, layer, rank, in_dim, out_dim):
    """Hash-seeded local RNG init; A ~ N(0, 0.05), B = 0 (standard LoRA init)."""
    seed = int.from_bytes(
        hashlib.sha256(f"init|{task_name}|{layer}|{rank}".encode()).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, 0.5, size=(rank, in_dim))
    B = np.zeros((out_dim, rank), dtype=np.float64)
    return A, B


def train_adapter_fixed(
    task,
    W_fixed,
    b_fixed,
    xs_fixed,
    ys_fixed,
    lr: float,
    epochs: int,
    frac_bits: int,
) -> TrainedAdapter:
    """Full fixed-point training. xs/ys are lists of fixed-point column vectors."""
    out_dim, in_dim = W_fixed.shape
    rank = task.rank
    ledger = Ledger()

    A_f, B_f = init_adapter_float(task.name, task.target_layer, rank, in_dim, out_dim)
    A = quantize(A_f, frac_bits, ledger, "init_A")
    B = quantize(B_f, frac_bits, ledger, "init_B")
    lr_q = quantize(np.float64(lr), frac_bits, ledger, "init_lr")
    scaling_q = quantize(np.float64(task.scaling), frac_bits, ledger, "init_scaling")

    wx_list = [matmul_fixed(W_fixed, x.reshape(-1, 1), frac_bits, ledger, "forward_Wx") for x in xs_fixed]
    losses_fixed = []

    for _epoch in range(epochs):
        epoch_loss = np.int64(0)
        for x, y, wx in zip(xs_fixed, ys_fixed, wx_list):
            x_col = x.reshape(-1, 1)
            y_col = y.reshape(-1, 1)
            ax = matmul_fixed(A, x_col, frac_bits, ledger, "forward_Ax")
            bax = matmul_fixed(B, ax, frac_bits, ledger, "forward_BAx")
            sbax = mul_fixed(bax, scaling_q, frac_bits, ledger, "forward_scale")
            yhat = wx + sbax + b_fixed.reshape(-1, 1)
            err = yhat - y_col
            sq = mul_fixed(err, err, frac_bits, ledger, "loss_square")
            epoch_loss = epoch_loss + np.int64(int(np.sum(sq)))

            # grad of mean-square loss wrt the adapter output: 2/out * err
            g = round_div(err * np.int64(2), out_dim, ledger, "grad_scale")
            # grad_B = lr * scaling * g @ (Ax)^T
            gs = mul_fixed(g, scaling_q, frac_bits, ledger, "grad_B_prescale")
            grad_b = matmul_fixed(gs, ax.reshape(1, -1), frac_bits, ledger, "grad_B")
            B = B - mul_fixed(grad_b, lr_q, frac_bits, ledger, "update_B")
            # grad_A = lr * scaling * (B^T g) @ x^T
            h = matmul_fixed(B.T, g, frac_bits, ledger, "grad_A_backprop")
            hs = mul_fixed(h, scaling_q, frac_bits, ledger, "grad_A_prescale")
            grad_a = matmul_fixed(hs, x_col.reshape(1, -1), frac_bits, ledger, "grad_A")
            A = A - mul_fixed(grad_a, lr_q, frac_bits, ledger, "update_A")
        mean_loss = round_div(epoch_loss, out_dim * len(xs_fixed), ledger, "loss_mean")
        losses_fixed.append(int(mean_loss))

    return TrainedAdapter(
        task_name=task.name,
        target_layer=task.target_layer,
        rank=rank,
        alpha=task.alpha,
        frac_bits=frac_bits,
        A=A,
        B=B,
        losses_fixed=losses_fixed,
        ledger=ledger.entries,
    )


def train_adapter_float64(task, W, b, xs, ys, lr: float, epochs: int):
    """Float64 reference path mirroring the fixed-point algorithm exactly."""
    out_dim, in_dim = W.shape
    rank = task.rank
    A, B = init_adapter_float(task.name, task.target_layer, rank, in_dim, out_dim)
    scaling = task.scaling
    losses = []
    wx_list = [W @ x for x in xs]
    with np.errstate(all="ignore"):
        for _epoch in range(epochs):
            epoch_loss = 0.0
            for x, y, wx in zip(xs, ys, wx_list):
                ax = A @ x
                yhat = wx + scaling * (B @ ax) + b
                err = yhat - y
                epoch_loss += float(np.sum(err * err))
                g = 2.0 / out_dim * err
                grad_b = scaling * np.outer(g, ax)
                B = B - lr * grad_b
                h = B.T @ g
                grad_a = scaling * np.outer(h, x)
                A = A - lr * grad_a
            losses.append(epoch_loss / (out_dim * len(xs)))
    return A, B, losses
