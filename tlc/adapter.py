"""任务适配器：线性探针 / 低秩矩阵，确定性训练与评估。

- 初始化种子由 (全局种子, 任务名, 目标层, 适配器配置) 派生，与线程调度无关；
- 固定批量顺序（按样本 ID 字典序）、固定学习率、固定轮数；
- 逐轮记录损失；验证集输出准确率与宏平均 F1；
- 产物记录引用的基座层版本集合与训练期间逐层贡献份额快照。
"""

import numpy as np

from .encoder import contribution_shares, target_matrix
from .hashing import sha256_hex, stable_seed


def _class_indices(task, bank, ids):
    pos = {c: i for i, c in enumerate(task.classes)}
    return np.asarray([pos[bank.samples[s]["labels"][task.name]] for s in ids], dtype=np.int64)


def _features_of(bank, ids):
    return np.asarray([bank.samples[s]["features"] for s in ids], dtype=np.float64)


def init_params(task, input_dim, num_classes, seed):
    rng = np.random.default_rng(seed)
    if task.adapter_type == "linear":
        return {
            "W": (rng.standard_normal((num_classes, input_dim)) / np.sqrt(input_dim)).tolist(),
            "b": [0.0] * num_classes,
        }
    return {
        "A": (rng.standard_normal((task.rank, input_dim)) / np.sqrt(input_dim)).tolist(),
        "B": [[0.0] * task.rank for _ in range(num_classes)],
        "b": [0.0] * num_classes,
    }


def _effective_W(task, params):
    if task.adapter_type == "linear":
        return np.asarray(params["W"], dtype=np.float64)
    A = np.asarray(params["A"], dtype=np.float64)
    B = np.asarray(params["B"], dtype=np.float64)
    return (task.scale / task.rank) * (B @ A)


def _softmax_ce(task, params, Z, Y):
    """返回 (loss, dZ)。softmax 交叉熵，均值规约，固定运算顺序。"""
    W = _effective_W(task, params)
    b = np.asarray(params["b"], dtype=np.float64)
    logits = Z @ W.T + b
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = Z.shape[0]
    loss = 0.0
    for i in range(n):
        loss -= float(np.log(probs[i, Y[i]]))
    loss /= n
    G = probs
    for i in range(n):
        G[i, Y[i]] -= 1.0
    G /= n
    return loss, G, W


def loss_and_dz(task, params, Z, Y):
    """供贡献追踪调用：返回 (loss, dZ)。"""
    loss, G, W = _softmax_ce(task, params, Z, Y)
    return loss, G @ W


def _grads(task, params, Z, Y):
    loss, G, W = _softmax_ce(task, params, Z, Y)
    grads = {"b": G.sum(axis=0)}
    if task.adapter_type == "linear":
        grads["W"] = G.T @ Z
    else:
        dWe = G.T @ Z
        A = np.asarray(params["A"], dtype=np.float64)
        B = np.asarray(params["B"], dtype=np.float64)
        grads["A"] = (task.scale / task.rank) * (B.T @ dWe)
        grads["B"] = (task.scale / task.rank) * (dWe @ A.T)
    return loss, grads


def _apply(params, grads, lr):
    out = {}
    for key in sorted(params):
        p = np.asarray(params[key], dtype=np.float64)
        out[key] = (p - lr * grads[key]).tolist()
    return out


def content_key(task, cfg, features_fingerprint):
    """任务侧内容键：任务规格 + 训练超参 + 特征指纹（不含基座版本）。

    基座变化由贡献偏移判定处理，不进入内容键。
    重训预算仅影响调度，不进入内容键。
    """
    task_dict = task.to_dict()
    task_dict.pop("retrain_budget", None)
    return sha256_hex(
        {
            "task": task_dict,
            "train": {
                "seed": cfg.seed,
                "learning_rate": cfg.learning_rate,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
            },
            "features": features_fingerprint,
        }
    )


def train_adapter(task, encoder, bank, cfg, sensitivities, features_fingerprint):
    """训练单个任务适配器，返回确定性产物字典。"""
    ids = sorted(task.train_ids)
    X = _features_of(bank, ids)
    Y = _class_indices(task, bank, ids)
    Z, _slices = target_matrix(encoder, X, task.target_layers)
    seed = stable_seed(
        "adapter", cfg.seed, task.name, tuple(task.target_layers),
        task.adapter_type, task.rank, task.scale, tuple(task.classes),
    )
    params = init_params(task, Z.shape[1], len(task.classes), seed)

    n = Z.shape[0]
    starts = list(range(0, n, cfg.batch_size))
    losses = []
    epoch_shares = []
    shares_loss = 0.0
    for _epoch in range(cfg.epochs):
        total = 0.0
        for s in starts:
            e = min(s + cfg.batch_size, n)
            loss, grads = _grads(task, params, Z[s:e], Y[s:e])
            params = _apply(params, grads, cfg.learning_rate)
            total += loss
        losses.append(total / len(starts))
        shares, shares_loss = contribution_shares(
            encoder, task, params, X, Y, sensitivities,
            lambda p, Zm, Ym: loss_and_dz(task, p, Zm, Ym),
        )
        epoch_shares.append(shares)

    artifact = {
        "task": task.name,
        "adapter": {"type": task.adapter_type, "rank": task.rank, "scale": task.scale},
        "params": params,
        "param_checksum": sha256_hex(params),
        "base_layer_versions": {layer.name: layer.version for layer in encoder.layers},
        "content_key": content_key(task, cfg, features_fingerprint),
        "train_losses": losses,
        "epoch_shares": epoch_shares,
        "shares": epoch_shares[-1],
        "shares_loss": shares_loss,
        "train_cost": cfg.epochs * len(task.train_ids),
    }
    return artifact


def evaluate(task, encoder, bank, params):
    """在验证样本上计算准确率与宏平均 F1（类别顺序取任务声明顺序）。"""
    ids = sorted(task.val_ids)
    X = _features_of(bank, ids)
    Y = _class_indices(task, bank, ids)
    Z, _slices = target_matrix(encoder, X, task.target_layers)
    W = _effective_W(task, params)
    b = np.asarray(params["b"], dtype=np.float64)
    pred = (Z @ W.T + b).argmax(axis=1)
    acc = float((pred == Y).mean())
    f1s = []
    for ci in range(len(task.classes)):
        tp = float(((pred == ci) & (Y == ci)).sum())
        fp = float(((pred == ci) & (Y != ci)).sum())
        fn = float(((pred != ci) & (Y == ci)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0.0 else 2 * tp / denom)
    macro = 0.0
    for v in f1s:
        macro += v
    macro /= len(f1s)
    return {"accuracy": acc, "macro_f1": macro}
