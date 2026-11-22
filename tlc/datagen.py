"""固定种子合成数据生成：特征库、任务描述、编码器参数包、
基座贡献清单、替换计划与批次覆盖样例。"""

import json
import os
import zlib

import numpy as np

from .loaders import layer_checksum

SEED = 20241101
INPUT_DIM = 6
N_SAMPLES = 60


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1))


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")


def _make_encoder_layer(rng, name, version, out_dim, in_dim):
    layer = {
        "name": name,
        "version": version,
        "weight": (rng.standard_normal((out_dim, in_dim)) / np.sqrt(in_dim)).tolist(),
        "bias": (rng.standard_normal(out_dim) * 0.1).tolist(),
    }
    layer["checksum"] = layer_checksum(layer)
    return layer


def generate(out_dir):
    rng = np.random.default_rng(SEED)

    # ---- 特征与标签：每任务一组类别原型，标签由最近原型确定 ----
    X = rng.standard_normal((N_SAMPLES, INPUT_DIM))
    ids = [f"s{i:03d}" for i in range(N_SAMPLES)]

    task_defs = [
        ("task_alpha", ["A0", "A1", "A2"], ["layer1", "layer3"], "linear", 1, 1.0),
        ("task_beta", ["B0", "B1"], ["layer3"], "lowrank", 2, 1.5),
        ("task_gamma", ["G0", "G1", "G2"], ["layer2", "layer3"], "linear", 1, 1.0),
        ("task_delta", ["D0", "D1"], ["layer2"], "lowrank", 2, 0.8),
    ]
    labels = {}
    for name, classes, _t, _a, _r, _s in task_defs:
        proto_rng = np.random.default_rng(SEED + zlib.crc32(name.encode("utf-8")) % 1000)
        protos = proto_rng.standard_normal((len(classes), INPUT_DIM)) * 1.5
        d = ((X[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
        y = d.argmin(axis=1)
        labels[name] = [classes[int(k)] for k in y]

    records = []
    for i, sid in enumerate(ids):
        records.append(
            {
                "id": sid,
                "features": [float(v) for v in X[i]],
                "labels": {name: labels[name][i] for name, *_ in task_defs},
            }
        )
    _write_jsonl(os.path.join(out_dir, "features.jsonl"), records)

    # ---- 任务描述：训练/验证划分由固定种子排列派生 ----
    split_rng = np.random.default_rng(SEED + 1)
    tasks = []
    splits = {}
    for name, classes, targets, atype, rank, scale in task_defs:
        if name == "task_delta":
            continue  # task_delta 于 batch_2 加入
        perm = split_rng.permutation(N_SAMPLES).tolist()
        train = sorted(ids[k] for k in perm[:24])
        val = sorted(ids[k] for k in perm[24:36])
        splits[name] = (train, val)
        tasks.append(
            {
                "name": name,
                "classes": classes,
                "train_ids": train,
                "val_ids": val,
                "target_layers": targets,
                "adapter": {"type": atype, "rank": rank, "scale": scale},
                "retrain_budget": 100000,
            }
        )
    _write_json(os.path.join(out_dir, "tasks.json"), {"tasks": tasks})

    # ---- 编码器参数包 ----
    enc_rng = np.random.default_rng(SEED + 2)
    mean = (enc_rng.standard_normal(INPUT_DIM) * 0.3).tolist()
    std = (np.abs(enc_rng.standard_normal(INPUT_DIM)) + 0.5).tolist()
    layers = [
        _make_encoder_layer(enc_rng, "layer1", 1, 8, INPUT_DIM),
        _make_encoder_layer(enc_rng, "layer2", 1, 8, 8),
        _make_encoder_layer(enc_rng, "layer3", 1, 4, 8),
    ]
    _write_json(
        os.path.join(out_dir, "encoder.json"),
        {"input_dim": INPUT_DIM, "norm": {"mean": mean, "std": std}, "layers": layers},
    )

    # ---- 基座贡献清单：逐层逐任务敏感度系数与冻结标记 ----
    sens = {
        "task_alpha": {"layer1": 0.99, "layer2": 0.005, "layer3": 0.005},
        "task_beta": {"layer1": 0.2, "layer2": 0.3, "layer3": 0.5},
        "task_gamma": {"layer1": 0.1, "layer2": 0.45, "layer3": 0.45},
        "task_delta": {"layer1": 0.15, "layer2": 0.55, "layer3": 0.3},
    }
    entries = []
    for layer in ("layer1", "layer2", "layer3"):
        for name, *_ in task_defs:
            entries.append(
                {
                    "layer": layer,
                    "task": name,
                    "sensitivity": sens[name][layer],
                    "frozen": False,
                }
            )
    _write_json(os.path.join(out_dir, "contributions.json"), {"entries": entries})

    # ---- 替换计划：三批逐层替换 ----
    _write_json(
        os.path.join(out_dir, "replacement_plan.json"),
        {
            "batches": [
                [{"layer": "layer2", "new_version": 2}],
                [{"layer": "layer1", "new_version": 2}],
                [{"layer": "layer3", "new_version": 2}],
            ]
        },
    )

    # ---- 全局配置 ----
    _write_json(
        os.path.join(out_dir, "config.json"),
        {
            "seed": 7,
            "learning_rate": 0.2,
            "epochs": 6,
            "batch_size": 8,
            "max_concurrency": 3,
            "contribution_shift_threshold": 0.02,
            "closure_rel_tol": 1e-9,
        },
    )

    # ---- 批次覆盖：任务增删改、预算收紧、阈值调整、特征替换 ----
    # batch_2：加入 task_delta；task_beta 秩 2->3；task_gamma 预算收紧触发待办
    tasks_b2 = json.loads(json.dumps(tasks))
    for t in tasks_b2:
        if t["name"] == "task_gamma":
            t["retrain_budget"] = 50  # 需求 6*24=144，预算不足 -> 待办
        if t["name"] == "task_beta":
            t["adapter"]["rank"] = 3
    perm = split_rng.permutation(N_SAMPLES).tolist()
    train = sorted(ids[k] for k in perm[36:58])
    val = sorted(ids[k] for k in perm[58:60] + perm[34:36])
    tasks_b2.append(
        {
            "name": "task_delta",
            "classes": ["D0", "D1"],
            "train_ids": train,
            "val_ids": sorted(set(val)),
            "target_layers": ["layer2"],
            "adapter": {"type": "lowrank", "rank": 2, "scale": 0.8},
            "retrain_budget": 100000,
        }
    )
    _write_json(os.path.join(out_dir, "batches", "batch_2", "tasks.json"),
                {"tasks": tasks_b2})

    # batch_3：恢复 task_gamma 预算；task_beta 类别集合修改；特征部分替换；阈值调整
    tasks_b3 = json.loads(json.dumps(tasks_b2))
    for t in tasks_b3:
        if t["name"] == "task_gamma":
            t["retrain_budget"] = 100000
        if t["name"] == "task_beta":
            t["classes"] = ["B0", "B1", "B2"]
    _write_json(os.path.join(out_dir, "batches", "batch_3", "tasks.json"),
                {"tasks": tasks_b3})
    rep_rng = np.random.default_rng(SEED + 3)
    records_b3 = []
    replaced = {ids[k] for k in rep_rng.permutation(N_SAMPLES)[:10]}
    for rec in records:
        rec = json.loads(json.dumps(rec))
        if rec["id"] in replaced:
            rec["features"] = [float(v) for v in rep_rng.standard_normal(INPUT_DIM)]
        records_b3.append(rec)
    _write_jsonl(os.path.join(out_dir, "batches", "batch_3", "features.jsonl"),
                 records_b3)
    _write_json(os.path.join(out_dir, "batches", "batch_3", "config.json"),
                {"contribution_shift_threshold": 0.05})
    return out_dir
