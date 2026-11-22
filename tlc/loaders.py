"""输入加载与校验层。

校验规则与业务码映射：
- 任务描述非法           -> E_TASK_INVALID
- 样本缺失/特征库损坏    -> E_FEATURE_MISSING
- 编码器结构/校验和非法  -> E_ENCODER_SHAPE
- 贡献清单非法           -> E_CONTRIB_INVALID
- 替换计划非法           -> E_PLAN_INVALID
"""

import json
import math
import os

from .errors import (
    BizError,
    E_CONTRIB_INVALID,
    E_ENCODER_SHAPE,
    E_FEATURE_MISSING,
    E_PLAN_INVALID,
    E_TASK_INVALID,
)
from .hashing import sha256_hex, stable_seed
from .model import Config, Encoder, EncoderLayer, FeatureBank, TaskSpec


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_config(path, overrides=None) -> Config:
    data = _read_json(path) if path and os.path.exists(path) else {}
    if overrides:
        data = {**data, **overrides}
    cfg = Config()
    for key in cfg.to_dict():
        if key in data:
            setattr(cfg, key, data[key])
    if cfg.epochs <= 0 or cfg.batch_size <= 0 or cfg.max_concurrency <= 0:
        raise BizError(E_TASK_INVALID, "配置中 epochs/batch_size/max_concurrency 必须为正整数")
    if cfg.contribution_shift_threshold < 0:
        raise BizError(E_TASK_INVALID, "贡献偏移阈值不能为负")
    return cfg


def load_tasks(path) -> list:
    data = _read_json(path)
    tasks = []
    seen = set()
    for raw in data.get("tasks", []):
        name = raw.get("name")
        if not name or name in seen:
            raise BizError(E_TASK_INVALID, f"任务名缺失或重复: {name!r}")
        seen.add(name)
        classes = raw.get("classes") or []
        if not classes or len(set(classes)) != len(classes):
            raise BizError(E_TASK_INVALID, f"任务 {name} 类别集合为空或存在重复")
        train_ids = list(raw.get("train_ids") or [])
        val_ids = list(raw.get("val_ids") or [])
        if not train_ids or not val_ids:
            raise BizError(E_TASK_INVALID, f"任务 {name} 训练/验证样本 ID 不能为空")
        if set(train_ids) & set(val_ids):
            raise BizError(E_TASK_INVALID, f"任务 {name} 训练与验证样本 ID 存在交集")
        target_layers = list(raw.get("target_layers") or [])
        if not target_layers or len(set(target_layers)) != len(target_layers):
            raise BizError(E_TASK_INVALID, f"任务 {name} 目标层白名单为空或重复")
        adapter = raw.get("adapter") or {}
        adapter_type = adapter.get("type", "linear")
        if adapter_type not in ("linear", "lowrank"):
            raise BizError(E_TASK_INVALID, f"任务 {name} 适配器类型非法: {adapter_type}")
        rank = int(adapter.get("rank", 1))
        scale = float(adapter.get("scale", 1.0))
        if rank <= 0 or scale <= 0:
            raise BizError(E_TASK_INVALID, f"任务 {name} 秩与缩放系数必须为正")
        tasks.append(
            TaskSpec(
                name=name,
                classes=list(classes),
                train_ids=train_ids,
                val_ids=val_ids,
                target_layers=target_layers,
                adapter_type=adapter_type,
                rank=rank,
                scale=scale,
                retrain_budget=int(raw.get("retrain_budget", 0)),
            )
        )
    if not tasks:
        raise BizError(E_TASK_INVALID, "任务集合为空")
    return tasks


def load_features(path) -> FeatureBank:
    samples = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("id")
            if not sid or sid in samples:
                raise BizError(E_FEATURE_MISSING, f"样本 ID 缺失或重复: {sid!r}")
            feats = rec.get("features")
            if not isinstance(feats, list) or not feats:
                raise BizError(E_FEATURE_MISSING, f"样本 {sid} 特征向量缺失")
            if any((not isinstance(v, (int, float))) or not math.isfinite(v) for v in feats):
                raise BizError(E_FEATURE_MISSING, f"样本 {sid} 特征含非有限值")
            samples[sid] = {
                "features": [float(v) for v in feats],
                "labels": dict(rec.get("labels") or {}),
            }
    if not samples:
        raise BizError(E_FEATURE_MISSING, "特征库为空")
    return FeatureBank(samples=samples, fingerprint="")


def fingerprint_features(bank: FeatureBank) -> str:
    bank.fingerprint = sha256_hex(bank.samples)
    return bank.fingerprint


def validate_task_samples(tasks, bank: FeatureBank):
    for task in tasks:
        for sid in task.train_ids + task.val_ids:
            if sid not in bank.samples:
                raise BizError(
                    E_FEATURE_MISSING, f"任务 {task.name} 引用的样本 {sid} 不在特征库中"
                )
            label = bank.samples[sid]["labels"].get(task.name)
            if label is None:
                raise BizError(
                    E_FEATURE_MISSING, f"样本 {sid} 缺少任务 {task.name} 的标签"
                )
            if label not in task.classes:
                raise BizError(
                    E_TASK_INVALID,
                    f"样本 {sid} 标签 {label!r} 不在任务 {task.name} 类别集合中",
                )


def layer_checksum(layer_dict) -> str:
    return sha256_hex(
        {
            "name": layer_dict["name"],
            "version": layer_dict["version"],
            "weight": layer_dict["weight"],
            "bias": layer_dict["bias"],
        }
    )


def _encoder_from_dict(data) -> Encoder:
    input_dim = int(data.get("input_dim", 0))
    norm = data.get("norm") or {}
    mean = [float(v) for v in norm.get("mean") or []]
    std = [float(v) for v in norm.get("std") or []]
    layers = [
        EncoderLayer(
            name=raw["name"],
            version=int(raw["version"]),
            weight=[[float(v) for v in row] for row in raw["weight"]],
            bias=[float(v) for v in raw["bias"]],
            checksum=raw.get("checksum", ""),
        )
        for raw in data.get("layers", [])
    ]
    return Encoder(input_dim=input_dim, norm_mean=mean, norm_std=std, layers=layers)


def validate_encoder(enc: Encoder) -> Encoder:
    if enc.input_dim <= 0:
        raise BizError(E_ENCODER_SHAPE, "编码器输入维度非法")
    if len(enc.norm_mean) != enc.input_dim or len(enc.norm_std) != enc.input_dim:
        raise BizError(E_ENCODER_SHAPE, "归一化统计量维度与输入维度不一致")
    if any(v <= 0 for v in enc.norm_std):
        raise BizError(E_ENCODER_SHAPE, "归一化标准差必须为正")
    seen = set()
    prev_dim = enc.input_dim
    for layer in enc.layers:
        if layer.name in seen:
            raise BizError(E_ENCODER_SHAPE, f"层名重复: {layer.name}")
        seen.add(layer.name)
        if layer.version < 0:
            raise BizError(E_ENCODER_SHAPE, f"层 {layer.name} 版本号非法")
        out_dim = len(layer.weight)
        if out_dim == 0 or any(len(row) != prev_dim for row in layer.weight):
            raise BizError(
                E_ENCODER_SHAPE, f"层 {layer.name} 权重维度与相邻层输入维度不一致"
            )
        if len(layer.bias) != out_dim:
            raise BizError(E_ENCODER_SHAPE, f"层 {layer.name} 偏置维度与输出维度不一致")
        flat = [v for row in layer.weight for v in row] + list(layer.bias)
        if any(not math.isfinite(v) for v in flat):
            raise BizError(E_ENCODER_SHAPE, f"层 {layer.name} 参数含非有限值")
        if layer.checksum and layer.checksum != layer_checksum(layer.to_dict()):
            raise BizError(E_ENCODER_SHAPE, f"层 {layer.name} 参数校验摘要不匹配")
        prev_dim = out_dim
    if not enc.layers:
        raise BizError(E_ENCODER_SHAPE, "编码器不含任何层")
    return enc


def load_encoder(path) -> Encoder:
    return validate_encoder(_encoder_from_dict(_read_json(path)))


def encoder_from_dict(data) -> Encoder:
    return validate_encoder(_encoder_from_dict(data))


def load_contributions(path, encoder: Encoder, tasks) -> dict:
    """返回 {(layer, task): {"sensitivity": float, "frozen": bool}}。"""
    data = _read_json(path)
    entries = {}
    for raw in data.get("entries", []):
        layer = raw.get("layer")
        task = raw.get("task")
        sens = raw.get("sensitivity")
        if layer is None or task is None:
            raise BizError(E_CONTRIB_INVALID, "贡献清单条目缺少层名或任务名")
        if not isinstance(sens, (int, float)) or not math.isfinite(sens):
            raise BizError(
                E_CONTRIB_INVALID, f"贡献清单 {layer}/{task} 敏感度系数非有限值"
            )
        entries[(layer, task)] = {
            "sensitivity": float(sens),
            "frozen": bool(raw.get("frozen", False)),
        }
    layer_names = [layer.name for layer in encoder.layers]
    for layer in layer_names:
        for task in tasks:
            if (layer, task.name) not in entries:
                raise BizError(
                    E_CONTRIB_INVALID,
                    f"贡献清单缺少层 {layer} 对任务 {task.name} 的敏感度声明",
                )
    return entries


def load_replacement_plan(path) -> list:
    data = _read_json(path)
    batches = []
    for batch in data.get("batches", []):
        items = []
        for raw in batch:
            layer = raw.get("layer")
            new_version = raw.get("new_version")
            if not layer or not isinstance(new_version, int):
                raise BizError(E_PLAN_INVALID, "替换计划条目缺少层名或新版本号")
            items.append({"layer": layer, "new_version": new_version})
        batches.append(items)
    return batches


def apply_replacements(encoder: Encoder, batch_items, contributions) -> Encoder:
    """应用一批基座层替换。

    新层参数由 (层名, 新版本号, 形状) 确定性派生，模拟外部基座更新，
    保证本地可重复。层版本号必须单调不减，冻结层禁止替换。
    """
    import numpy as np

    layers = [EncoderLayer(**{**layer.to_dict()}) for layer in encoder.layers]
    index = {layer.name: i for i, layer in enumerate(layers)}
    for item in batch_items:
        name = item["layer"]
        new_version = item["new_version"]
        if name not in index:
            raise BizError(E_PLAN_INVALID, f"替换计划引用的层 {name} 不在编码器中")
        layer = layers[index[name]]
        if new_version < layer.version:
            raise BizError(
                E_PLAN_INVALID,
                f"层 {name} 新版本号 {new_version} 小于当前版本 {layer.version}（须单调不减）",
            )
        for (entry_layer, _task), entry in contributions.items():
            if entry_layer == name and entry["frozen"]:
                raise BizError(
                    E_PLAN_INVALID, f"层 {name} 被贡献清单声明为冻结，禁止替换"
                )
        out_dim = len(layer.weight)
        in_dim = len(layer.weight[0])
        rng = np.random.default_rng(stable_seed("encoder-layer", name, new_version))
        weight = (rng.standard_normal((out_dim, in_dim)) / math.sqrt(in_dim)).tolist()
        bias = (rng.standard_normal(out_dim) * 0.1).tolist()
        layers[index[name]] = EncoderLayer(
            name=name, version=new_version, weight=weight, bias=bias
        )
        layers[index[name]].checksum = layer_checksum(layers[index[name]].to_dict())
    return validate_encoder(
        Encoder(
            input_dim=encoder.input_dim,
            norm_mean=list(encoder.norm_mean),
            norm_std=list(encoder.norm_std),
            layers=layers,
        )
    )
