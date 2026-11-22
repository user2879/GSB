"""配置与数据模型定义。"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class Config:
    seed: int = 7
    learning_rate: float = 0.2
    epochs: int = 6
    batch_size: int = 8
    max_concurrency: int = 3
    contribution_shift_threshold: float = 0.02
    closure_rel_tol: float = 1e-9

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TaskSpec:
    name: str
    classes: List[str]
    train_ids: List[str]
    val_ids: List[str]
    target_layers: List[str]
    adapter_type: str = "linear"   # linear | lowrank
    rank: int = 1
    scale: float = 1.0
    retrain_budget: int = 0        # 按 训练轮次*样本数 计

    def sort_key(self):
        # 由任务名与目标层名派生的稳定排序键
        return (self.name, tuple(self.target_layers))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "classes": list(self.classes),
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "target_layers": list(self.target_layers),
            "adapter": {
                "type": self.adapter_type,
                "rank": self.rank,
                "scale": self.scale,
            },
            "retrain_budget": self.retrain_budget,
        }


@dataclass
class EncoderLayer:
    name: str
    version: int
    weight: List[List[float]]   # (out_dim, in_dim)
    bias: List[float]           # (out_dim,)
    checksum: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "weight": self.weight,
            "bias": self.bias,
            "checksum": self.checksum,
        }


@dataclass
class Encoder:
    input_dim: int
    norm_mean: List[float]
    norm_std: List[float]
    layers: List[EncoderLayer] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "norm": {"mean": list(self.norm_mean), "std": list(self.norm_std)},
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def layer_index(self) -> Dict[str, int]:
        return {layer.name: i for i, layer in enumerate(self.layers)}


@dataclass
class FeatureBank:
    # id -> {"features": [...], "labels": {task: label}}
    samples: Dict[str, dict]
    fingerprint: str = ""


@dataclass
class Snapshot:
    """已提交状态：适配器产物、最近贡献份额、待办清单。"""

    manifest: Optional[dict] = None          # 已提交清单（含基座版本指纹与配置哈希）
    encoder: Optional[Encoder] = None        # 当前基座
    adapters: Dict[str, dict] = field(default_factory=dict)
    last_shares: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pending: List[str] = field(default_factory=list)

    @property
    def is_blank(self) -> bool:
        return self.manifest is None
