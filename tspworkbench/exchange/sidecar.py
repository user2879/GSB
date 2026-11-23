"""v2 -> v1 降级旁路元数据。

旁路文件与 v1 交换文件同目录同名加 ``.sidecar.json`` 后缀，记录：
- 配置指纹与版本指纹（再次升版时校验来源，防止错配）；
- 每条权重的四舍五入截断残差、原始精度标签；
- v1 无法表达而被丢弃的命名缺失处理方式。

再次 v1 -> v2 时优先据此还原命名信息，而不是退回默认值。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import WorkbenchError, E_SIDECAR_MISMATCH

SIDECAR_SUFFIX = ".sidecar.json"


@dataclass
class SidecarEntry:
    series_id: str
    window_index: int
    model: str
    original_weight: float
    rounded_permille: int
    residual: float
    precision_label: str


@dataclass
class Sidecar:
    config_fingerprint: str
    version_fingerprint: str
    missing_strategy: str
    entries: list[SidecarEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_fingerprint": self.config_fingerprint,
            "version_fingerprint": self.version_fingerprint,
            "missing_strategy": self.missing_strategy,
            "entries": [entry.__dict__ for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sidecar":
        return cls(
            config_fingerprint=data["config_fingerprint"],
            version_fingerprint=data["version_fingerprint"],
            missing_strategy=data["missing_strategy"],
            entries=[SidecarEntry(**e) for e in data.get("entries", [])],
        )

    def entry_key(self) -> dict[tuple[str, int, str], SidecarEntry]:
        return {(e.series_id, e.window_index, e.model): e for e in self.entries}

    def write(self, v1_path: str | Path) -> Path:
        path = Path(str(v1_path) + SIDECAR_SUFFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2,
                                  sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path


def read_sidecar(v1_path: str | Path) -> Sidecar | None:
    path = Path(str(v1_path) + SIDECAR_SUFFIX)
    if not path.exists():
        return None
    return Sidecar.from_dict(json.loads(path.read_text(encoding="utf-8")))


def assert_sidecar_matches(sidecar: Sidecar, config_fingerprint: str,
                           version_fingerprint: str) -> None:
    if (sidecar.config_fingerprint != config_fingerprint
            or sidecar.version_fingerprint != version_fingerprint):
        raise WorkbenchError(
            E_SIDECAR_MISMATCH, "旁路元数据指纹与当前配置/版本不匹配，拒绝还原命名信息",
            context={
                "sidecar_config": sidecar.config_fingerprint,
                "expect_config": config_fingerprint,
                "sidecar_version": sidecar.version_fingerprint,
                "expect_version": version_fingerprint,
            },
        )
