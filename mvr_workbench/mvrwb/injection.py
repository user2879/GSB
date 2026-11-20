"""独立故障注入配置：对指定视图 / 视图对的指定阶段施加真实输入故障。

注入项编号（injection_id）随失败记录写入诊断报告，映射到业务码：
- corrupt_png / missing_camera      -> E_VIEW_INVALID
- parallax_out_of_range             -> E_PARALLAX_OUT_OF_RANGE
- non_finite_feature                -> E_NON_FINITE_FEATURE
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import (
    E_NON_FINITE_FEATURE,
    E_PARALLAX_OUT_OF_RANGE,
    E_VIEW_INVALID,
    StageError,
)
from .util import read_json

SUPPORTED_TYPES = {
    "corrupt_png",
    "missing_camera",
    "parallax_out_of_range",
    "non_finite_feature",
}

TYPE_TO_CODE = {
    "corrupt_png": E_VIEW_INVALID,
    "missing_camera": E_VIEW_INVALID,
    "parallax_out_of_range": E_PARALLAX_OUT_OF_RANGE,
    "non_finite_feature": E_NON_FINITE_FEATURE,
}


@dataclass
class InjectionPlan:
    items: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | None) -> "InjectionPlan":
        if not path:
            return cls()
        if not os.path.exists(path):
            raise StageError(E_VIEW_INVALID, f"注入配置文件不存在: {path}", stage="injection")
        raw = read_json(path)
        items = raw.get("injections", [])
        seen: set[str] = set()
        for item in items:
            injection_id = str(item["id"])
            if injection_id in seen:
                raise StageError(E_VIEW_INVALID, f"注入项编号重复: {injection_id}", stage="injection")
            seen.add(injection_id)
            if item["type"] not in SUPPORTED_TYPES:
                raise StageError(
                    E_VIEW_INVALID, f"注入项 {injection_id} 类型非法: {item['type']}", stage="injection"
                )
            if "view_id" not in item and "pair_id" not in item:
                raise StageError(
                    E_VIEW_INVALID, f"注入项 {injection_id} 缺少视图 / 视图对目标", stage="injection"
                )
        return cls(items=list(items))

    def _matching(self, fault_type: str, *, view_id: int | None = None,
                  pair_id: str | None = None) -> list[dict]:
        result = []
        for item in self.items:
            if item["type"] != fault_type:
                continue
            if view_id is not None and item.get("view_id") != view_id:
                continue
            if pair_id is not None and item.get("pair_id") != pair_id:
                continue
            result.append(item)
        return result

    def _record_hit(self, item: dict, stage: str) -> None:
        self.hits.append(
            {
                "injection_id": item["id"],
                "type": item["type"],
                "code": TYPE_TO_CODE[item["type"]],
                "stage": stage,
                "view_id": item.get("view_id"),
                "pair_id": item.get("pair_id"),
            }
        )

    # -- 加载阶段：PNG 损坏字节 / 相机缺失 ----------------------------------

    def corrupt_image_bytes(self, view_id: int, raw_bytes: bytes, stage: str = "load"):
        for item in self._matching("corrupt_png", view_id=view_id):
            self._record_hit(item, stage)
            # 保留 PNG 容器头，破坏尾部 IDAT/IEND 区段（64 字节足以令解码失败），
            # 使故障落在真实 PNG 解码阶段，而容器头校验仍先通过
            damage = bytearray(raw_bytes)
            for offset in range(1, min(65, len(damage))):
                damage[-offset] = (damage[-offset] + offset * 31) & 0xFF
            return bytes(damage), item["id"]
        return raw_bytes, None

    def require_camera(self, view_id: int, camera_path: str, stage: str = "load") -> None:
        for item in self._matching("missing_camera", view_id=view_id):
            self._record_hit(item, stage)
            raise StageError(
                E_VIEW_INVALID,
                f"注入项 {item['id']} 命中：视图 {view_id} 相机参数缺失",
                stage=stage, view_id=view_id, injection_id=item["id"],
            )

    # -- 特征阶段：非有限特征值 ---------------------------------------------

    def feature_fault(self, view_id: int) -> str | None:
        for item in self._matching("non_finite_feature", view_id=view_id):
            self._record_hit(item, "feature")
            return item["id"]
        return None

    def raise_feature_fault(self, view_id: int) -> None:
        injection_id = self.feature_fault(view_id)
        if injection_id is not None:
            raise StageError(
                E_NON_FINITE_FEATURE,
                f"注入项 {injection_id} 命中：视图 {view_id} 特征响应出现非有限值 (NaN/Inf)",
                stage="feature", view_id=view_id, injection_id=injection_id,
            )

    # -- 匹配阶段：视差越界 -------------------------------------------------

    def parallax_fault(self, pair_id: str) -> str | None:
        for item in self._matching("parallax_out_of_range", pair_id=pair_id):
            self._record_hit(item, "matching")
            return item["id"]
        return None

    def report(self) -> list[dict]:
        return sorted(self.hits, key=lambda item: str(item["injection_id"]))
