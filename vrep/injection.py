"""独立注入配置与运行时故障注入（业务要求 8）。

注入配置示例::

    {
      "injections": [
        {"id": "INJ-001", "type": "truncate_png", "image": "img_0003",
         "keep_bytes": 40},
        {"id": "INJ-002", "type": "truth_out_of_bounds", "image": "img_0005"},
        {"id": "INJ-003", "type": "nonfinite_feature", "image": "img_0002",
         "tile_seq": 0},
        {"id": "INJ-004", "type": "quant_level_out_of_range"}
      ]
    }

每个注入项是否命中都写入 JSON 诊断报告，并映射到对应业务码。
"""

from __future__ import annotations

from .jsonio import read_json
from .errors import (
    E_IMAGE_FORMAT,
    E_TRUTH_INVALID,
    E_CONFIG_INVALID,
    E_NONFINITE_FEATURE,
    E_QUANT_LEVEL,
    Failure,
)

VALID_TYPES = (
    "truncate_png",
    "truth_out_of_bounds",
    "nonfinite_feature",
    "quant_level_out_of_range",
)

TYPE_TO_CODE = {
    "truncate_png": E_IMAGE_FORMAT,
    "truth_out_of_bounds": E_TRUTH_INVALID,
    "nonfinite_feature": E_NONFINITE_FEATURE,
    "quant_level_out_of_range": E_QUANT_LEVEL,
}


class InjectionManager:
    def __init__(self, config_path: str | None):
        self.entries: list[dict] = []
        if config_path:
            payload = read_json(config_path)
            entries = payload.get("injections", []) if isinstance(payload, dict) else []
            for entry in entries:
                if entry.get("type") not in VALID_TYPES:
                    raise ValueError(f"未知注入类型: {entry.get('type')}")
                self.entries.append(dict(entry))
        self.hits: list[dict] = []

    def _find(self, injection_type: str, image: str | None = None):
        for entry in self.entries:
            if entry["type"] != injection_type:
                continue
            if image is None or entry.get("image") == image:
                return entry
        return None

    def _record_hit(self, entry: dict, code: str, detail: str) -> None:
        if any(hit["id"] == entry["id"] for hit in self.hits):
            return
        self.hits.append(
            {
                "id": entry["id"],
                "type": entry["type"],
                "image": entry.get("image"),
                "code": code,
                "detail": detail,
            }
        )

    # ---- 输入侧注入 ----
    def maybe_truncate_png(self, image: str, raw: bytes):
        """返回 (可能被截断的字节, Failure|None)。"""
        entry = self._find("truncate_png", image)
        if entry is None:
            return raw, None
        keep = int(entry.get("keep_bytes", 32))
        clipped = raw[: max(0, keep)]
        failure = Failure(
            image, "load", E_IMAGE_FORMAT,
            f"注入截断 PNG，仅保留 {len(clipped)} 字节", entry["id"],
        )
        self._record_hit(entry, E_IMAGE_FORMAT, failure.detail)
        return clipped, failure

    def maybe_corrupt_truth(self, image: str, truth_data: dict, width: int, height: int):
        """返回 (可能被改写的真值数据, Failure|None)。"""
        entry = self._find("truth_out_of_bounds", image)
        if entry is None:
            return truth_data, None
        corrupted = {
            "regions": [
                {
                    "category": "injected_oob_rect",
                    "box": [width - 2, height - 2, width + 4, height + 4],
                }
            ]
        }
        failure = Failure(
            image, "truth", E_TRUTH_INVALID,
            "注入真值矩形越界", entry["id"],
        )
        self._record_hit(entry, E_TRUTH_INVALID, failure.detail)
        return corrupted, failure

    def maybe_config_quant_fault(self, config_dict: dict):
        """量化档位越界注入：返回 (可能被改写的配置, Failure|None)。"""
        entry = self._find("quant_level_out_of_range")
        if entry is None:
            return config_dict, None
        corrupted = dict(config_dict)
        corrupted["quant_level"] = int(entry.get("bad_level", 13))
        failure = Failure(
            "<config>", "config", E_CONFIG_INVALID,
            f"注入非法量化档位 {corrupted['quant_level']}", entry["id"],
        )
        self._record_hit(entry, E_QUANT_LEVEL, failure.detail)
        return corrupted, failure

    # ---- 计算侧注入 ----
    def nonfinite_targets(self) -> dict[str, set[int]]:
        targets: dict[str, set[int]] = {}
        for entry in self.entries:
            if entry["type"] == "nonfinite_feature":
                targets.setdefault(entry["image"], set()).add(
                    int(entry.get("tile_seq", 0))
                )
        return targets

    def record_nonfinite_hit(self, image: str, tile_seq: int) -> Failure:
        entry = self._find("nonfinite_feature", image)
        detail = f"注入非有限特征值: {image} 分块 seq={tile_seq}"
        if entry is not None and int(entry.get("tile_seq", 0)) == tile_seq:
            self._record_hit(entry, E_NONFINITE_FEATURE, detail)
            return Failure(image, "feature", E_NONFINITE_FEATURE, detail, entry["id"])
        return Failure(image, "feature", E_NONFINITE_FEATURE, detail)

    # ---- 诊断报告 ----
    def diagnostics(self) -> dict:
        hit_ids = {hit["id"] for hit in self.hits}
        items = []
        for entry in self.entries:
            items.append(
                {
                    "id": entry["id"],
                    "type": entry["type"],
                    "image": entry.get("image"),
                    "expected_code": TYPE_TO_CODE[entry["type"]],
                    "hit": entry["id"] in hit_ids,
                }
            )
        items.sort(key=lambda item: item["id"])
        return {
            "injection_count": len(self.entries),
            "hit_count": len(hit_ids),
            "all_hit": len(hit_ids) == len(self.entries),
            "items": items,
            "hits": sorted(self.hits, key=lambda hit: hit["id"]),
        }
