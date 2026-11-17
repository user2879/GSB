"""异常注入：以独立注入配置对指定实体与阶段施加真实输入故障。

注入项编号（id）在命中时写入诊断报告，以证明故障确实被触发。
"""
from __future__ import annotations

import json
from pathlib import Path


class Injector:
    def __init__(self, config_path: Path | None):
        self.items: list[dict] = []
        self.hits: list[str] = []
        if config_path and Path(config_path).exists():
            doc = json.loads(Path(config_path).read_text(encoding="utf-8"))
            self.items = list(doc.get("items", []))

    def faults_for(self, stage: str, entity: str | None = None) -> list[dict]:
        matched = []
        for item in self.items:
            if item.get("stage") != stage:
                continue
            if entity is None:
                matched.append(item)
                continue
            target = item.get("entity", "*")
            if target in ("*", entity):
                matched.append(item)
        return matched

    def hit(self, item: dict) -> None:
        if item["id"] not in self.hits:
            self.hits.append(item["id"])

    def apply_parse_v1(self, text: str) -> str:
        """定宽行超长注入：把目标实体的首条数据行填充为超长。"""
        for item in self.faults_for("parse_v1"):
            if item.get("fault") != "fixed_width_overflow":
                continue
            entity = item["entity"]
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith(entity) and len(line) == 37:
                    lines[i] = line + "OVERFLOW"
                    self.hit(item)
                    break
            text = "\n".join(lines) + "\n"
        return text

    def apply_parse_v3(self, data: bytes) -> bytes:
        """未知步骤枚举注入：在步骤区追加一个未登记枚举。"""
        import struct
        for item in self.faults_for("parse_v3"):
            if item.get("fault") != "unknown_step_enum":
                continue
            (n_steps,) = struct.unpack_from("<I", data, 4)
            patched = bytearray()
            patched += data[:4]
            patched += struct.pack("<I", n_steps + 1)
            patched += data[8:8 + n_steps * 4]
            patched += struct.pack("<I", 0xDEAD)
            patched += data[8 + n_steps * 4:]
            pos = 8 + (n_steps + 1) * 4
            (n_features,) = struct.unpack_from("<I", patched, pos)
            pos += 4
            for _ in range(n_features):
                (flen,) = struct.unpack_from("<H", patched, pos)
                pos += 2 + flen
            (n_entities,) = struct.unpack_from("<I", patched, pos)
            pos += 4
            for _ in range(n_entities):
                (elen,) = struct.unpack_from("<H", patched, pos)
                pos += 2 + elen
                (offset,) = struct.unpack_from("<Q", patched, pos)
                struct.pack_into("<Q", patched, pos, offset + 4)
                pos += 12
            self.hit(item)
            data = bytes(patched)
        return data

    def apply_parse_v2(self, text: str) -> str:
        """精度标签缺失注入：删除目标实体首个特征的精度标签。"""
        for item in self.faults_for("parse_v2"):
            if item.get("fault") != "missing_precision_label":
                continue
            doc = json.loads(text)
            entity = item["entity"]
            prec = doc.get("precision", {}).get(entity, {})
            if prec:
                prec.pop(sorted(prec)[0])
                self.hit(item)
            text = json.dumps(doc, ensure_ascii=False, sort_keys=True)
        return text
