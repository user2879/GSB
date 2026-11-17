"""v1 / v2 / v3 三版本管道文件的解析与序列化。

v1 定宽文本行：
    第 0 行  "V1PIPE"
    第 1 行  "STEPS|" 逗号分隔步骤名（字符串数组，顺序即执行顺序）
    数据行  entity_id(12) | feature(16) | permille(8，右对齐有符号整数) | missing(1)
    末行    "END"
    数据行宽度必须为 12+16+8+1 = 37，否则 E_FIELD_WIDTH。

v2 JSON 对象：
    {"version": 2, "entities": {eid: {feat: float|null}},
     "precision": {eid: {feat: label}}, "steps": [{"name", "params"}]}
    缺键 = 缺失；显式 null = 缺失 + 显式空值标记（两者语义不同）。
    非空数值缺精度标签 -> E_PRECISION_MISSING（记失败清单，默认 double）。

v3 二进制分区文件（小端）：
    magic "RPV3" | u32 步骤数 | 步骤枚举 u32*n
    u32 特征名数 | (u16 长度 + UTF-8 名)*n
    u32 实体数 | (u16 ID 长 + ID + u64 偏移 + u32 记录数)*n
    偏移处记录块：(u16 特征索引 + u8 缺失位 + float32 值)*记录数
    偏移越界 -> E_PARTITION_OOB；未知步骤枚举 -> E_STEP_UNKNOWN。
"""
from __future__ import annotations

import json
import struct

import numpy as np

from .errors import (
    E_FIELD_WIDTH,
    E_PARTITION_OOB,
    E_PRECISION_MISSING,
    E_STEP_MAP_MISSING,
    E_STEP_UNKNOWN,
    Failure,
    PipelineError,
)
from .ir import (
    PRECISION_DOUBLE,
    PRECISION_FLOAT32,
    PRECISION_PERMILLE,
    FeatureSet,
    IR,
    Step,
)

V1_MAGIC = "V1PIPE"
V1_END = "END"
V1_W_ENTITY = 12
V1_W_FEATURE = 16
V1_W_VALUE = 8
V1_W_MISSING = 1
V1_LINE_WIDTH = V1_W_ENTITY + V1_W_FEATURE + V1_W_VALUE + V1_W_MISSING

V3_MAGIC = b"RPV3"
V3_RECORD = struct.Struct("<HBf")
V3_UNKNOWN_ENUM = 0xFFFF


def _all_entities(ir: IR) -> list[str]:
    ents = set(ir.features.values) | set(ir.features.missing)
    return sorted(ents)


def _all_features(ir: IR, entity: str) -> list[str]:
    feats = set(ir.features.values.get(entity, {})) | set(ir.features.missing.get(entity, set()))
    return sorted(feats)


def _permille_to_float(p: int) -> float:
    return p / 1000.0


def parse_v1(text: str, failures: list[Failure]) -> IR:
    lines = text.splitlines()
    if not lines or lines[0].strip() != V1_MAGIC:
        raise PipelineError("E_BAD_MAGIC", "v1 文件缺少 V1PIPE 头")
    steps: list[Step] = []
    data_start = 1
    if len(lines) > 1 and lines[1].startswith("STEPS|"):
        names = [n for n in lines[1][len("STEPS|"):].split(",") if n]
        steps = [Step(name=n) for n in names]
        data_start = 2
    fs = FeatureSet()
    for lineno, line in enumerate(lines[data_start:], start=data_start):
        if line.strip() == V1_END:
            break
        entity_guess = line[:V1_W_ENTITY].strip() or None
        if len(line) != V1_LINE_WIDTH:
            failures.append(Failure(
                entity_id=entity_guess, stage="parse_v1", code=E_FIELD_WIDTH,
                exc_type="FieldWidthError",
                detail=f"行 {lineno} 宽度 {len(line)} != {V1_LINE_WIDTH}"))
            continue
        entity = line[:V1_W_ENTITY].strip()
        feature = line[V1_W_ENTITY:V1_W_ENTITY + V1_W_FEATURE].strip()
        raw_val = line[V1_W_ENTITY + V1_W_FEATURE:V1_W_ENTITY + V1_W_FEATURE + V1_W_VALUE]
        miss_flag = line[V1_LINE_WIDTH - 1]
        try:
            permille = int(raw_val)
        except ValueError:
            failures.append(Failure(
                entity_id=entity, stage="parse_v1", code=E_FIELD_WIDTH,
                exc_type="ValueError", detail=f"行 {lineno} 千分比字段非整数: {raw_val!r}"))
            continue
        fs.values.setdefault(entity, {})[feature] = _permille_to_float(permille)
        fs.precision.setdefault(entity, {})[feature] = PRECISION_PERMILLE
        if miss_flag == "1":
            fs.missing.setdefault(entity, set()).add(feature)
    return IR(features=fs, steps=steps, source_version="v1")


def serialize_v1(ir: IR) -> str:
    out = [V1_MAGIC, "STEPS|" + ",".join(s.name for s in ir.steps)]
    for entity in _all_entities(ir):
        for feature in _all_features(ir, entity):
            value = ir.features.values.get(entity, {}).get(feature, 0.0)
            permille = int(round(value * 1000))
            miss = "1" if feature in ir.features.missing.get(entity, set()) else "0"
            out.append(f"{entity:<{V1_W_ENTITY}}{feature:<{V1_W_FEATURE}}"
                       f"{permille:>{V1_W_VALUE}}{miss}")
    out.append(V1_END)
    return "\n".join(out) + "\n"


def parse_v2(text: str, failures: list[Failure]) -> IR:
    doc = json.loads(text)
    fs = FeatureSet()
    precision_doc = doc.get("precision", {})
    for entity in sorted(doc.get("entities", {})):
        feats = doc["entities"][entity] or {}
        prec = precision_doc.get(entity, {})
        for feature in sorted(feats):
            value = feats[feature]
            if value is None:
                fs.missing.setdefault(entity, set()).add(feature)
                fs.explicit_null.setdefault(entity, set()).add(feature)
                continue
            fs.values.setdefault(entity, {})[feature] = float(value)
            label = prec.get(feature)
            if label is None:
                failures.append(Failure(
                    entity_id=entity, stage="parse_v2", code=E_PRECISION_MISSING,
                    exc_type="PrecisionLabelMissing",
                    detail=f"特征 {feature} 缺精度标签，按 double 处理"))
                label = PRECISION_DOUBLE
            fs.precision.setdefault(entity, {})[feature] = label
    steps = [Step(name=s["name"], params=dict(s.get("params", {})))
             for s in doc.get("steps", [])]
    return IR(features=fs, steps=steps, source_version="v2")


def serialize_v2(ir: IR) -> str:
    entities: dict[str, dict] = {}
    precision: dict[str, dict] = {}
    for entity in _all_entities(ir):
        feats: dict[str, object] = {}
        for feature in _all_features(ir, entity):
            if feature in ir.features.missing.get(entity, set()):
                feats[feature] = None
            else:
                feats[feature] = ir.features.values[entity][feature]
        entities[entity] = feats
        precision[entity] = dict(sorted(ir.features.precision.get(entity, {}).items()))
    doc = {
        "version": 2,
        "entities": entities,
        "precision": precision,
        "steps": [s.to_dict() for s in ir.steps],
    }
    return json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def parse_v3(data: bytes, name_map: dict[str, int], failures: list[Failure]) -> IR:
    if data[:4] != V3_MAGIC:
        raise PipelineError("E_BAD_MAGIC", "v3 文件缺少 RPV3 头")
    reverse_map = {code: name for name, code in name_map.items()}
    pos = 4
    (n_steps,) = struct.unpack_from("<I", data, pos)
    pos += 4
    steps: list[Step] = []
    for _ in range(n_steps):
        (code,) = struct.unpack_from("<I", data, pos)
        pos += 4
        name = reverse_map.get(code)
        if name is None:
            failures.append(Failure(
                entity_id=None, stage="parse_v3", code=E_STEP_UNKNOWN,
                exc_type="UnknownStepEnum",
                detail=f"未知步骤枚举 {code}，保留为占位步骤"))
            name = f"__unknown_enum_{code}"
        steps.append(Step(name=name))
    (n_features,) = struct.unpack_from("<I", data, pos)
    pos += 4
    feature_names: list[str] = []
    for _ in range(n_features):
        (flen,) = struct.unpack_from("<H", data, pos)
        pos += 2
        feature_names.append(data[pos:pos + flen].decode("utf-8"))
        pos += flen
    (n_entities,) = struct.unpack_from("<I", data, pos)
    pos += 4
    entries = []
    for _ in range(n_entities):
        (elen,) = struct.unpack_from("<H", data, pos)
        pos += 2
        entity = data[pos:pos + elen].decode("utf-8")
        pos += elen
        offset, n_records = struct.unpack_from("<QI", data, pos)
        pos += 12
        entries.append((entity, offset, n_records))
    fs = FeatureSet()
    for entity, offset, n_records in entries:
        end = offset + n_records * V3_RECORD.size
        if end > len(data):
            failures.append(Failure(
                entity_id=entity, stage="parse_v3", code=E_PARTITION_OOB,
                exc_type="PartitionOutOfBounds",
                detail=f"偏移 {offset}+{n_records * V3_RECORD.size} 超出文件长度 {len(data)}"))
            continue
        for i in range(n_records):
            feat_idx, miss_flag, value = V3_RECORD.unpack_from(data, offset + i * V3_RECORD.size)
            if feat_idx >= len(feature_names):
                failures.append(Failure(
                    entity_id=entity, stage="parse_v3", code=E_PARTITION_OOB,
                    exc_type="FeatureIndexOOB",
                    detail=f"特征索引 {feat_idx} 越界"))
                continue
            feature = feature_names[feat_idx]
            fs.values.setdefault(entity, {})[feature] = float(np.float32(value))
            fs.precision.setdefault(entity, {})[feature] = PRECISION_FLOAT32
            if miss_flag:
                fs.missing.setdefault(entity, set()).add(feature)
    return IR(features=fs, steps=steps, source_version="v3")


def serialize_v3(ir: IR, name_map: dict[str, int],
                 failures: list[Failure] | None = None) -> bytes:
    feature_names = sorted({f for e in _all_entities(ir) for f in _all_features(ir, e)})
    feat_index = {name: i for i, name in enumerate(feature_names)}
    header = bytearray()
    header += V3_MAGIC
    header += struct.pack("<I", len(ir.steps))
    for step in ir.steps:
        code = name_map.get(step.name)
        if code is None:
            if failures is not None:
                failures.append(Failure(
                    entity_id=None, stage="serialize_v3", code=E_STEP_MAP_MISSING,
                    exc_type="StepMapMissing",
                    detail=f"步骤 {step.name} 不在名称映射表，写入未知枚举标记，IR 中保留"))
            code = V3_UNKNOWN_ENUM
        header += struct.pack("<I", code)
    header += struct.pack("<I", len(feature_names))
    for name in feature_names:
        encoded = name.encode("utf-8")
        header += struct.pack("<H", len(encoded)) + encoded
    entities = _all_entities(ir)
    offset_table = bytearray()
    offset_table += struct.pack("<I", len(entities))
    blocks = bytearray()
    base = len(header) + len(offset_table) + sum(
        2 + len(e.encode("utf-8")) + 12 for e in entities)
    cursor = base
    for entity in entities:
        encoded = entity.encode("utf-8")
        records = bytearray()
        for feature in _all_features(ir, entity):
            miss = 1 if feature in ir.features.missing.get(entity, set()) else 0
            value = ir.features.values.get(entity, {}).get(feature, 0.0)
            records += V3_RECORD.pack(feat_index[feature], miss,
                                      float(np.float32(value)))
        offset_table += struct.pack("<H", len(encoded)) + encoded
        offset_table += struct.pack("<QI", cursor, len(records) // V3_RECORD.size)
        blocks += records
        cursor += len(records)
    return bytes(header + offset_table + blocks)
