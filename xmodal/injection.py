"""异常注入配置与真实输入故障。

注入项编号 (INJ-xxx) 贯穿故障应用与最终报告，用于证明注入命中并映射到
业务码。故障作用于输入副本，不污染原始样本目录。
"""

import json
import struct

import numpy as np

from .errors import ErrorCode
from .formats import HEADER_SIZE

INJECTION_TYPES = (
    "truncate_png",
    "missing_text_field",
    "offset_table_oob",
    "non_finite_float",
    "threshold_boundary",
    "unknown_format_version",
    "stage_delay",
)

CODE_BY_TYPE = {
    "truncate_png": ErrorCode.SAMPLE_INVALID,
    "missing_text_field": ErrorCode.SAMPLE_INVALID,
    "offset_table_oob": ErrorCode.SAMPLE_INVALID,
    "non_finite_float": ErrorCode.SAMPLE_INVALID,
    "threshold_boundary": ErrorCode.RETRIEVAL_LOW_SCORE,
    "unknown_format_version": ErrorCode.SAMPLE_INVALID,
    "stage_delay": None,
}


def load_injection_config(path):
    if not path:
        return {"injections": []}
    with open(path, "rb") as fh:
        cfg = json.loads(fh.read().decode("utf-8"))
    items = cfg.get("injections", [])
    for item in items:
        if item["type"] not in INJECTION_TYPES:
            raise ValueError("未知注入类型: " + item["type"])
        item.setdefault("id", "INJ-%03d" % (items.index(item) + 1))
    return cfg


def select(items, sample=None, stage=None, inject_type=None):
    out = []
    for item in items:
        if sample is not None and item.get("sample") != sample:
            continue
        if stage is not None and item.get("stage") != stage:
            continue
        if inject_type is not None and item.get("type") != inject_type:
            continue
        out.append(item)
    return out


def apply_file_faults(items, sample, stage, file_path):
    """在阶段读取前对单个输入文件施加真实字节/JSON 故障。"""
    hits = []
    for item in select(items, sample, stage):
        itype = item["type"]
        if itype == "truncate_png" and file_path.endswith("image.png"):
            with open(file_path, "rb") as fh:
                blob = fh.read()
            cut = max(8, int(len(blob) * item.get("keep_ratio", 0.5)))
            with open(file_path, "wb") as fh:
                fh.write(blob[:cut])
            hits.append(item["id"])
        elif itype == "missing_text_field" and file_path.endswith("text.json"):
            with open(file_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            field = item.get("field", "assertions")
            payload.pop(field, None)
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True)
            hits.append(item["id"])
        elif itype == "offset_table_oob" and file_path.endswith(".xmf1"):
            with open(file_path, "rb") as fh:
                blob = bytearray(fh.read())
            table = HEADER_SIZE
            huge = (len(blob) + 4096).to_bytes(4, "little")
            blob[table:table + 4] = huge
            with open(file_path, "wb") as fh:
                fh.write(blob)
            hits.append(item["id"])
        elif itype == "non_finite_float" and file_path.endswith(".xmf1"):
            with open(file_path, "rb") as fh:
                blob = bytearray(fh.read())
            (image_off,) = struct.unpack("<I", bytes(blob[HEADER_SIZE:
                                                          HEADER_SIZE + 4]))
            nan = struct.unpack("<I", np.float32(np.nan).tobytes())[0]
            blob[image_off:image_off + 4] = struct.pack("<I", nan)
            with open(file_path, "wb") as fh:
                fh.write(blob)
            hits.append(item["id"])
        elif itype == "unknown_format_version" and file_path.endswith(".xmf1"):
            with open(file_path, "rb") as src:
                blob = bytearray(src.read())
            blob[8:10] = struct.pack("<H", 99)
            with open(file_path, "wb") as fh:
                fh.write(blob)
            hits.append(item["id"])
    return hits


def boundary_threshold_item(items):
    found = select(items, inject_type="threshold_boundary")
    return found[0] if found else None


def delay_item(items, stage):
    found = select(items, stage=stage, inject_type="stage_delay")
    return found[0] if found else None
