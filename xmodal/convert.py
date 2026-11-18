"""v1 <-> v2 版本兼容转换与损失审计。

v1->v2: float32 数值转 JSON 浮点并标注 float32-roundtrip 精度标签；
v2->v1: JSON 浮点四舍五入写 float32，截断残差、精度标签与来源标记写入
旁路 .sidecar.json；超出 float32 表示范围或非有限值输出 E_FORMAT_LOSSY
并记录具体特征键。
"""

import json
import os

import numpy as np

from .errors import ErrorCode, PipelineError
from .features import FeatureRecord
from .formats import (V1_VERSION, V2_VERSION, read_v1, read_v2, write_v1,
                      write_v2)

F32_MAX = float(np.finfo(np.float32).max)
F32_MIN_NORM = float(np.finfo(np.float32).tiny)


def sidecar_path(pack_path):
    return pack_path + ".sidecar.json"


def _audit_value(key, value, lossy_keys):
    value = float(value)
    if not np.isfinite(value) or abs(value) > F32_MAX:
        lossy_keys.append(key)
        return False
    return True


def _audit_record(record):
    lossy_keys = []
    for row, values in enumerate(record.image_blocks):
        for col, value in enumerate(values):
            _audit_value("image.block_%03d[%d]" % (row, col), value,
                         lossy_keys)
    for col, value in enumerate(record.text_vector):
        _audit_value("text.vector[%d]" % col, value, lossy_keys)
    return lossy_keys


def convert_v1_to_v2(v1_path, v2_path):
    record = read_v1(v1_path)
    record.precision_labels = dict(record.precision_labels)
    record.precision_labels["image"] = "float32-roundtrip"
    record.precision_labels["text"] = "float32-roundtrip"
    record.source_version = V2_VERSION
    write_v2(v2_path, record)
    residual = _roundtrip_residual(record)
    sidecar = {
        "direction": "v1_to_v2",
        "sample": record.sample,
        "source": "v1",
        "precision_labels": record.precision_labels,
        "truncation_residual": residual,
        "lossy_keys": [],
    }
    _write_sidecar(sidecar_path(v2_path), sidecar)
    return sidecar


def convert_v2_to_v1(v2_path, v1_path):
    record = read_v2(v2_path)
    lossy_keys = _audit_record(record)
    if lossy_keys:
        raise PipelineError(
            ErrorCode.FORMAT_LOSSY,
            "存在无法以 float32 表达的特征: " + ",".join(lossy_keys[:8])
            + (" ..." if len(lossy_keys) > 8 else ""),
        )
    original_image = record.image_blocks.astype(np.float64).copy()
    original_text = record.text_vector.astype(np.float64).copy()
    rounded = FeatureRecord(
        record.sample,
        original_image.astype(np.float32),
        original_text.astype(np.float32),
        record.align_weights,
        V1_VERSION,
        precision_labels={
            "image": "float32-rounded",
            "text": "float32-rounded",
        },
    )
    write_v1(v1_path, rounded)
    residual_img = float(
        np.max(np.abs(original_image - rounded.image_blocks.astype(
            np.float64)))) if original_image.size else 0.0
    residual_txt = float(
        np.max(np.abs(original_text - rounded.text_vector.astype(
            np.float64)))) if original_text.size else 0.0
    sidecar = {
        "direction": "v2_to_v1",
        "sample": record.sample,
        "source": "v2",
        "precision_labels": rounded.precision_labels,
        "truncation_residual": {
            "image_max_abs": round(residual_img, 12),
            "text_max_abs": round(residual_txt, 12),
        },
        "lossy_keys": [],
    }
    _write_sidecar(sidecar_path(v1_path), sidecar)
    return sidecar


def _roundtrip_residual(record):
    image = record.image_blocks.astype(np.float64)
    text = record.text_vector.astype(np.float64)
    image_res = float(np.max(np.abs(
        image - image.astype(np.float32).astype(np.float64)))) \
        if image.size else 0.0
    text_res = float(np.max(np.abs(
        text - text.astype(np.float32).astype(np.float64)))) \
        if text.size else 0.0
    return {
        "image_max_abs": round(image_res, 12),
        "text_max_abs": round(text_res, 12),
    }


def _write_sidecar(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, ensure_ascii=False,
                  indent=2)
    os.replace(tmp, path)


def cross_format_deviation(records_a, records_b):
    """比较两份统一表示（不同格式读回）的数值偏差摘要。"""
    summary = []
    for sample in sorted(records_a):
        ra = records_a[sample]
        rb = records_b[sample]
        img_diff = np.max(np.abs(
            ra.image_blocks.astype(np.float64)
            - rb.image_blocks.astype(np.float64)))
        txt_diff = np.max(np.abs(
            ra.text_vector.astype(np.float64)
            - rb.text_vector.astype(np.float64)))
        weight_diff = max(
            abs(float(ra.align_weights[k]) - float(rb.align_weights[k]))
            for k in ra.align_weights)
        summary.append({
            "sample": sample,
            "image_max_abs": round(float(img_diff), 12),
            "text_max_abs": round(float(txt_diff), 12),
            "weight_max_abs": round(float(weight_diff), 12),
            "name_restored": ra.sample == rb.sample,
            "weights_restored": ra.align_weights == rb.align_weights,
        })
    return summary
