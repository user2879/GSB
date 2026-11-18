"""v1 紧凑二进制特征包与 v2 文本 JSON 特征包读写器。

两种格式解析为统一的 FeatureRecord，同一中间表示写入任一格式再读回，
除格式精度标签可解释的 float32 舍入外逐字段一致。

v1 二进制布局（小端）:
  头部 48B:
    magic "XMF1\x00\x00\x00\x00" (8) | version uint16 (2) | reserved (2) |
    block_count uint32 (4) | block_dim uint32 (4) | text_dim uint32 (4) |
    n_weights uint32 (4) | n_labels uint32 (4) | sample_len uint32 (4) |
    table_offset uint32 (4) | data_offset uint32 (4)
  偏移表 (table_offset 起, 每项 uint32, 共 6 项):
    image_off image_len text_off text_len weight_off weight_len
  数据区 (data_offset 起):
    sample 名原始字节 | float32 图像块矩阵 | float32 文本向量 |
    权重名(\\0 分隔) + float32 权重值 | 精度标签 JSON
"""

import json
import os
import struct

import numpy as np

from .errors import ErrorCode, PipelineError
from .features import FeatureRecord
from .hashing import digest_bytes

V1_MAGIC = b"XMF1\x00\x00\x00\x00"
V1_VERSION = 1
V2_VERSION = 2
HEADER_SIZE = 48
TABLE_ENTRIES = 6


def _check_finite(array, where):
    if not np.all(np.isfinite(array)):
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "float32 特征包含非有限值: " + where)


def write_v1(path, record):
    """FeatureRecord -> v1 二进制包。"""
    image = np.asarray(record.image_blocks, dtype=np.float32)
    text = np.asarray(record.text_vector, dtype=np.float32)
    _check_finite(image, record.sample + "/image")
    _check_finite(text, record.sample + "/text")
    sample_bytes = record.sample.encode("utf-8")
    weight_names = sorted(record.align_weights)
    weight_blob = bytearray()
    for name in weight_names:
        value = float(record.align_weights[name])
        if not np.isfinite(value):
            raise PipelineError(ErrorCode.FORMAT_LOSSY,
                                "对齐权重无法以 float32 表达: " + name)
        weight_blob.extend(name.encode("utf-8"))
        weight_blob.append(0)
        weight_blob.extend(struct.pack("<f", np.float32(value)))
    labels_blob = json.dumps(
        record.precision_labels, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    table_offset = HEADER_SIZE
    data_offset = table_offset + TABLE_ENTRIES * 4
    cursor = data_offset + len(sample_bytes)

    def place(blob):
        nonlocal cursor
        off, length = cursor, len(blob)
        cursor += length
        return off, length

    image_blob = np.ascontiguousarray(image, dtype="<f4").tobytes()
    text_blob = np.ascontiguousarray(text, dtype="<f4").tobytes()
    image_off, image_len = place(image_blob)
    text_off, text_len = place(text_blob)
    weight_off, weight_len = place(bytes(weight_blob))
    label_off, label_len = place(labels_blob)

    header = struct.pack(
        "<8sHHIIIIIIIII", V1_MAGIC, V1_VERSION, 0, 0,
        int(image.shape[0]), int(image.shape[1]), int(text.shape[0]),
        len(weight_names), len(record.precision_labels), len(sample_bytes),
        table_offset, data_offset,
    )
    table = struct.pack(
        "<6I", image_off, image_len, text_off, text_len,
        weight_off, weight_len,
    )
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(header)
        fh.write(table)
        fh.write(sample_bytes)
        fh.write(image_blob)
        fh.write(text_blob)
        fh.write(bytes(weight_blob))
        fh.write(labels_blob)
    os.replace(tmp, path)


def read_v1(path, expected_sample=None, validate_offsets=True):
    """v1 二进制包 -> FeatureRecord。"""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < HEADER_SIZE:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "v1 包长度小于头部: " + path)
    unpacked = struct.unpack("<8sHHIIIIIIIII", blob[:HEADER_SIZE])
    (magic, version, _reserved, _reserved2, block_count, block_dim,
     text_dim, n_weights, _n_labels, sample_len, table_offset,
     data_offset) = unpacked
    if magic != V1_MAGIC:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "格式版本未知或 magic 非法: %r" % magic[:4])
    if version != V1_VERSION:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "格式版本未知: v%d" % version)
    table_end = table_offset + TABLE_ENTRIES * 4
    if table_end > len(blob):
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "二进制偏移表越界: 表尾超出文件")
    (image_off, image_len, text_off, text_len,
     weight_off, weight_len) = struct.unpack(
        "<6I", blob[table_offset:table_end])
    spans = [(image_off, image_len, "image"), (text_off, text_len, "text"),
             (weight_off, weight_len, "weights")]
    if validate_offsets:
        if data_offset + sample_len > len(blob):
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "二进制偏移表越界: 样本名超出文件")
        for off, length, name in spans:
            if off < data_offset or off + length > len(blob):
                raise PipelineError(
                    ErrorCode.SAMPLE_INVALID,
                    "二进制偏移表越界: %s 段超出文件" % name)
    sample = blob[data_offset:data_offset + sample_len].decode("utf-8")
    if expected_sample is not None and sample != expected_sample:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "v1 包样本名不匹配: %s != %s"
                            % (sample, expected_sample))
    image = np.frombuffer(
        blob, dtype="<f4", count=block_count * block_dim,
        offset=image_off).reshape(block_count, block_dim).astype(np.float32)
    text = np.frombuffer(
        blob, dtype="<f4", count=text_dim,
        offset=text_off).astype(np.float32)
    _check_finite(image, sample + "/image")
    _check_finite(text, sample + "/text")
    weight_blob = blob[weight_off:weight_off + weight_len]
    weights = {}
    cursor_w = 0
    for _ in range(n_weights):
        end = weight_blob.index(b"\x00", cursor_w)
        name = weight_blob[cursor_w:end].decode("utf-8")
        cursor_w = end + 1
        (value,) = struct.unpack("<f", weight_blob[cursor_w:cursor_w + 4])
        weights[name] = float(value)
        cursor_w += 4
    label_start = weight_off + weight_len
    labels = {}
    if label_start < len(blob):
        labels = json.loads(blob[label_start:].decode("utf-8"))
    return FeatureRecord(sample, image, text, weights, V1_VERSION, labels)


def write_v2(path, record):
    """FeatureRecord -> v2 JSON 特征包。"""
    image = np.asarray(record.image_blocks, dtype=np.float64)
    text = np.asarray(record.text_vector, dtype=np.float64)
    payload = {
        "format_version": V2_VERSION,
        "sample": record.sample,
        "align_weights": {k: float(v)
                          for k, v in sorted(record.align_weights.items())},
        "precision_labels": dict(sorted(record.precision_labels.items())),
        "features": {
            "image.block_%03d" % i: [float(x) for x in row]
            for i, row in enumerate(image)
        },
        "features_text": {
            "text.vector": [float(x) for x in text],
        },
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, ensure_ascii=False,
                  separators=(",", ":"), allow_nan=False)
    os.replace(tmp, path)


def read_v2(path, expected_sample=None):
    """v2 JSON 特征包 -> FeatureRecord。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "v2 包 JSON 结构损坏: " + str(exc))
    if not isinstance(payload, dict) or \
            payload.get("format_version") != V2_VERSION:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "格式版本未知: v2 包缺少合法 format_version")
    sample = payload.get("sample")
    if not isinstance(sample, str):
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "v2 包缺少样本名")
    if expected_sample is not None and sample != expected_sample:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "v2 包样本名不匹配: %s != %s"
                            % (sample, expected_sample))
    features = payload.get("features", {})
    block_keys = sorted(features)
    if not block_keys:
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "v2 包缺少图像块特征")
    image = np.array([[float(v) for v in features[k]] for k in block_keys],
                     dtype=np.float64)
    text = np.array(
        [float(v) for v in payload["features_text"]["text.vector"]],
        dtype=np.float64)
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(text)):
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "v2 特征包含非有限值: " + sample)
    weights = {k: float(v) for k, v in payload.get(
        "align_weights", {}).items()}
    labels = payload.get("precision_labels", {})
    return FeatureRecord(sample, image, text, weights, V2_VERSION, labels)


def write_pack(path, record, version):
    if version == V1_VERSION:
        write_v1(path, record)
    elif version == V2_VERSION:
        write_v2(path, record)
    else:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "格式版本未知: %r" % version)


def read_pack(path, expected_sample=None):
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == V1_MAGIC[:4]:
        return read_v1(path, expected_sample)
    return read_v2(path, expected_sample)


def pack_fingerprint(path):
    with open(path, "rb") as fh:
        return digest_bytes(fh.read(), prefix="pack")
