"""样本加载、校验与图像/文本特征抽取。

图像按配置分块尺寸做确定性分块统计：区域均值、Sobel 梯度强度与主方向
直方图；文本把实体、属性与断言 token 化后按维度取模哈希映射到定长向量。
"""

import json
import os

import numpy as np

from .config import load_json
from .errors import ErrorCode, PipelineError
from .hashing import digest_array, digest_bytes, digest_text
from .png_codec import decode_png

REQUIRED_TEXT_KEYS = ("scene_id", "entities", "assertions")


class FeatureRecord:
    """统一中间表示（v1/v2 读写器与打分阶段共用）。"""

    def __init__(self, sample, image_blocks, text_vector, align_weights,
                 source_version, precision_labels=None, region_truth=None):
        self.sample = sample
        # 保留输入 dtype：正常管线为 float32，转换审计需要保留 float64
        # 越界值以便在写入目标格式时触发 E_FORMAT_LOSSY。
        img = np.asarray(image_blocks)
        txt = np.asarray(text_vector)
        self.image_blocks = img if img.dtype == np.float64 \
            else img.astype(np.float32)
        self.text_vector = txt if txt.dtype == np.float64 \
            else txt.astype(np.float32)
        self.align_weights = dict(align_weights)
        self.source_version = source_version
        self.precision_labels = dict(precision_labels or {})
        self.region_truth = region_truth

    def to_meta(self):
        return {
            "sample": self.sample,
            "source_version": self.source_version,
            "block_count": int(self.image_blocks.shape[0]),
            "block_dim": int(self.image_blocks.shape[1]),
            "text_dim": int(self.text_vector.shape[0]),
            "align_weights": self.align_weights,
            "precision_labels": self.precision_labels,
        }

    def content_fingerprint(self):
        return digest_bytes(
            self.sample.encode("utf-8"),
            prefix="record",
        ) + ":" + digest_array(self.image_blocks, "img")[:16] \
            + ":" + digest_array(self.text_vector, "txt")[:16]


def list_samples(input_dir):
    if not os.path.isdir(input_dir):
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "样本目录不存在: " + input_dir)
    names = sorted(
        name for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name))
    )
    if not names:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "样本目录为空: " + input_dir)
    return names


def _validate_text(text_obj):
    if not isinstance(text_obj, dict):
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "文本描述不是 JSON 对象")
    missing = [k for k in REQUIRED_TEXT_KEYS if k not in text_obj]
    if missing:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "文本断言字段缺失: " + ",".join(missing))
    if not isinstance(text_obj["entities"], list) \
            or not isinstance(text_obj["assertions"], list):
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "entities/assertions 必须为列表")
    for entity in text_obj["entities"]:
        if not isinstance(entity, dict) or "name" not in entity:
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "实体缺少 name 字段")
    for assertion in text_obj["assertions"]:
        for key in ("subject", "predicate", "object"):
            if key not in assertion:
                raise PipelineError(
                    ErrorCode.SAMPLE_INVALID,
                    "文本断言字段缺失: assertion." + key)


def load_sample_files(input_dir, sample, image_size, require_regions=True,
                      base_dir=None):
    """加载并校验 PNG / 区域真值 / 文本，返回 (uint8 数组, 文本对象)。

    base_dir 给定时直接作为样本目录（用于快照临时副本），否则按
    input_dir/sample 拼接。
    """
    sample_dir = base_dir if base_dir is not None \
        else os.path.join(input_dir, sample)
    png_path = os.path.join(sample_dir, "image.png")
    text_path = os.path.join(sample_dir, "text.json")
    regions_path = os.path.join(sample_dir, "regions.json")
    required = (png_path, text_path) + ((regions_path,)
                                        if require_regions else ())
    for path in required:
        if not os.path.isfile(path):
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "样本文件缺失: " + os.path.basename(path))
    with open(png_path, "rb") as fh:
        blob = fh.read()
    image, _, _ = decode_png(blob, expected_size=image_size)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    text_obj = load_json(text_path)
    _validate_text(text_obj)
    if require_regions:
        regions = load_json(regions_path)
        if not isinstance(regions, list):
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "区域真值必须为列表")
    else:
        regions = []
    return image.astype(np.uint8), text_obj, regions


def _quantize(values, levels):
    levels = max(2, int(levels))
    scaled = np.round(np.clip(values, 0.0, 1.0) * (levels - 1)) / (levels - 1)
    return scaled.astype(np.float32)


def _block_stats(block, orientation_bins, quant_levels):
    gray = block.mean(axis=2).astype(np.float32)
    means = block.reshape(-1, 3).mean(axis=0) / 255.0
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    grad_mag = np.sqrt(gx * gx + gy * gy)
    grad_strength = float(grad_mag.mean()) / (255.0 * np.sqrt(2.0))
    angles = np.arctan2(gy, gx) + np.pi
    hist = np.zeros(orientation_bins, dtype=np.float32)
    idx = np.floor(angles / (2.0 * np.pi) * orientation_bins).astype(int)
    np.clip(idx, 0, orientation_bins - 1, out=idx)
    weight = grad_mag / (grad_mag.sum() + 1e-12)
    for b in range(orientation_bins):
        hist[b] = weight[idx == b].sum()
    raw = np.concatenate([
        means.astype(np.float32),
        np.array([grad_strength], dtype=np.float32),
        hist,
    ])
    return _quantize(raw, quant_levels)


def extract_image_blocks(image, cfg):
    block_w, block_h = cfg["block_size"]
    bins = int(cfg["orientation_bins"])
    height, width = image.shape[:2]
    rows = height // block_h
    cols = width // block_w
    blocks = []
    for r in range(rows):
        for c in range(cols):
            block = image[r * block_h:(r + 1) * block_h,
                          c * block_w:(c + 1) * block_w]
            blocks.append(_block_stats(block, bins, cfg["quant_levels"]))
    return np.stack(blocks).astype(np.float32)


def _token_stream(text_obj):
    tokens = []
    for entity in text_obj["entities"]:
        tokens.append("entity:" + str(entity["name"]))
        tokens.append("type:" + str(entity.get("type", "unknown")))
        for attr in entity.get("attributes", []):
            tokens.append("attr:{}={}".format(attr.get("name"),
                                              attr.get("value")))
    for assertion in text_obj["assertions"]:
        tokens.append("assert:{}:{}:{}".format(
            assertion["subject"], assertion["predicate"],
            assertion["object"]))
    return tokens


def extract_text_vector(text_obj, text_dim):
    vector = np.zeros(text_dim, dtype=np.float32)
    for token in _token_stream(text_obj):
        digest = digest_text(token, prefix="text-token")
        slot = int(digest[:8], 16) % text_dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[slot] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector.astype(np.float32)


def extract_sample(input_dir, sample, cfg, source_version="raw"):
    image, text_obj, regions = load_sample_files(
        input_dir, sample, cfg["image_size"])
    image_blocks = extract_image_blocks(image, cfg)
    text_vector = extract_text_vector(text_obj, cfg["text_dim"])
    return FeatureRecord(
        sample=sample,
        image_blocks=image_blocks,
        text_vector=text_vector,
        align_weights=cfg["align_weights"],
        source_version=source_version,
        region_truth=regions,
    )


def raw_input_fingerprints(input_dir, sample, image_size):
    """直接对原始输入取内容指纹，供增量失效判定。"""
    sample_dir = os.path.join(input_dir, sample)
    with open(os.path.join(sample_dir, "image.png"), "rb") as fh:
        png_digest = digest_bytes(fh.read(), prefix="png")
    with open(os.path.join(sample_dir, "text.json"), "rb") as fh:
        text_digest = digest_bytes(fh.read(), prefix="text")
    with open(os.path.join(sample_dir, "regions.json"), "rb") as fh:
        region_digest = digest_bytes(fh.read(), prefix="regions")
    return {
        "png": png_digest,
        "text": text_digest,
        "regions": region_digest,
        "size": "x".join(str(x) for x in image_size),
    }
