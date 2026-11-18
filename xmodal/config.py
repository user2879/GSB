"""跨模态对齐配置与交换格式规范的加载、校验与指纹。"""

import json
import os

from .errors import ErrorCode, PipelineError
from .hashing import digest_json

REQUIRED_KEYS = (
    "image_size", "block_size", "quant_levels", "text_dim",
    "orientation_bins", "align_weights", "retrieval_threshold",
    "top_k", "max_workers", "snapshot_commit_interval",
)


def load_json(path):
    if not os.path.isfile(path):
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "JSON 文件不存在: " + path)
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "JSON 结构损坏 (%s): %s" % (path, exc))


def validate_config(cfg):
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "配置缺少字段: " + ",".join(missing))
    width, height = cfg["image_size"]
    bw, bh = cfg["block_size"]
    if width % bw or height % bh:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "图像尺寸必须可被分块尺寸整除")
    if not isinstance(cfg["text_dim"], int) or cfg["text_dim"] <= 0:
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "text_dim 必须为正整数")
    weights = cfg["align_weights"]
    for key in ("mean", "gradient", "orientation", "text"):
        if key not in weights:
            raise PipelineError(ErrorCode.SAMPLE_INVALID,
                                "align_weights 缺少权重: " + key)
    if cfg["max_workers"] < 1:
        raise PipelineError(ErrorCode.SAMPLE_INVALID, "max_workers 必须 >= 1")
    if cfg["snapshot_commit_interval"] < 1:
        raise PipelineError(ErrorCode.SAMPLE_INVALID,
                            "snapshot_commit_interval 必须 >= 1")
    return cfg


def load_config(path):
    return validate_config(load_json(path))


def config_fingerprint(cfg):
    return digest_json(cfg, prefix="config")


def spec_fingerprint(spec):
    return digest_json(spec, prefix="format-spec")
