"""融合配置加载、校验与配置指纹派生。"""

import hashlib
import json

import yaml

from .errors import E_CONFIG_INVALID, PipelineError
from .fixedpoint import float_to_fixed, scale_of

_REQUIRED_INT_KEYS = (
    "manifest_version", "block_size", "visual_dim", "text_dim",
    "frac_bits", "quant_levels", "max_workers", "global_mac_budget",
    "per_sample_quota", "cache_resident_budget", "snapshot_commit_interval",
    "edge_threshold",
)


class FusionConfig:
    def __init__(self, raw):
        self.raw = dict(raw)
        fb = raw["frac_bits"]
        self.frac_bits = fb
        self.scale = scale_of(fb)
        self.manifest_version = raw["manifest_version"]
        self.image_width, self.image_height = (int(v) for v in raw["image_size"])
        self.block_size = raw["block_size"]
        self.visual_dim = raw["visual_dim"]
        self.text_dim = raw["text_dim"]
        self.w_visual = float_to_fixed(float(raw["fusion_weights"]["visual"]), fb)
        self.w_text = float_to_fixed(float(raw["fusion_weights"]["text"]), fb)
        self.temperature = float_to_fixed(float(raw["fusion_temperature"]), fb)
        self.quant_levels = raw["quant_levels"]
        self.quant_range = float_to_fixed(float(raw["quant_range"]), fb)
        self.max_workers = raw["max_workers"]
        self.global_mac_budget = raw["global_mac_budget"]
        self.per_sample_quota = raw["per_sample_quota"]
        self.cache_resident_budget = raw["cache_resident_budget"]
        self.snapshot_commit_interval = raw["snapshot_commit_interval"]
        self.edge_threshold = raw["edge_threshold"]
        self.score_clip_lo = float_to_fixed(float(raw["score_clip"][0]), fb)
        self.score_clip_hi = float_to_fixed(float(raw["score_clip"][1]), fb)
        self.decision_threshold = float_to_fixed(float(raw["decision_threshold"]), fb)
        self.conflict_num = int(raw["conflict_reduction"][0])
        self.conflict_den = int(raw["conflict_reduction"][1])
        self.default_fill_seed_tag = str(raw["default_fill_seed_tag"])
        self.fingerprint = self._fingerprint()

    def _canonical(self):
        """定点化后的规范字典，用于配置指纹（与运行时预算覆盖无关）。"""
        return {
            "manifest_version": self.manifest_version,
            "image_size": [self.image_width, self.image_height],
            "block_size": self.block_size,
            "visual_dim": self.visual_dim,
            "text_dim": self.text_dim,
            "w_visual": self.w_visual,
            "w_text": self.w_text,
            "temperature": self.temperature,
            "frac_bits": self.frac_bits,
            "quant_levels": self.quant_levels,
            "quant_range": self.quant_range,
            "max_workers": self.max_workers,
            "global_mac_budget": self.global_mac_budget,
            "per_sample_quota": self.per_sample_quota,
            "cache_resident_budget": self.cache_resident_budget,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "edge_threshold": self.edge_threshold,
            "score_clip": [self.score_clip_lo, self.score_clip_hi],
            "decision_threshold": self.decision_threshold,
            "conflict_reduction": [self.conflict_num, self.conflict_den],
            "default_fill_seed_tag": self.default_fill_seed_tag,
        }

    def _fingerprint(self):
        blob = json.dumps(self._canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def blocks_per_side(self):
        return self.image_width // self.block_size

    @property
    def n_blocks(self):
        n = self.blocks_per_side
        return n * n

    @property
    def n_text_blocks(self):
        return -(-self.text_dim // self.block_size)  # 向上取整


def _fail(message):
    raise PipelineError(E_CONFIG_INVALID, "load_validate", message)


def validate_raw(raw):
    """配置校验，任何不合法均抛 E_CONFIG_INVALID。"""
    if not isinstance(raw, dict):
        _fail("config root must be a mapping")
    for key in _REQUIRED_INT_KEYS:
        if key not in raw:
            _fail(f"missing config key: {key}")
        if not isinstance(raw[key], int) or isinstance(raw[key], bool):
            _fail(f"config key {key} must be an integer")
    for key in ("block_size", "visual_dim", "text_dim", "max_workers",
                "per_sample_quota", "cache_resident_budget",
                "snapshot_commit_interval", "quant_levels"):
        if raw[key] <= 0:
            _fail(f"config key {key} must be a positive integer")
    if raw["global_mac_budget"] < 0:
        _fail("global_mac_budget must be >= 0")
    if not (8 <= raw["frac_bits"] <= 30):
        _fail("frac_bits must be in [8, 30]")
    if raw["quant_levels"] < 2:
        _fail("quant_levels must be >= 2")
    size = raw.get("image_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(not isinstance(v, int) or v <= 0 for v in size)):
        _fail("image_size must be a pair of positive integers")
    if size[0] % raw["block_size"] != 0 or size[1] % raw["block_size"] != 0:
        _fail("block_size must divide image edges")
    weights = raw.get("fusion_weights")
    if not isinstance(weights, dict):
        _fail("fusion_weights must be a mapping")
    for name in ("visual", "text"):
        if float(weights.get(name, -1)) < 0:
            _fail(f"fusion_weights.{name} must be >= 0")
    if float(raw.get("fusion_temperature", 0)) <= 0:
        _fail("fusion_temperature must be > 0")
    clip = raw.get("score_clip")
    if not isinstance(clip, list) or len(clip) != 2 or float(clip[0]) > float(clip[1]):
        _fail("score_clip must be [lo, hi] with lo <= hi")
    red = raw.get("conflict_reduction")
    if (not isinstance(red, list) or len(red) != 2
            or int(red[0]) <= 0 or int(red[1]) <= 0 or int(red[0]) > int(red[1])):
        _fail("conflict_reduction must be [num, den] with 0 < num <= den")


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    validate_raw(raw)
    return FusionConfig(raw)
