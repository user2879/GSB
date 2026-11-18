"""基于内容指纹的缓存键与精确重算范围判定。

缓存键由上游内容指纹、配置指纹与格式版本号派生；样本级输入或配置变更
只会使受影响样本及其下游阶段失效，未失效产物直接复用。
"""

from .config import config_fingerprint, spec_fingerprint
from .hashing import cache_key, combine
from .features import raw_input_fingerprints

STAGES = ("validate", "extract", "align", "convert", "retrieve")


def extract_config_fingerprint(cfg):
    """抽取阶段相关配置：分块、量化、文本维度、方向直方图桶数。"""
    return combine([
        str(cfg["image_size"]),
        str(cfg["block_size"]),
        str(cfg["quant_levels"]),
        str(cfg["text_dim"]),
        str(cfg["orientation_bins"]),
    ], prefix="extract-cfg")


def align_config_fingerprint(cfg):
    return combine([str(sorted(cfg["align_weights"].items())),
                    str(cfg["orientation_bins"])], prefix="align-cfg")


def retrieve_config_fingerprint(cfg):
    return combine([str(cfg["retrieval_threshold"]), str(cfg["top_k"])],
                   prefix="retrieve-cfg")


class CachePlanner:
    """根据当前输入/配置与历史清单，决定每个阶段的复用与重算集合。"""

    def __init__(self, cfg, spec, format_version, input_dir, image_size):
        self.cfg = cfg
        self.spec = spec
        self.format_version = format_version
        self.input_dir = input_dir
        self.image_size = image_size
        self.cfg_fp = config_fingerprint(cfg)
        self.spec_fp = spec_fingerprint(spec)
        self.extract_cfg_fp = extract_config_fingerprint(cfg)
        self.align_cfg_fp = align_config_fingerprint(cfg)
        self.retrieve_cfg_fp = retrieve_config_fingerprint(cfg)

    def input_fingerprints(self, samples):
        return {sample: raw_input_fingerprints(
            self.input_dir, sample, self.image_size) for sample in samples}

    def extract_key(self, sample, input_fp):
        return cache_key("extract", [
            input_fp["png"], input_fp["text"], input_fp["regions"],
            self.extract_cfg_fp,
        ])

    def align_key(self, sample, extract_key_value):
        return cache_key("align", [
            extract_key_value, self.align_cfg_fp,
        ])

    def convert_key(self, sample, extract_key_value):
        return cache_key("convert", [
            extract_key_value, self.spec_fp, str(self.format_version),
        ])

    def validate_key(self, sample, input_fp):
        return cache_key("validate", [
            input_fp["png"], input_fp["text"], input_fp["regions"],
            input_fp["size"],
        ])

    def retrieve_key(self, align_keys):
        return cache_key("retrieve", [
            combine(sorted(align_keys.values()), prefix="align-set"),
            self.retrieve_cfg_fp,
        ])

    def plan(self, samples, previous):
        """返回每阶段 recompute/reuse 集合与当前键表。

        previous 为上次运行清单中的键表（可为空）；样本替换、配置调整、
        格式切换通过键比对自动收敛为精确失效。
        """
        previous = previous or {}
        input_fps = self.input_fingerprints(samples)
        current = {"validate": {}, "extract": {}, "align": {},
                   "convert": {}, "input": input_fps}
        plan = {stage: {"recompute": [], "reuse": []}
                for stage in ("validate", "extract", "align", "convert")}
        prev_keys = previous.get("keys", {})

        for sample in sorted(samples):
            input_fp = input_fps[sample]
            vkey = self.validate_key(sample, input_fp)
            ekey = self.extract_key(sample, input_fp)
            akey = self.align_key(sample, ekey)
            ckey = self.convert_key(sample, ekey)
            current["validate"][sample] = vkey
            current["extract"][sample] = ekey
            current["align"][sample] = akey
            current["convert"][sample] = ckey
            for stage, key in (("validate", vkey), ("extract", ekey),
                               ("align", akey), ("convert", ckey)):
                old = prev_keys.get(stage, {}).get(sample)
                if old == key and sample in (previous.get("valid_samples")
                                              or []):
                    plan[stage]["reuse"].append(sample)
                else:
                    plan[stage]["recompute"].append(sample)

        current["retrieve"] = self.retrieve_key(current["align"])
        old_retrieve = prev_keys.get("retrieve")
        plan["retrieve"] = {
            "recompute": [] if old_retrieve == current["retrieve"]
            else ["__index__"],
            "reuse": ["__index__"] if old_retrieve == current["retrieve"]
            else [],
        }
        return plan, current
