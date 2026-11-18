"""JSON configuration loading and validation."""

import json
import os

from .errors import BusinessError, E_EMBED_INVALID
from .fixedpoint import is_power_of_two

AGG_MODES = ("sum", "mean")

STRUCTURAL_FIELDS = (
    "block_size",
    "rows_per_block",
    "dim",
    "frac_bits",
    "table_path",
)


class Config:
    def __init__(self, data, base_dir="."):
        self.raw = dict(data)
        self.base_dir = base_dir
        self.block_size = int(data["block_size"])
        self.rows_per_block = int(data["rows_per_block"])
        self.dim = int(data["dim"])
        self.quant_level = int(data["quant_level"])
        self.frac_bits = int(data["frac_bits"])
        self.agg_mode = str(data["agg_mode"])
        self.max_concurrency = int(data["max_concurrency"])
        self.global_point_budget = int(data["global_point_budget"])
        self.per_request_quota = int(data["per_request_quota"])
        self.table_path = self._resolve(data["table_path"])
        self.requests_path = self._resolve(data["requests_path"])
        self.snapshot_dir = self._resolve(data["snapshot_dir"])
        self.report_path = self._resolve(data["report_path"])
        self.validate()

    def _resolve(self, path):
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.base_dir, path))

    def validate(self):
        problems = []
        if self.block_size <= 0:
            problems.append("block_size must be > 0")
        if self.rows_per_block <= 0:
            problems.append("rows_per_block must be > 0")
        if self.dim <= 0:
            problems.append("dim must be > 0")
        if not is_power_of_two(self.quant_level):
            problems.append("quant_level must be a power of two >= 2")
        if self.frac_bits < 0:
            problems.append("frac_bits must be >= 0")
        if self.agg_mode not in AGG_MODES:
            problems.append(f"agg_mode must be one of {AGG_MODES}")
        if self.max_concurrency <= 0:
            problems.append("max_concurrency must be > 0")
        if self.global_point_budget < 0:
            problems.append("global_point_budget must be >= 0")
        if self.per_request_quota < 0:
            problems.append("per_request_quota must be >= 0")
        if problems:
            raise BusinessError(E_EMBED_INVALID,
                                {"config_problems": problems})

    def structural_dict(self):
        return {k: self.raw[k] for k in STRUCTURAL_FIELDS}

    def to_runtime_dict(self):
        data = dict(self.raw)
        data["quant_level"] = self.quant_level
        data["agg_mode"] = self.agg_mode
        return data


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    base_dir = os.path.dirname(os.path.abspath(path))
    return Config(data, base_dir=base_dir)
