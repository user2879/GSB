from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .canonical import fingerprint
from .errors import E_CONFIG_INVALID, DiagnosticError, Failure


QUANT_LEVEL_BITS = {"Q8": 8, "Q16": 16, "Q24": 24}
QUANT_LEVEL_FRACTION_SHIFT = {"Q8": -8, "Q16": 0, "Q24": 4}
ALLOWED_FRACTIONAL_BITS = (4, 8, 12)
ALLOWED_SIGN_TIE_RULES = ("positive", "negative")

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": "build-config/v1",
    "seed": 239,
    "chunk_window": 10,
    "chunk_overlap": 3,
    "pad_token": "<PAD>",
    "subword_min_n": 3,
    "subword_max_n": 4,
    "hash_buckets": 64,
    "projection_dim": 16,
    "projection_pool_dim": 32,
    "bucket_bits": 4,
    "quant_level": "Q16",
    "fractional_bits": 8,
    "sign_tie_rule": "positive",
    "similarity_threshold": 0.35,
    "max_concurrency": 4,
    "projection_point_budget": 100,
}


@dataclass
class ConfigBundle:
    data: Dict[str, Any]
    path: Optional[str] = None

    @property
    def config_hash(self) -> str:
        return fingerprint(self.data, 20)

    @property
    def chunk_fingerprint(self) -> str:
        return fingerprint(
            {
                "chunk_window": self.data["chunk_window"],
                "chunk_overlap": self.data["chunk_overlap"],
                "pad_token": self.data["pad_token"],
                "rule": "terms-sliding-window-v1",
            },
            20,
        )

    @property
    def feature_fingerprint(self) -> str:
        return fingerprint(
            {
                "hash_buckets": self.data["hash_buckets"],
                "subword_min_n": self.data["subword_min_n"],
                "subword_max_n": self.data["subword_max_n"],
                "pad_token": self.data["pad_token"],
                "rule": "boundary-marked-character-ngram-fnv1a-v1",
            },
            20,
        )

    @property
    def projection_fingerprint(self) -> str:
        pool_dim = int(self.data.get("projection_pool_dim") or self.data["projection_dim"])
        return fingerprint(
            {
                "seed": self.data["seed"],
                "hash_buckets": self.data["hash_buckets"],
                "projection_pool_dim": pool_dim,
                "rule": "seedsequence-normal-matrix-v1",
            },
            20,
        )

    @property
    def effective_fractional_bits(self) -> int:
        shift = QUANT_LEVEL_FRACTION_SHIFT[self.data["quant_level"]]
        return max(0, int(self.data["fractional_bits"]) + shift)

    @property
    def quant_fingerprint(self) -> str:
        return fingerprint(
            {
                "quant_level": self.data["quant_level"],
                "quant_bits": QUANT_LEVEL_BITS[self.data["quant_level"]],
                "fractional_bits": self.data["fractional_bits"],
                "effective_fractional_bits": self.effective_fractional_bits,
                "rule": "round-half-away-from-zero-clip-v1",
            },
            20,
        )

    @property
    def bucket_fingerprint(self) -> str:
        return fingerprint(
            {
                "bucket_bits": self.data["bucket_bits"],
                "sign_tie_rule": self.data["sign_tie_rule"],
                "rule": "msb-first-negative-sign-bit-v1",
            },
            20,
        )

    @property
    def param_fingerprint(self) -> str:
        return fingerprint(
            {
                "chunk": self.chunk_fingerprint,
                "feature": self.feature_fingerprint,
                "projection": self.projection_fingerprint,
                "quant": self.quant_fingerprint,
                "bucket": self.bucket_fingerprint,
                "projection_dim": self.data["projection_dim"],
            },
            24,
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_config(data: Any, path: Optional[str] = None) -> ConfigBundle:
    failures: List[Failure] = []
    if not isinstance(data, dict):
        raise DiagnosticError(
            E_CONFIG_INVALID,
            "configuration must be a YAML mapping",
            [Failure(E_CONFIG_INVALID, "top-level value is not a mapping", path=path)],
        )

    merged = dict(DEFAULT_CONFIG)
    merged.update(data)

    required_positive = (
        "seed",
        "chunk_window",
        "projection_dim",
        "hash_buckets",
        "bucket_bits",
        "max_concurrency",
        "projection_point_budget",
    )
    for key in required_positive:
        value = merged.get(key)
        if not _is_int(value) or value <= 0:
            failures.append(Failure(E_CONFIG_INVALID, f"{key} must be a positive integer", path=path, field=key))

    overlap = merged.get("chunk_overlap")
    if not _is_int(overlap) or overlap < 0:
        failures.append(Failure(E_CONFIG_INVALID, "chunk_overlap must be a non-negative integer", path=path, field="chunk_overlap"))

    window = merged.get("chunk_window")
    if _is_int(window) and _is_int(overlap) and overlap >= window:
        failures.append(
            Failure(
                E_CONFIG_INVALID,
                "chunk_overlap must be smaller than chunk_window",
                path=path,
                field="chunk_overlap",
                details={"chunk_window": window, "chunk_overlap": overlap},
            )
        )

    dim = merged.get("projection_dim")
    pool_dim = merged.get("projection_pool_dim")
    if pool_dim is not None:
        if not _is_int(pool_dim) or pool_dim <= 0:
            failures.append(Failure(E_CONFIG_INVALID, "projection_pool_dim must be a positive integer", path=path, field="projection_pool_dim"))
        elif _is_int(dim) and pool_dim < dim:
            failures.append(Failure(E_CONFIG_INVALID, "projection_pool_dim must be >= projection_dim", path=path, field="projection_pool_dim"))

    bucket_bits = merged.get("bucket_bits")
    if _is_int(bucket_bits):
        if bucket_bits < 1 or bucket_bits > 8:
            failures.append(Failure(E_CONFIG_INVALID, "bucket_bits must be in [1, 8]", path=path, field="bucket_bits", details={"value": bucket_bits}))
        elif _is_int(dim) and bucket_bits > dim:
            failures.append(Failure(E_CONFIG_INVALID, "bucket_bits must not exceed projection_dim", path=path, field="bucket_bits", details={"bucket_bits": bucket_bits, "projection_dim": dim}))

    quant_level = merged.get("quant_level")
    if quant_level not in QUANT_LEVEL_BITS:
        failures.append(Failure(E_CONFIG_INVALID, f"quant_level must be one of {sorted(QUANT_LEVEL_BITS)}", path=path, field="quant_level"))

    fraction = merged.get("fractional_bits")
    if fraction not in ALLOWED_FRACTIONAL_BITS or isinstance(fraction, bool):
        failures.append(Failure(E_CONFIG_INVALID, f"fractional_bits must be one of {ALLOWED_FRACTIONAL_BITS}", path=path, field="fractional_bits"))
    elif quant_level in QUANT_LEVEL_BITS:
        effective = max(0, int(fraction) + QUANT_LEVEL_FRACTION_SHIFT[quant_level])
        if effective >= QUANT_LEVEL_BITS[quant_level]:
            failures.append(Failure(E_CONFIG_INVALID, "effective fractional width does not fit quant_level", path=path, field="fractional_bits"))

    threshold = merged.get("similarity_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold) <= 1.0):
        failures.append(Failure(E_CONFIG_INVALID, "similarity_threshold must be in [0, 1]", path=path, field="similarity_threshold"))

    if merged.get("sign_tie_rule") not in ALLOWED_SIGN_TIE_RULES:
        failures.append(Failure(E_CONFIG_INVALID, f"sign_tie_rule must be one of {ALLOWED_SIGN_TIE_RULES}", path=path, field="sign_tie_rule"))

    for key in ("subword_min_n", "subword_max_n"):
        if not _is_int(merged.get(key)) or merged[key] < 1:
            failures.append(Failure(E_CONFIG_INVALID, f"{key} must be a positive integer", path=path, field=key))
    if _is_int(merged.get("subword_min_n")) and _is_int(merged.get("subword_max_n")) and merged["subword_min_n"] > merged["subword_max_n"]:
        failures.append(Failure(E_CONFIG_INVALID, "subword_min_n must be <= subword_max_n", path=path, field="subword_min_n"))

    max_concurrency = merged.get("max_concurrency")
    if _is_int(max_concurrency) and max_concurrency > 16:
        failures.append(Failure(E_CONFIG_INVALID, "max_concurrency must not exceed 16", path=path, field="max_concurrency"))

    if failures:
        raise DiagnosticError(E_CONFIG_INVALID, f"{len(failures)} configuration error(s)", failures)

    merged["similarity_threshold"] = float(merged["similarity_threshold"])
    merged["projection_pool_dim"] = int(merged.get("projection_pool_dim") or max(32, int(merged["projection_dim"])))
    return ConfigBundle(merged, path=path)


def load_config_file(path: str | Path) -> ConfigBundle:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # YAML and IO failures share the stable code.
        failure = Failure(E_CONFIG_INVALID, f"cannot parse YAML: {exc}", path=str(path))
        raise DiagnosticError(E_CONFIG_INVALID, str(exc), [failure]) from exc
    return validate_config(raw or {}, path=str(path))
