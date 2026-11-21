"""Loading and validation of the four deterministic inputs.

Objects exposed here are plain, JSON-derived dataclasses so they can themselves
be fingerprinted.  :class:`EffectiveConfig` resolves the global defaults plus a
per-scale-tier override profile into one immutable view used by the compute
components.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .common import content_hash
from .errors import E_CONFIG, E_DIM_MISMATCH, DiagError

SCHEMES = ("rope", "linear", "ntk")
SCHEME_LABELS = {
    "rope": "rotary_position_encoding",
    "linear": "linear_interpolation",
    "ntk": "ntk_aware_scaling",
}
DEFAULT_TOLERANCES = {"position": 1e-6, "score": 1e-6, "weight": 1e-8}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


@dataclass(frozen=True)
class EffectiveConfig:
    tier_id: str
    base: float
    train_len: int
    target_len: int
    dim: int
    scale: float
    chunk_size: int
    precision_tiers: tuple[str, ...]
    workers: int
    snapshot_interval: int
    batch_point_budget: int
    memory_limit_mb: int
    tolerances: dict
    sensitivity_base_grid: tuple[float, ...]
    sensitivity_scale_grid: tuple[float, ...]

    def param_subset(self, stage: str) -> dict:
        """Stage-relevant parameter subset (drives precise invalidation)."""
        common = {
            "tier_id": self.tier_id,
            "train_len": self.train_len,
            "target_len": self.target_len,
            "dim": self.dim,
            "chunk_size": self.chunk_size,
        }
        if stage == "encode":
            common.update(
                base=self.base,
                scale=self.scale,
            )
        elif stage == "score":
            common.update(
                base=self.base,
                scale=self.scale,
            )
        elif stage == "error":
            common.update(
                base=self.base,
                scale=self.scale,
                tolerances=self.tolerances,
                precision_tiers=list(self.precision_tiers),
            )
        elif stage == "sensitivity":
            common.update(
                tolerances=self.tolerances,
                sensitivity_base_grid=list(self.sensitivity_base_grid),
                sensitivity_scale_grid=list(self.sensitivity_scale_grid),
            )
        return common

    def with_overrides(self, **changes) -> "EffectiveConfig":
        return replace(self, **changes)


@dataclass
class Config:
    raw: dict
    schemes: tuple[str, ...]
    global_cfg: dict
    scale_profiles: dict
    paths: dict

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = load_json(path)
        try:
            schemes = tuple(raw["schemes"])
            global_cfg = raw["global"]
            scale_profiles = raw.get("scale_profiles", {})
            paths = raw.get("paths", {})
        except KeyError as exc:  # pragma: no cover - guarded by generator
            raise DiagError(E_CONFIG, "missing config section: {}".format(exc))
        for scheme in schemes:
            if scheme not in SCHEMES:
                raise DiagError(E_CONFIG, "unknown scheme: {}".format(scheme))
        cfg = cls(raw=raw, schemes=schemes, global_cfg=global_cfg,
                  scale_profiles=scale_profiles, paths=paths)
        for tier_id in scale_profiles:
            cfg.effective(tier_id)  # validate eagerly
        return cfg

    def effective(self, tier_id: str) -> EffectiveConfig:
        g = self.global_cfg
        prof = self.scale_profiles.get(tier_id, {})

        def pick(key: str, default: Any = None) -> Any:
            if key in prof:
                return prof[key]
            if key in g:
                return g[key]
            return default

        dim = int(pick("dim"))
        chunk_size = int(pick("chunk_size"))
        if dim <= 0 or chunk_size <= 0:
            raise DiagError(E_CONFIG, "dim and chunk_size must be positive")
        if dim % 2 != 0:
            raise DiagError(E_CONFIG, "dim must be even for rotary pairs")
        workers = int(pick("workers"))
        snapshot_interval = int(pick("snapshot_interval"))
        if workers < 1 or snapshot_interval < 1:
            raise DiagError(E_CONFIG, "workers/snapshot_interval must be >= 1")
        tolerances = dict(DEFAULT_TOLERANCES)
        tolerances.update(g.get("tolerances", {}))
        precision_tiers = tuple(pick("precision_tiers", ["float64", "float32"]))
        return EffectiveConfig(
            tier_id=tier_id,
            base=float(pick("base")),
            train_len=int(pick("train_len")),
            target_len=int(pick("target_len")),
            dim=dim,
            scale=float(pick("scale")),
            chunk_size=chunk_size,
            precision_tiers=precision_tiers,
            workers=workers,
            snapshot_interval=snapshot_interval,
            batch_point_budget=int(pick("batch_point_budget")),
            memory_limit_mb=float(pick("memory_limit_mb")),
            tolerances=tolerances,
            sensitivity_base_grid=tuple(
                float(x) for x in g["sensitivity"]["base_grid"]),
            sensitivity_scale_grid=tuple(
                float(x) for x in g["sensitivity"]["scale_grid"]),
        )

    def fingerprint_inputs(self, samples_doc: Any, scale_doc: Any,
                           injection_doc: Any) -> dict:
        return {
            "config_hash": content_hash(self.raw),
            "injection_hash": content_hash(injection_doc),
            "scale_hash": content_hash(scale_doc),
            "samples_hash": content_hash(samples_doc),
        }


@dataclass(frozen=True)
class Sample:
    sample_id: str
    tier_id: str
    query_pos: int
    key_positions: tuple[int, ...]
    query_features: tuple[float, ...]
    key_features: tuple[tuple[float, ...], ...]
    index: int

    @property
    def dim(self) -> int:
        return len(self.query_features)

    def content_dict(self) -> dict:
        """Fingerprinted sample content (id + tier + positions + features)."""
        return {
            "sample_id": self.sample_id,
            "tier_id": self.tier_id,
            "query_pos": self.query_pos,
            "key_positions": list(self.key_positions),
            "query_features": [float(x) for x in self.query_features],
            "key_features": [[float(x) for x in row] for row in self.key_features],
        }


def load_samples(path: str | Path) -> list[Sample]:
    rows = load_jsonl(path)
    samples: list[Sample] = []
    for index, row in enumerate(rows):
        qf = row["query_features"]
        kfs = row["key_features"]
        dim = len(qf)
        kps = row["key_positions"]
        if any(len(kf) != dim for kf in kfs) or len(kfs) != len(kps):
            raise DiagError(
                E_DIM_MISMATCH,
                "sample {} feature/key dimension mismatch".format(row["sample_id"]),
            )
        samples.append(
            Sample(
                sample_id=str(row["sample_id"]),
                tier_id=str(row["tier_id"]),
                query_pos=int(row["query_pos"]),
                key_positions=tuple(int(x) for x in kps),
                query_features=tuple(float(x) for x in qf),
                key_features=tuple(tuple(float(x) for x in kf) for kf in kfs),
                index=index,
            )
        )
    return samples


@dataclass(frozen=True)
class Injection:
    injection_id: str
    sample_id: str
    stage: str
    kind: str
    params: dict = field(default_factory=dict)


def load_injections(path: str | Path) -> list[Injection]:
    doc = load_json(path)
    out: list[Injection] = []
    for item in doc["injections"]:
        out.append(
            Injection(
                injection_id=str(item["injection_id"]),
                sample_id=str(item["sample_id"]),
                stage=str(item["stage"]),
                kind=str(item["kind"]),
                params=dict(item.get("params", {})),
            )
        )
    return out
