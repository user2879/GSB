"""PEFT YAML config loading and validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import yaml

from .errors import E_QUANT_INVALID, failure
from .fixedpoint import TIERS


@dataclass
class PeftConfig:
    target_layers: list
    learning_rate: float
    epochs: int
    frac_bits: int
    quant_tier: int
    atol: float
    rtol: float
    max_workers: int
    merge_budget: int
    snapshot_interval: int
    raw: dict = field(default_factory=dict)

    def config_hash(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def train_hash(self) -> str:
        """Hash of the fields that influence training results."""
        relevant = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "frac_bits": self.frac_bits,
            "quant_tier": self.quant_tier,
        }
        return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def _pos_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_config(path: str):
    """Returns (PeftConfig|None, failures)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        return None, [failure(E_QUANT_INVALID, f"config unreadable: {exc}")]
    if not isinstance(raw, dict):
        return None, [failure(E_QUANT_INVALID, "config is not a mapping")]

    failures = []

    def err(msg):
        failures.append(failure(E_QUANT_INVALID, msg))

    target_layers = raw.get("target_layers")
    if not isinstance(target_layers, list) or not target_layers or not all(
        isinstance(x, str) for x in target_layers
    ):
        err("target_layers must be a non-empty list of layer names")
        target_layers = []
    if len(set(target_layers)) != len(target_layers):
        err("target_layers contains duplicates")

    lr = raw.get("learning_rate")
    if not _pos_finite(lr):
        err("learning_rate must be a positive finite number")
        lr = 0.0

    epochs = raw.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        err("epochs must be a positive integer")
        epochs = 0

    frac_bits = raw.get("frac_bits")
    quant_tier = raw.get("quant_tier")
    if frac_bits not in TIERS.values():
        err(f"frac_bits must be one of {sorted(TIERS.values())}")
    if quant_tier not in TIERS:
        err(f"quant_tier must be one of {sorted(TIERS)}")
    elif frac_bits in TIERS.values() and TIERS[quant_tier] != frac_bits:
        err(f"quant_tier {quant_tier} inconsistent with frac_bits {frac_bits}")

    atol = raw.get("atol")
    rtol = raw.get("rtol")
    if not _pos_finite(atol):
        err("atol must be a positive finite number")
        atol = 0.0
    if not _pos_finite(rtol):
        err("rtol must be a positive finite number")
        rtol = 0.0

    max_workers = raw.get("max_workers")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers < 1:
        err("max_workers must be a positive integer")
        max_workers = 1

    merge_budget = raw.get("merge_budget")
    if not isinstance(merge_budget, int) or isinstance(merge_budget, bool) or merge_budget < 0:
        err("merge_budget must be a non-negative integer")
        merge_budget = 0

    snapshot_interval = raw.get("snapshot_interval")
    if not isinstance(snapshot_interval, int) or isinstance(snapshot_interval, bool) or snapshot_interval < 1:
        err("snapshot_interval must be a positive integer")
        snapshot_interval = 1

    if failures:
        return None, failures
    return (
        PeftConfig(
            target_layers=list(target_layers),
            learning_rate=float(lr),
            epochs=epochs,
            frac_bits=frac_bits,
            quant_tier=quant_tier,
            atol=float(atol),
            rtol=float(rtol),
            max_workers=max_workers,
            merge_budget=merge_budget,
            snapshot_interval=snapshot_interval,
            raw=raw,
        ),
        [],
    )
