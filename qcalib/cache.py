"""Content-addressed cache keys and precise invalidation/reuse bookkeeping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hashing import array_fingerprint, composite_key, fingerprint
from .models import CalibrationSet, LayerQuantConfig, WeightLayer

STAGE_CAL = "quantile_calibration"
STAGE_QNT = "fixed_point_requantization"
STAGE_ERR = "dequantization_error"
STAGE_DST = "distillation_correction"
STAGE_RCP = "receipt_ledger"
CACHE_STAGES = (STAGE_CAL, STAGE_QNT, STAGE_ERR, STAGE_DST, STAGE_RCP)


def layer_content_fingerprint(layer: WeightLayer) -> str:
    return array_fingerprint(layer.weights, prefix="lc-")


def group_content_fingerprint(values) -> str:
    import numpy as np
    return array_fingerprint(np.asarray(values, dtype=np.float64), prefix="gc-")


def parameter_fingerprint(cfg: LayerQuantConfig) -> str:
    return fingerprint(cfg.to_material(), prefix="pm-")


def calibration_parameter_fingerprint(cfg: LayerQuantConfig) -> str:
    # Calibration depends only on quantiles, grouping and scheme, not on the
    # integer level/frac_bits, which belong to the requantization stage.
    material = {
        "name": cfg.name,
        "group_size": cfg.group_size,
        "q_lo": cfg.q_lo,
        "q_hi": cfg.q_hi,
        "scheme": cfg.scheme,
    }
    return fingerprint(material, prefix="cpm-")


def calibration_fingerprint(cal: CalibrationSet, sample_refs: tuple[str, ...]) -> str:
    indices = [cal.sample_ids.index(sid) for sid in sample_refs]
    material = {
        "inputs": cal.inputs[indices].tolist(),
        "references": cal.references[indices].tolist(),
        "sample_ids": list(sample_refs),
    }
    return fingerprint(material, prefix="calib-")


def quant_level_fingerprint(cfg: LayerQuantConfig) -> str:
    return fingerprint({"level": cfg.level, "bits": cfg.bits,
                        "frac_bits": cfg.frac_bits}, prefix="lvl-")


def group_cache_keys(layer: WeightLayer, cfg: LayerQuantConfig,
                     group_values: list, group_index: int) -> dict[str, str]:
    lcf = layer_content_fingerprint(layer)
    gcf = group_content_fingerprint(group_values)
    pmf = parameter_fingerprint(cfg)
    cpmf = calibration_parameter_fingerprint(cfg)
    lvl = quant_level_fingerprint(cfg)
    return {
        STAGE_CAL: composite_key({"lcf": lcf, "gcf": gcf, "cpmf": cpmf}),
        STAGE_QNT: composite_key({"lcf": lcf, "gcf": gcf, "pmf": pmf, "lvl": lvl}),
        STAGE_ERR: composite_key({"lcf": lcf, "gcf": gcf, "pmf": pmf, "lvl": lvl}),
    }


def distill_cache_key(layer: WeightLayer, cfg: LayerQuantConfig,
                      cal: CalibrationSet, sample_refs: tuple[str, ...],
                      distill_material: dict[str, Any]) -> str:
    return composite_key({
        "lcf": layer_content_fingerprint(layer),
        "pmf": parameter_fingerprint(cfg),
        "lvl": quant_level_fingerprint(cfg),
        "calib": calibration_fingerprint(cal, sample_refs),
        "distill": fingerprint(distill_material, prefix="dpm-"),
    })


def receipt_cache_key(receipt_ledger_material: dict[str, Any]) -> str:
    return composite_key({"receipts": fingerprint(receipt_ledger_material,
                                                  prefix="rl-")})


@dataclass
class InvalidationRecord:
    layer: str
    group_index: int | None
    stage: str
    old_key: str | None
    new_key: str | None
    reason: str
    hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "group_index": self.group_index,
            "stage": self.stage,
            "old_key": self.old_key,
            "new_key": self.new_key,
            "reason": self.reason,
            "hit": self.hit,
        }


class ReuseLedger:
    """Records every cache hit/miss with an explicit reason and evidence."""

    def __init__(self) -> None:
        self.records: list[InvalidationRecord] = []

    def hit(self, layer: str, stage: str, key: str,
            group_index: int | None = None) -> None:
        self.records.append(InvalidationRecord(
            layer=layer, group_index=group_index, stage=stage,
            old_key=key, new_key=key, reason="identical_cache_key", hit=True))

    def miss(self, layer: str, stage: str, new_key: str, reason: str,
             old_key: str | None = None,
             group_index: int | None = None) -> None:
        self.records.append(InvalidationRecord(
            layer=layer, group_index=group_index, stage=stage,
            old_key=old_key, new_key=new_key, reason=reason, hit=False))

    def to_list(self) -> list[dict[str, Any]]:
        ordered = sorted(self.records,
                         key=lambda r: (r.layer,
                                        r.group_index if r.group_index is not None else -1,
                                        r.stage))
        return [record.to_dict() for record in ordered]

    def canonical_invalidation(self) -> list[dict[str, Any]]:
        """Run-history-independent view used for full/incremental parity.

        It keeps every stage key and its invalidation reason but drops the
        transient hit flag, so a cold rebuild and an incremental rebuild emit
        identical canonical ledgers while the diagnostic view still reports
        hits for the incremental run.
        """
        ordered = sorted(self.records,
                         key=lambda r: (r.layer,
                                        r.group_index if r.group_index is not None else -1,
                                        r.stage,
                                        r.new_key or ""))
        normalized = []
        for record in ordered:
            reason = ("cache_key_unchanged" if record.hit
                      or record.reason == "first_build"
                      or record.old_key is None
                      else record.reason)
            normalized.append({
                "layer": record.layer,
                "group_index": record.group_index,
                "stage": record.stage,
                "old_key": record.old_key if not record.hit else None,
                "new_key": record.new_key,
                "reason": reason,
            })
        return normalized
