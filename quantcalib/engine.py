"""Deterministic end-to-end quant-calib pipeline engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quantcalib.calib.quantiles import calibrate_group
from quantcalib.data.config import QuantConfig
from quantcalib.data.weightpack import load_weight_pack
from quantcalib.data.calibration import load_calibration
from quantcalib.distill.correction import correct as distill_correct
from quantcalib.fingerprint import fingerprint, fingerprint_json
from quantcalib.ledger.ledger import adjudicate, ledger_fingerprint
from quantcalib.models import UnifiedReceipt
from quantcalib.parsers.receipts import parse_any, validate_references
from quantcalib.persistence.cache import (
    ReuseLedger,
    build_layer_cache_keys,
    invalidation_boundary,
    load_prior_group_keys,
)
from quantcalib.persistence.snapshot import (
    SnapshotStore,
    expected_manifest,
)
from quantcalib.quant.kernel import quantize_group
from quantcalib.schedule.budget import BudgetScheduler, WorkUnit


@dataclass
class EngineContext:
    workdir: Path
    config_path: str = "config/quant.yaml"
    weights_path: str = "weights/weights.jsonl"
    calibration_path: str = "calib/calibration.json"
    receipt_paths: tuple[str, ...] = (
        "receipts/history.q1",
        "receipts/history.q2.json",
        "receipts/history.q3.bin",
    )
    snapshot_dir: str = "out/snapshot"

    def resolve(self, rel: str) -> Path:
        return self.workdir / rel


class QuantCalibEngine:
    def __init__(self, context: EngineContext, cfg: QuantConfig) -> None:
        self.ctx = context
        self.cfg = cfg
        self.store = SnapshotStore(context.resolve(context.snapshot_dir))
        self.reuse = ReuseLedger()
        self.failures: list[dict[str, Any]] = []
        self.reuse_boundaries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    def _load_inputs(self) -> tuple:
        layers, layer_issues = load_weight_pack(
            self.ctx.resolve(self.ctx.weights_path))
        calib, calib_issues = load_calibration(
            self.ctx.resolve(self.ctx.calibration_path))
        all_receipts: list[UnifiedReceipt] = []
        for rel in self.ctx.receipt_paths:
            path = self.ctx.resolve(rel)
            if path.exists():
                all_receipts.extend(parse_any(path))
        rank = {"q1": 1, "q2": 2, "q3": 3}
        ceiling = rank[self.cfg.active_receipt_version]
        active_receipts = [r for r in all_receipts
                           if rank[r.source_version] <= ceiling]
        known = {layer.name: layer.content_version for layer in layers}
        receipt_issues = validate_references(active_receipts, known)
        issues = layer_issues + calib_issues + receipt_issues
        return layers, calib, active_receipts, issues

    # ------------------------------------------------------------------ #
    def _group_key(self, layer_name: str, group_index: int, stage_key: str) -> str:
        # The stage fingerprint itself is the unique name; keep it readable by
        # prefixing the layer/group coordinates.
        return f"{layer_name}__g{group_index:04d}__{stage_key}"

    def run(self, global_version: int) -> dict[str, Any]:
        layers, calib, receipts, issues = self._load_inputs()
        layers = sorted(layers, key=lambda layer: layer.name)

        weight_fp = fingerprint([layer.to_record() for layer in layers])
        calib_fp = fingerprint({
            "inputs": calib.inputs,
            "reference_outputs": calib.reference_outputs,
            "seed": calib.seed,
        })
        receipt_fp = fingerprint_json(
            [r.to_dict() for r in sorted(receipts, key=lambda r: (r.layer, r.source_version))])
        active_receipt_version = self._active_receipt_version(receipts)

        manifest = expected_manifest(
            global_version, self.cfg.to_dict(), weight_fp, calib_fp,
            receipt_fp, active_receipt_version,
        )
        resume, reason = self.store.resume_decision(manifest)
        if not resume:
            self.store.invalidate(reason)
            prior_manifest = None
        else:
            prior_manifest = self.store.read_manifest()
        # Reclaim any interrupted transaction before opening the new one.
        self.store.rollback()
        self.store.begin_commit()

        scheduler = BudgetScheduler(
            max_workers=self.cfg.max_workers,
            global_budget=self.cfg.global_point_budget,
            per_layer_quota=self.cfg.per_layer_quota,
            batch_budget=self.cfg.batch_point_budget,
        )

        layer_results: dict[str, dict[str, Any]] = {}
        units: list[WorkUnit] = []
        for layer in layers:
            granularity = self.cfg.layer_granularity(layer.name)
            for group_index in range(layer.size() // granularity):
                units.append(WorkUnit(
                    layer=layer.name,
                    group_index=group_index,
                    point_cost=granularity,
                    stage="quant",
                ))

        consumed = prior_manifest.consumed_budget if prior_manifest else 0
        per_layer_consumed = (
            dict(prior_manifest.per_layer_consumed) if prior_manifest else {})
        outcome, _ = scheduler.run(
            units, lambda unit: None, consumed, per_layer_consumed)
        self.failures.extend(outcome.failures)
        allowed = {(u.layer, u.group_index) for u in outcome.planned}

        quant_weights: list[np.ndarray] = []
        full_weights: list[np.ndarray] = []
        saturation_counts: list[int] = []

        for layer in layers:
            granularity = self.cfg.layer_granularity(layer.name)
            bits = self.cfg.layer_bits(layer.name)
            flat = layer.weights.reshape(-1)
            codes_flat = np.zeros_like(flat, dtype=np.int64)
            dequant_flat = np.zeros_like(flat)
            group_payloads: list[dict[str, Any]] = []
            layer_saturation = 0

            for group_index in range(layer.size() // granularity):
                if (layer.name, group_index) not in allowed:
                    continue
                start = group_index * granularity
                values = flat[start:start + granularity]
                keys = build_layer_cache_keys(layer, group_index, self.cfg, calib_fp)
                calib_key = keys.stage_key("calibration", granularity)
                prior_matches = self._prior_keys(
                    layer.name, group_index, keys.group_content)
                representative = self._select_representative(
                    prior_matches, keys)
                boundary = invalidation_boundary(
                    layer, group_index, bool(prior_matches), keys,
                    representative)
                self.reuse_boundaries.append({
                    "layer": layer.name,
                    "group_index": group_index,
                    "invalidated_stages": boundary,
                    "old_keys": representative.to_dict() if representative else None,
                    "new_keys": keys.to_dict(),
                })
                cached_calib = self.store.read_artifact(
                    "calibration", self._group_key(layer.name, group_index, calib_key))
                calib_hit = cached_calib is not None and "calibration" not in boundary
                self.reuse.record(
                    layer.name, group_index, "calibration", calib_key,
                    calib_hit, upstream=[], calibration_refs=[])
                if calib_hit:
                    from quantcalib.calib.quantiles import GroupCalibration
                    calibration = GroupCalibration(**cached_calib["calibration"])
                else:
                    calibration = calibrate_group(
                        layer.name, group_index, values, bits,
                        self.cfg.fractional_bits, self.cfg.q_lo_permille,
                        self.cfg.q_hi_permille, self.cfg.scheme)
                    self.store.write_artifact("calibration", self._group_key(
                        layer.name, group_index, calib_key), {
                            "calibration": calibration.to_dict(),
                            "cache_keys": keys.to_dict(),
                        })

                quant_key = keys.stage_key("quantization", granularity)
                cached_quant = self.store.read_artifact(
                    "quantization",
                    self._group_key(layer.name, group_index, quant_key))
                quant_hit = cached_quant is not None and not (
                    set(boundary) & {"quantization", "quantization-level",
                                     "params", "group-content"})
                self.reuse.record(
                    layer.name, group_index, "quantization", quant_key,
                    quant_hit, upstream=[calib_key], calibration_refs=[])
                if (quant_hit and cached_quant is not None
                        and len(cached_quant["quantization"]["integer_codes"]) == granularity):
                    quantization = cached_quant["quantization"]
                else:
                    quant_hit = False
                    quantization = quantize_group(
                        layer.name, group_index, values, calibration, bits,
                        self.cfg.error_tolerance, self.cfg.rounding_mode,
                        self.cfg.saturation_policy).to_dict()
                    self.store.write_artifact("quantization", self._group_key(
                        layer.name, group_index, quant_key), {
                            "quantization": quantization,
                            "cache_keys": keys.to_dict(),
                        })

                error_key = keys.stage_key("error", granularity)
                error_payload = {
                    "abs_errors": quantization["abs_errors"],
                    "rel_errors": quantization["rel_errors"],
                    "over_tolerance": quantization["over_tolerance"],
                    "error_sources": quantization["error_sources"],
                    "source_evidence": quantization["source_evidence"],
                    "tolerance": self.cfg.error_tolerance,
                }
                cached_error = self.store.read_artifact(
                    "error",
                    self._group_key(layer.name, group_index, error_key))
                error_hit = cached_error is not None and "error" not in boundary
                self.reuse.record(
                    layer.name, group_index, "error", error_key,
                    error_hit, upstream=[quant_key],
                    calibration_refs=list(range(calib.sample_count)))
                if not error_hit:
                    self.store.write_artifact(
                        "error",
                        self._group_key(layer.name, group_index, error_key),
                        {"error": error_payload, "cache_keys": keys.to_dict()})

                codes_flat[start:start + granularity] = np.asarray(
                    quantization["integer_codes"], dtype=np.int64)
                dequant_flat[start:start + granularity] = np.asarray(
                    quantization["dequantized"], dtype=np.float64)
                layer_saturation += int(quantization["saturation_count"])
                group_payloads.append({
                    "group_index": group_index,
                    "calibration": calibration.to_dict(),
                    "quantization": quantization,
                    "error": error_payload,
                    "cache_keys": keys.to_dict(),
                    "calibration_refs": list(range(calib.sample_count)),
                })

            quant_weights.append(dequant_flat.reshape(layer.weights.shape))
            full_weights.append(layer.weights.copy())
            saturation_counts.append(layer_saturation)
            layer_results[layer.name] = {
                "layer": layer.name,
                "shape": list(layer.shape),
                "content_version": layer.content_version,
                "bits": bits,
                "group_granularity": granularity,
                "groups": group_payloads,
                "integer_codes": codes_flat.tolist(),
                "saturation_count": layer_saturation,
            }

        distill_report = self._distillation(
            layers, full_weights, quant_weights, calib, saturation_counts, calib_fp)
        ledger_entries = adjudicate(receipts)
        ledger_by_layer = {entry.layer: entry for entry in ledger_entries}
        for layer in layers:
            entry = ledger_by_layer.get(layer.name)
            receipt_key = fingerprint_json({
                "stage": "receipt",
                "layer": layer.name,
                "receipt_version": self.cfg.active_receipt_version,
                "entry": entry.to_dict() if entry else None,
                "layer_content": fingerprint(layer.weights),
            })
            cached_receipt = self.store.read_artifact("receipt", receipt_key)
            self.reuse.record(
                layer.name, 0, "receipt", receipt_key,
                cached_receipt is not None, upstream=[],
                calibration_refs=list(range(calib.sample_count)))
            self.store.write_artifact("receipt", receipt_key, {
                "entry": entry.to_dict() if entry else None,
            })

        completed = sorted({
            unit.layer for unit in outcome.planned
        })
        manifest.completed_layers = completed
        manifest.consumed_budget = outcome.consumed_budget
        manifest.per_layer_consumed = outcome.per_layer_consumed
        self.store.commit(manifest)

        return {
            "global_version": global_version,
            "resume_reason": reason,
            "resumed": resume,
            "issues": [issue.to_dict() for issue in issues],
            "layers": [layer_results[name] for name in sorted(layer_results)],
            "distillation": distill_report.to_dict(),
            "ledger": [entry.to_dict() for entry in ledger_entries],
            "ledger_fingerprint": ledger_fingerprint(ledger_entries),
            "reuse": self.reuse.to_dict(),
            "invalidation_boundaries": self.reuse_boundaries,
            "schedule": outcome.to_dict(),
            "failures": self.failures,
            "manifest": manifest.to_dict(),
            "fingerprints": {
                "weight_pack": weight_fp,
                "calibration": calib_fp,
                "receipts": receipt_fp,
                "config": fingerprint_json(self.cfg.to_dict()),
            },
        }

    # ------------------------------------------------------------------ #
    def _prior_keys(self, layer_name: str, group_index: int,
                    group_fingerprint: str):
        return load_prior_group_keys(
            self.store.stage_path("calibration"),
            layer_name, group_index, group_fingerprint,
        )

    @staticmethod
    def _select_representative(prior_matches, current_keys):
        """Pick the closest prior keys for boundary comparison.

        Prefer an exact parameter+level match, then parameter-only match
        (level change only), then any same-content artifact.
        """
        if not prior_matches:
            return None
        for item in prior_matches:
            if (item.params == current_keys.params
                    and item.quantization_level == current_keys.quantization_level
                    and item.calibration_set == current_keys.calibration_set):
                return item
        for item in prior_matches:
            if item.params == current_keys.params:
                return item
        return prior_matches[0]

    def _active_receipt_version(self, receipts: list[UnifiedReceipt]) -> str:
        return self.cfg.active_receipt_version

    def _distillation(self, layers, full_weights, quant_weights, calib,
                      saturation_counts, calib_fp) -> Any:
        names = [layer.name for layer in layers]
        distill_key = fingerprint_json({
            "layers": names,
            "calibration": calib_fp,
            "temperature": self.cfg.distill_temperature,
            "steps": self.cfg.distill_steps,
            "soft_label_weight": self.cfg.soft_label_weight,
            "learning_rate": self.cfg.distill_learning_rate,
            "quant_weights": [fingerprint(w) for w in quant_weights],
        })
        cached = self.store.read_artifact("distillation", distill_key)
        self.reuse.record(
            "__all__", 0, "distillation", distill_key, cached is not None,
            upstream=[], calibration_refs=list(range(calib.sample_count)))
        if cached is not None:
            from quantcalib.distill.correction import DistillReport
            return DistillReport(**cached["report"])

        report = distill_correct(
            full_weights, quant_weights, calib.inputs,
            calib.reference_outputs, names,
            temperature=self.cfg.distill_temperature,
            steps=self.cfg.distill_steps,
            soft_label_weight=self.cfg.soft_label_weight,
            learning_rate=self.cfg.distill_learning_rate,
            saturation_counts=saturation_counts,
        )
        self.store.write_artifact("distillation", distill_key,
                                  {"report": report.to_dict()})
        return report
