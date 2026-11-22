"""Stage-wise fixed-point error propagation diagnostics.

For each task the adapter is fully re-computed at the configured tier and at
the next higher tier, plus a float64 reference path.  Three stages are
compared element-wise against the reference: adapter weights, merged weights
and target-layer outputs.  When a sample's final verdict differs between the
two tiers the flip is attributed to quantization truncation, merge order or
base version change.
"""

from __future__ import annotations

import numpy as np

from .fixedpoint import dequant, higher_frac_bits, quantize
from .merge import merge_layer
from .train import train_adapter_fixed, train_adapter_float64

STAGES = ("adapter", "merged", "output")

CAUSE_QUANT_TRUNCATION = "quant_truncation"
CAUSE_MERGE_ORDER = "merge_order"
CAUSE_BASE_VERSION = "base_version_change"


def _within_tol(diff, ref, atol, rtol):
    return bool(np.all(np.abs(diff) <= atol + rtol * np.abs(ref)))


def _diff_report(diff, ref):
    diff = np.asarray(diff, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return {
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "ref_max_abs": float(np.max(np.abs(ref))) if ref.size else 0.0,
    }


def _train_and_merge(task, layer, xs, ys, lr, epochs, frac_bits, reverse_merge=False):
    W_q = quantize(layer.weight, frac_bits)
    b_q = quantize(layer.bias, frac_bits)
    xs_q = [quantize(x, frac_bits) for x in xs]
    ys_q = [quantize(y, frac_bits) for y in ys]
    adapter = train_adapter_fixed(task, W_q, b_q, xs_q, ys_q, lr, epochs, frac_bits)
    merged, _steps, _sat = merge_layer(W_q, [adapter], frac_bits, reverse=reverse_merge)
    outputs = [
        dequant(merged, frac_bits) @ dequant(x, frac_bits) + dequant(b_q, frac_bits)
        for x in xs_q
    ]
    return adapter, merged, outputs


def diagnose_task(task, layer, teacher_delta, samples, sample_inputs, lr, epochs,
                  frac_bits, atol, rtol,
                  base_fingerprint_before=None, base_fingerprint_after=None,
                  layer_adapters=None):
    """Run the three-stage diagnosis for one task. Returns a report dict."""
    xs = sample_inputs
    w_ref = layer.weight if teacher_delta is None else layer.weight + teacher_delta
    ys = [w_ref @ x + layer.bias for x in xs]

    # float64 reference path
    A_ref, B_ref, _ = train_adapter_float64(task, layer.weight, layer.bias, xs, ys, lr, epochs)
    delta_ref = task.scaling * (B_ref @ A_ref)
    merged_ref = layer.weight + delta_ref
    outputs_ref = [merged_ref @ x + layer.bias for x in xs]

    hi_bits = higher_frac_bits(frac_bits)
    tier_runs = {}
    for label, bits in (("config", frac_bits), ("higher", hi_bits)):
        adapter, merged, outputs = _train_and_merge(task, layer, xs, ys, lr, epochs, bits)
        tier_runs[label] = {"bits": bits, "adapter": adapter, "merged": merged, "outputs": outputs}

    stage_reports = {}
    first_exceeding = None
    for label, run in tier_runs.items():
        bits = run["bits"]
        adapter_diff = np.concatenate(
            [
                (dequant(run["adapter"].A, bits) - A_ref).ravel(),
                (dequant(run["adapter"].B, bits) - B_ref).ravel(),
            ]
        )
        adapter_ref = np.concatenate([A_ref.ravel(), B_ref.ravel()])
        merged_diff = dequant(run["merged"], bits) - merged_ref
        out_diffs = [o - r for o, r in zip(run["outputs"], outputs_ref)]
        out_diff = np.concatenate([d.ravel() for d in out_diffs]) if out_diffs else np.zeros(0)
        out_ref = np.concatenate([r.ravel() for r in outputs_ref]) if outputs_ref else np.zeros(0)

        stages = {
            "adapter": (adapter_diff, adapter_ref),
            "merged": (merged_diff, merged_ref),
            "output": (out_diff, out_ref),
        }
        stage_reports[label] = {}
        for stage in STAGES:
            diff, ref = stages[stage]
            exceeded = not _within_tol(diff, ref, atol, rtol)
            stage_reports[label][stage] = {
                **_diff_report(diff, ref),
                "exceeds_tolerance": exceeded,
            }
            if exceeded and first_exceeding is None and label == "config":
                first_exceeding = stage

    def verdicts(run):
        return [_within_tol(o - r, r, atol, rtol) for o, r in zip(run["outputs"], outputs_ref)]

    verdict_config = verdicts(tier_runs["config"])
    verdict_higher = verdicts(tier_runs["higher"])

    flips = []
    for idx, sample in enumerate(samples):
        if verdict_config[idx] == verdict_higher[idx]:
            continue
        cause = _attribute_flip(
            task, layer, frac_bits,
            base_fingerprint_before, base_fingerprint_after,
            layer_adapters=layer_adapters,
        )
        flips.append(
            {
                "sample_id": sample.sample_id,
                "verdict_config_tier": bool(verdict_config[idx]),
                "verdict_higher_tier": bool(verdict_higher[idx]),
                "cause": cause,
            }
        )

    return {
        "task": task.name,
        "layer": task.target_layer,
        "tiers": {"config": frac_bits, "higher": hi_bits},
        "stages": stage_reports,
        "first_exceeding_stage": first_exceeding,
        "flips": flips,
    }


def _attribute_flip(task, layer, frac_bits,
                    base_fingerprint_before, base_fingerprint_after,
                    layer_adapters=None):
    # Base version change dominates: the two computations used different bases.
    if (
        base_fingerprint_before is not None
        and base_fingerprint_after is not None
        and base_fingerprint_before != base_fingerprint_after
    ):
        return CAUSE_BASE_VERSION
    # Merge-order probe: merge all adapters of this layer at the config tier
    # in fixed vs reversed order; a difference means merge order is observable.
    W_q = quantize(layer.weight, frac_bits)
    adapters = list(layer_adapters) if layer_adapters else []
    if len(adapters) < 2:
        return CAUSE_QUANT_TRUNCATION
    merged_fwd, _, _ = merge_layer(W_q, adapters, frac_bits, reverse=False)
    merged_rev, _, _ = merge_layer(W_q, adapters, frac_bits, reverse=True)
    if not np.array_equal(merged_fwd, merged_rev):
        return CAUSE_MERGE_ORDER
    return CAUSE_QUANT_TRUNCATION


def cross_version_attribution(flips, fingerprint_before, fingerprint_after):
    """Re-label flips when a base upgrade happened between two runs."""
    if fingerprint_before == fingerprint_after:
        return flips
    return [{**flip, "cause": CAUSE_BASE_VERSION} for flip in flips]
