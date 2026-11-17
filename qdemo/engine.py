"""Incremental recompute engine.

Cache keys are derived from layer content + parameters. A config change
invalidates exactly the affected layers and their downstream eval / distill
stages; everything else is reused. All computation is deterministic: same
inputs always produce bit-identical outputs.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import evaluate, quantize
from .common import (
    E_DEPENDENCY_FAILED,
    E_LAYER_FAILED,
    fingerprint,
)
from .config import layer_params

STAGES = ("quant", "eval", "distill")


def _fail_hook(layer: str, stage: str) -> None:
    """Deterministic test hook: QDEMO_FAIL_LAYER / QDEMO_FAIL_STAGE inject a fault."""
    fl = os.environ.get("QDEMO_FAIL_LAYER")
    fs = os.environ.get("QDEMO_FAIL_STAGE")
    if fl and fl == layer and (not fs or fs == stage):
        raise RuntimeError(f"injected failure at {layer}/{stage}")


class Engine:
    def __init__(self, cfg: dict, weights, calib):
        self.cfg = cfg
        self.weights = weights
        self.calib = calib
        self.order = list(weights.layer_names)  # model chain order
        self.quant_cache = {}
        self.eval_cache = {}
        self.distill_cache = {}
        self.student_outputs = {}
        self.prev_eval_keys = {}
        self.teacher_outputs = None
        self.failures = []
        self.last_invalidation = {"invalidated": [], "reused": []}
        self.last_results = None
        self._lock = threading.Lock()

    # -- keys -------------------------------------------------------------
    def _quant_key(self, name: str) -> str:
        lfp = self.weights.layer_fingerprints[name]
        if name in self.cfg["whitelist"]:
            return fingerprint({"whitelist": True, "layer": name, "wfp": lfp})
        return fingerprint(
            {
                "layer": name,
                "wfp": lfp,
                "params": layer_params(self.cfg, name),
                "sig_digits": self.cfg["rounding"]["scale_sig_digits"],
            }
        )

    def _eval_keys(self, qkeys: dict) -> dict:
        keys = {}
        prefix = []
        for name in self.order:
            prefix.append(qkeys[name])
            keys[name] = fingerprint(
                {
                    "calib": self.calib.fingerprint,
                    "teacher": self.weights.fingerprint,
                    "prefix": list(prefix),
                    "layer": name,
                }
            )
        return keys

    def _distill_key(self, eval_key: str) -> str:
        return fingerprint(
            {
                "eval": eval_key,
                "temperature": self.cfg["distill"]["temperature"],
                "alpha": self.cfg["distill"]["soft_label_weight"],
            }
        )

    def _delay(self) -> None:
        ms = self.cfg["interaction"].get("compute_delay_ms_per_layer", 0)
        if ms:
            time.sleep(ms / 1000.0)

    # -- stage workers ------------------------------------------------------
    def _quant_task(self, name: str):
        _fail_hook(name, "quant")
        self._delay()
        params = layer_params(self.cfg, name)
        w = self.weights.arrays[name]
        sig = self.cfg["rounding"]["scale_sig_digits"]
        both = quantize.quantize_both_schemes(
            w, params["bits"], params["group_size"], params["per_channel"], sig
        )
        active = params["scheme"]
        return {
            "params": params,
            "active_scheme": active,
            "schemes": {
                s: {
                    "scales": both[s]["scales"],
                    "zero_points": both[s]["zero_points"],
                    "dequant_digest": both[s]["dequant_digest"],
                    "error": both[s]["error"],
                }
                for s in ("symmetric", "asymmetric")
            },
            "_dequant": both[active]["dequant"],
        }

    def _eval_task(self, name: str):
        _fail_hook(name, "eval")
        self._delay()
        student = self.student_outputs[name]
        teacher = self.teacher_outputs[name]
        result = {
            "vs_teacher": evaluate.error_stats(student, teacher),
            "per_sample_rmse": evaluate.per_sample_errors(student, teacher),
        }
        if name == self.order[-1]:
            result["vs_reference"] = evaluate.error_stats(student, self.calib.Y_ref)
        return result

    def _distill_task(self, name: str):
        _fail_hook(name, "distill")
        self._delay()
        out = evaluate.distill_correct(
            self.teacher_outputs[name],
            self.student_outputs[name],
            float(self.cfg["distill"]["temperature"]),
            float(self.cfg["distill"]["soft_label_weight"]),
        )
        return {
            "temperature": self.cfg["distill"]["temperature"],
            "soft_label_weight": self.cfg["distill"]["soft_label_weight"],
            "error_before": out["error_before"],
            "error_after": out["error_after"],
            "stable": out["stable"],
            "per_sample_correction_l2": out["per_sample_correction_l2"],
        }

    # -- main compute -------------------------------------------------------
    def compute(self) -> dict:
        """Run one full incremental pass; returns results + invalidation lists."""
        with self._lock:
            return self._compute_locked()

    def _compute_locked(self) -> dict:
        self.failures = []
        concurrency = self.cfg["interaction"]["concurrency"]
        whitelist = set(self.cfg["whitelist"])
        invalidated, reused = [], []

        if self.teacher_outputs is None:
            self.teacher_outputs = evaluate.forward(
                self.weights.ordered_arrays(), self.calib.X
            )

        # ---- quant stage -------------------------------------------------
        qkeys = {n: self._quant_key(n) for n in self.order}
        quant_todo = [
            n for n in self.order if n not in whitelist and qkeys[n] not in self.quant_cache
        ]
        quant_failed = set()
        if quant_todo:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {n: pool.submit(self._quant_task, n) for n in quant_todo}
                for n in quant_todo:  # model order -> deterministic assembly
                    try:
                        self.quant_cache[qkeys[n]] = futures[n].result()
                    except Exception as exc:  # noqa: BLE001 - isolation by design
                        quant_failed.add(n)
                        self.failures.append(
                            {
                                "layer": n,
                                "stage": "quant",
                                "code": E_LAYER_FAILED,
                                "exception": type(exc).__name__,
                            }
                        )
        for n in self.order:
            tag = f"quant:{n}"
            if n in whitelist or qkeys[n] in self.quant_cache and n not in quant_todo:
                reused.append(tag)
            else:
                invalidated.append(tag)

        # ---- student forward (sequential, reuse clean prefix) ------------
        eval_keys = self._eval_keys(qkeys)
        dequant_of = {}
        for n in self.order:
            if n in whitelist or n in quant_failed:
                dequant_of[n] = self.weights.arrays[n]
            else:
                dequant_of[n] = self.quant_cache[qkeys[n]]["_dequant"]

        first_dirty = None
        for i, n in enumerate(self.order):
            if self.prev_eval_keys.get(n) != eval_keys[n] or n not in self.student_outputs:
                first_dirty = i
                break
        if first_dirty is None and not self.student_outputs:
            first_dirty = 0
        if first_dirty is not None:
            acts = (
                self.calib.X
                if first_dirty == 0
                else self.student_outputs[self.order[first_dirty - 1]]
            )
            last = len(self.order) - 1
            for i in range(first_dirty, len(self.order)):
                n = self.order[i]
                acts = acts @ np.asarray(dequant_of[n], dtype=np.float64).T
                if i != last:
                    acts = np.tanh(acts)
                self.student_outputs[n] = acts
        self.prev_eval_keys = dict(eval_keys)

        # ---- eval stage ----------------------------------------------------
        eval_todo = [n for n in self.order if eval_keys[n] not in self.eval_cache]
        eval_failed = set(quant_failed)
        for n in quant_failed:
            self.failures.append(
                {
                    "layer": n,
                    "stage": "eval",
                    "code": E_DEPENDENCY_FAILED,
                    "exception": "QuantFailure",
                }
            )
        eval_todo = [n for n in eval_todo if n not in eval_failed]
        if eval_todo:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {n: pool.submit(self._eval_task, n) for n in eval_todo}
                for n in eval_todo:
                    try:
                        self.eval_cache[eval_keys[n]] = futures[n].result()
                    except Exception as exc:  # noqa: BLE001
                        eval_failed.add(n)
                        self.failures.append(
                            {
                                "layer": n,
                                "stage": "eval",
                                "code": E_LAYER_FAILED,
                                "exception": type(exc).__name__,
                            }
                        )
        for n in self.order:
            tag = f"eval:{n}"
            if n in eval_failed:
                invalidated.append(tag)
            elif n in eval_todo:
                invalidated.append(tag)
            else:
                reused.append(tag)

        # ---- distill stage -------------------------------------------------
        dkeys = {n: self._distill_key(eval_keys[n]) for n in self.order}
        distill_todo = [
            n
            for n in self.order
            if n not in whitelist and n not in eval_failed and dkeys[n] not in self.distill_cache
        ]
        distill_failed = set()
        for n in self.order:
            if n not in whitelist and n in eval_failed:
                distill_failed.add(n)
                self.failures.append(
                    {
                        "layer": n,
                        "stage": "distill",
                        "code": E_DEPENDENCY_FAILED,
                        "exception": "EvalFailure",
                    }
                )
        if distill_todo:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {n: pool.submit(self._distill_task, n) for n in distill_todo}
                for n in distill_todo:
                    try:
                        self.distill_cache[dkeys[n]] = futures[n].result()
                    except Exception as exc:  # noqa: BLE001
                        distill_failed.add(n)
                        self.failures.append(
                            {
                                "layer": n,
                                "stage": "distill",
                                "code": E_LAYER_FAILED,
                                "exception": type(exc).__name__,
                            }
                        )
        for n in self.order:
            tag = f"distill:{n}"
            if n in whitelist:
                reused.append(tag)
            elif n in distill_todo or n in distill_failed:
                invalidated.append(tag)
            else:
                reused.append(tag)

        # ---- assemble (sorted by layer name for stable output order) -------
        layers = []
        for n in sorted(self.order):
            entry = {"name": n, "whitelist": n in whitelist, "quant": None, "eval": None, "distill": None}
            if n not in whitelist and qkeys[n] in self.quant_cache:
                q = self.quant_cache[qkeys[n]]
                entry["quant"] = {
                    "params": q["params"],
                    "active_scheme": q["active_scheme"],
                    "schemes": q["schemes"],
                    "dequant_digest": self.quant_cache[qkeys[n]]["schemes"][q["active_scheme"]]["dequant_digest"],
                }
            if eval_keys[n] in self.eval_cache:
                entry["eval"] = self.eval_cache[eval_keys[n]]
            if n not in whitelist and dkeys[n] in self.distill_cache:
                entry["distill"] = self.distill_cache[dkeys[n]]
            layers.append(entry)

        self.failures.sort(key=lambda f: (f["layer"], f["stage"]))
        self.last_invalidation = {"invalidated": invalidated, "reused": reused}
        results = {
            "layers": layers,
            "failures": list(self.failures),
            "invalidation": self.last_invalidation,
            "summary": self._summary(layers),
        }
        self.last_results = results
        return results

    def _summary(self, layers) -> dict:
        completed = sum(1 for l in layers if l["eval"] is not None)
        stable = sorted(l["name"] for l in layers if l["distill"] and l["distill"]["stable"])
        unstable = sorted(l["name"] for l in layers if l["distill"] and not l["distill"]["stable"])
        final_entry = next((l for l in layers if l["name"] == self.order[-1]), None)
        final = final_entry["eval"] if final_entry else None
        return {
            "total_layers": len(layers),
            "completed_layers": completed,
            "failed_layers": sorted({f["layer"] for f in self.failures}),
            "final_layer_error": final["vs_teacher"] if final else None,
            "distill_stable_layers": stable,
            "distill_unstable_layers": unstable,
        }
