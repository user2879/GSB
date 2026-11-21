"""Fusion pipeline orchestration.

Runs the three-channel evidence scoring, the fixed-point fusion chain at
the configured precision tier and one tier higher, stage-by-stage error
propagation tracking, budget-constrained concurrent scheduling and
content-fingerprint caching.  The report is assembled in a fully
deterministic order so concurrent and serial runs are field-identical.
"""

from __future__ import annotations

import math

from . import codes, fusion, scheduler, scoring
from .cache import StageCache
from .errorprop import compare_tiers, higher_tier
from .fingerprint import fingerprint

STAGE_ORDER = ("normalize", "quantize", "weighted_sum", "clip")


def _runtime_failure(turn_id, stage, code, detail, injection_id=None):
    entry = {"code": code, "turn_id": turn_id, "stage": stage, "detail": detail}
    if injection_id is not None:
        entry["injection_id"] = injection_id
    return entry


class Pipeline:
    def __init__(self, turns, acoustics, tools, lexicon, config, cache=None):
        self.turns = {t["turn_id"]: t for t in turns}
        self.acoustics = acoustics
        self.tools = tools
        self.lexicon = lexicon
        self.config = config
        self.cache = cache if cache is not None else StageCache()
        self.quantile_keys = [str(q) for q in config.get("acoustic_quantiles", ["0.5"])]
        self.sig_digits = config.get("significant_digits", 6)
        self.expected_dimension = config.get("expected_tool_dimension", "sentiment_polarity")
        self.reuse = []
        self.recompute = []
        self.failures = []

    # ------------------------------------------------------------------
    # Evidence scoring (per turn, exception-isolated)
    # ------------------------------------------------------------------

    def _evidence(self, turn_id):
        turn = self.turns[turn_id]
        evidence = {"text": None, "acoustic": None, "tool": None}
        raw = {"text": None, "acoustic": None, "tool": None}
        turn_codes = []
        failures = []

        try:
            text = scoring.score_text(turn, self.lexicon)
            evidence["text"] = text
            raw["text"] = text["score"]
        except Exception as exc:  # noqa: BLE001 - isolation per requirement 8
            failures.append(_runtime_failure(turn_id, "score_text",
                                             codes.E_FUSE_EVIDENCE_MISSING, str(exc)))

        acoustic = self.acoustics.get(turn_id)
        if acoustic is None:
            turn_codes.append(codes.E_FUSE_EVIDENCE_MISSING)
            failures.append(_runtime_failure(turn_id, "score_acoustic",
                                             codes.E_FUSE_EVIDENCE_MISSING,
                                             "no acoustic evidence for turn"))
        else:
            try:
                stats = [acoustic["pitch_mean"], acoustic["pitch_var"],
                         *acoustic["energy_quantiles"].values()]
                if not all(math.isfinite(v) for v in stats):
                    raise ValueError("non-finite acoustic statistic")
                ac = scoring.score_acoustic(acoustic, self.quantile_keys)
                evidence["acoustic"] = ac
                raw["acoustic"] = ac["score"]
            except Exception as exc:  # noqa: BLE001
                turn_codes.append(codes.E_LOAD_ACOUSTIC_NONFINITE)
                failures.append(_runtime_failure(
                    turn_id, "score_acoustic", codes.E_LOAD_ACOUSTIC_NONFINITE, str(exc),
                    injection_id=acoustic.get("_injection_id")))

        calls = turn.get("tool_calls", [])
        if not calls:
            turn_codes.append(codes.E_FUSE_EVIDENCE_MISSING)
        else:
            call = calls[0]
            tool = self.tools.get(call.get("tool"))
            if tool is None:
                turn_codes.append(codes.E_FUSE_EVIDENCE_MISSING)
                failures.append(_runtime_failure(turn_id, "call_tool",
                                                 codes.E_FUSE_EVIDENCE_MISSING,
                                                 f"tool {call.get('tool')!r} not in manifest"))
            else:
                try:
                    result = scoring.call_tool(tool, call.get("params", {}))
                    evidence["tool"] = result
                    if result["dimension"] != self.expected_dimension:
                        turn_codes.append(codes.E_FUSE_DIM_INCOMPAT)
                        failures.append(_runtime_failure(
                            turn_id, "call_tool", codes.E_FUSE_DIM_INCOMPAT,
                            f"dimension {result['dimension']!r} incompatible with "
                            f"{self.expected_dimension!r}",
                            injection_id=tool.get("_injection_id")))
                    else:
                        raw["tool"] = result["value"]
                except Exception as exc:  # noqa: BLE001
                    turn_codes.append(codes.E_FUSE_EVIDENCE_MISSING)
                    failures.append(_runtime_failure(turn_id, "call_tool",
                                                     codes.E_FUSE_EVIDENCE_MISSING, str(exc)))

        return {
            "turn_id": turn_id,
            "evidence": evidence,
            "raw": raw,
            "codes": turn_codes,
            "failures": failures,
        }

    # ------------------------------------------------------------------
    # Cached stage products
    # ------------------------------------------------------------------

    def _cached(self, stage, turn_id, tier, upstream_fps, params, compute):
        value, reused, key, refs = self.cache.lookup_or(
            stage, turn_id, upstream_fps, params, compute)
        record = {"stage": stage, "turn_id": turn_id, "tier": tier, "key": key}
        (self.reuse if reused else self.recompute).append(record)
        return value, key, refs

    def _stage_params(self, stage, tier_bits, quantum):
        """Parameter fingerprint scope per stage: a command only invalidates
        the stages whose parameters it actually touches."""
        cfg = self.config
        if stage == "normalize":
            return {"sig_digits": self.sig_digits}
        if stage == "quantize":
            return {"frac_bits": tier_bits, "quantum": quantum}
        if stage == "weighted_sum":
            return {"weights": cfg["weights"], "temperature": cfg["temperature"],
                    "bias": cfg.get("bias", 0), "frac_bits": tier_bits}
        if stage == "clip":
            return {"interval": cfg["confidence_interval"],
                    "threshold": cfg.get("sentiment_threshold", 0.5),
                    "frac_bits": tier_bits}
        return {  # composite fuse stage
            "weights": cfg["weights"],
            "temperature": cfg["temperature"],
            "bias": cfg.get("bias", 0),
            "interval": cfg["confidence_interval"],
            "threshold": cfg.get("sentiment_threshold", 0.5),
            "frac_bits": tier_bits,
            "quantum": quantum,
            "sig_digits": self.sig_digits,
        }

    def _turn_chain(self, turn_id, raw, stats_key, stats, tier_bits, quantum, tier_name):
        evidence_fp = fingerprint({"turn_id": turn_id, "raw": raw})

        normalize, norm_key, _ = self._cached(
            "normalize", turn_id, tier_name, [stats_key, evidence_fp],
            self._stage_params("normalize", tier_bits, quantum),
            lambda: {ch: fusion.normalize_value(raw.get(ch), stats.get(ch), self.sig_digits)
                     for ch in fusion.CHANNELS})

        quantized, quant_key, _ = self._cached(
            "quantize", turn_id, tier_name, [norm_key],
            self._stage_params("quantize", tier_bits, quantum),
            lambda: {ch: (fusion.quantize(normalize[ch], tier_bits, quantum)
                          if normalize.get(ch) is not None else None)
                     for ch in fusion.CHANNELS})

        fused, fuse_key, _ = self._cached(
            "fuse", turn_id, tier_name, [quant_key],
            self._stage_params("fuse", tier_bits, quantum),
            lambda: fusion.fuse_turn(normalize, self.config, tier_bits, quantum))

        # Register the weighted_sum and clip products with their own
        # stage-scoped parameter fingerprints and upstream references.
        _, ws_key, _ = self._cached(
            "weighted_sum", turn_id, tier_name, [quant_key],
            self._stage_params("weighted_sum", tier_bits, quantum),
            lambda: fused.get("weighted_sum"))
        self._cached(
            "clip", turn_id, tier_name, [ws_key],
            self._stage_params("clip", tier_bits, quantum),
            lambda: {"confidence_int": fused.get("confidence_int"),
                     "confidence": fused.get("confidence")})

        return {
            "normalize": normalize,
            "quantize": quantized,
            "weighted_sum": fused.get("weighted_sum"),
            "clip": {"confidence_int": fused.get("confidence_int"),
                     "confidence": fused.get("confidence")},
            "codes": fused["codes"],
            "clipped": fused["clipped"],
        }

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self):
        cfg = self.config
        turn_ids = sorted(self.turns)
        tier_lo = (cfg["precision"]["frac_bits"], cfg["precision"]["quantum"])
        tier_hi = higher_tier(cfg)

        # Budget planning happens before any execution, in turn-id order.
        in_batch, pending, deductions = scheduler.plan_budget(
            turn_ids, cfg["fusion_point_budget"])

        # Stage: evidence scoring, concurrent, exception-isolated per turn.
        tasks = [((tid, "evidence"), (lambda tid=tid: self._evidence(tid)))
                 for tid in in_batch]
        evidences = scheduler.run_tasks(tasks, cfg["concurrency"])
        evidence_by_turn = {e["turn_id"]: e for e in evidences}

        # Stage: normalization statistics (batch level, fixed turn order).
        raw_rows = [{"turn_id": tid, "raw": evidence_by_turn[tid]["raw"]}
                    for tid in in_batch]
        raw_fp = fingerprint(raw_rows)
        stats_params = {"sig_digits": self.sig_digits, "channels": fusion.CHANNELS}
        stats, stats_reused, stats_key, _ = self.cache.lookup_or(
            "normalize_stats", "*", [raw_fp], stats_params,
            lambda: fusion.normalization_stats(raw_rows))
        (self.reuse if stats_reused else self.recompute).append(
            {"stage": "normalize_stats", "turn_id": "*", "tier": "-", "key": stats_key})

        # Stage: per-turn fusion chain at both tiers, concurrent.
        def chain(tid):
            raw = evidence_by_turn[tid]["raw"]
            lo = self._turn_chain(tid, raw, stats_key, stats, tier_lo[0], tier_lo[1], "config")
            hi = self._turn_chain(tid, raw, stats_key, stats, tier_hi[0], tier_hi[1], "higher")
            return tid, lo, hi

        chain_tasks = [((tid, "fuse"), (lambda tid=tid: chain(tid))) for tid in in_batch]
        chains = scheduler.run_tasks(chain_tasks, cfg["concurrency"])

        # Assemble the report in deterministic turn-id order.
        turn_reports = []
        for tid, lo, hi in chains:
            evidence = evidence_by_turn[tid]
            error = compare_tiers(lo, hi, cfg)
            turn_codes = list(evidence["codes"])
            for code in lo["codes"]:
                if code not in turn_codes:
                    turn_codes.append(code)
            turn_reports.append({
                "turn_id": tid,
                "raw": evidence["raw"],
                "evidence": evidence["evidence"],
                "codes": turn_codes,
                "tier_config": lo,
                "tier_higher": hi,
                "confidence": lo["clip"]["confidence"],
                "confidence_int": lo["clip"]["confidence_int"],
                "classification": error["classification"]["tier_config"],
                "error_propagation": error,
            })
            self.failures.extend(evidence["failures"])

        for tid in pending:
            turn_reports.append({
                "turn_id": tid,
                "raw": None,
                "evidence": None,
                "codes": [codes.E_FUSE_BUDGET_PENDING],
                "tier_config": None,
                "tier_higher": None,
                "confidence": None,
                "confidence_int": None,
                "classification": None,
                "error_propagation": None,
            })
        turn_reports.sort(key=lambda r: r["turn_id"])

        # Fixed-order failure list: turn_id, then stage.
        self.failures.sort(key=lambda f: (f.get("turn_id") or "", f.get("stage") or ""))

        # Deterministic reuse / recompute lists.
        stage_rank = {name: idx for idx, name in
                      enumerate(("normalize_stats",) + STAGE_ORDER + ("fuse",))}

        def order_key(r):
            return (r["turn_id"], stage_rank.get(r["stage"], 99), r["tier"])

        self.reuse.sort(key=order_key)
        self.recompute.sort(key=order_key)

        completed = [r for r in turn_reports if r["confidence"] is not None]
        lo_i, hi_i = cfg["confidence_interval"]
        mid = lo_i + (hi_i - lo_i) / 2
        distribution = {
            "low": sum(1 for r in completed if r["confidence"] < mid),
            "high": sum(1 for r in completed if r["confidence"] >= mid),
        }

        return {
            "config_fingerprint": fingerprint(cfg),
            "precision_tiers": {"config": list(tier_lo), "higher": list(tier_hi)},
            "normalize_stats": stats,
            "turns": turn_reports,
            "failures": self.failures,
            "budget": scheduler.budget_report(cfg["fusion_point_budget"], deductions, pending),
            "reuse": self.reuse,
            "recompute": self.recompute,
            "summary": {
                "completed_turns": len(completed),
                "total_turns": len(turn_reports),
                "pending_turns": len(pending),
                "confidence_distribution": distribution,
                "flips": [r["turn_id"] for r in turn_reports
                          if r["error_propagation"] and r["error_propagation"]["flip"]],
            },
        }
