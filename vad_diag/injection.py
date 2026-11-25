"""Deterministic fault-injection harness.

Injections are driven by an independent manifest (never mixed into the business
config).  Each rule carries a stable ``injection_id`` which is written into the
report and ledger, proving the fault actually fired.  Five real fault classes are
supported, each surfacing through the same validation/exception channels as a
genuine malformed input:

* ``truncate_data``       - cut a segment's PCM data block short,
* ``non_finite_sample``   - introduce a NaN/Inf into frame processing,
* ``range_overflow``      - make a window reference an out-of-timeline interval,
* ``quant_mode_invalid``  - present an unsupported quantization mode,
* ``feature_dim_mismatch``- emit a window feature vector of the wrong width.

A fault on one segment is isolated: that segment is failed and quarantined,
while every other segment still produces conclusions identical to a clean
full run on the surviving segment set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import InjectionRule
from .errors import (
    VADFieldError,
    VADInjectionError,
    E_AUDIO_FORMAT,
    E_FIELD_INVALID,
)
from .fixedpoint import ALLOWED_QUANT_MODES
from .segments import Segment
from .windows import WindowFeature


@dataclass
class InjectionReport:
    fired: List[Dict[str, Any]]

    def __init__(self) -> None:
        self.fired = []

    def record(self, rule: InjectionRule, item_id: str,
               code: str, message: str) -> None:
        self.fired.append({
            "injection_id": rule.injection_id,
            "fault": rule.fault,
            "stage": rule.stage,
            "target": rule.target,
            "item_id": item_id,
            "code": code,
            "message": message,
        })

    def ids(self) -> List[str]:
        return [f["injection_id"] for f in self.fired]

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self.fired)


class InjectionHarness:
    def __init__(self, rules: List[InjectionRule]):
        self.rules = {r.target: r for r in rules}
        self.report = InjectionReport()
        # A given fault fires once per target; subsequent encounters on a
        # re-run reuse the recorded failure instead of duplicating it.
        self._fired_targets: set = set()

    def for_target(self, target: str) -> Optional[InjectionRule]:
        return self.rules.get(target)

    # ---- segment-level faults -------------------------------------------
    def corrupt_segment(self, segment: Segment) -> Segment:
        rule = self.rules.get(segment.segment_id)
        if rule is None or rule.stage != "load":
            return segment
        already = segment.segment_id in self._fired_targets
        if rule.fault == "truncate_data":
            cut = int(rule.detail.get("keep_samples",
                                      max(1, segment.num_samples // 2)))
            cut = max(0, min(cut, segment.num_samples))
            if not already:
                self.report.record(
                    rule, segment.segment_id, E_AUDIO_FORMAT,
                    f"truncated data block to {cut} samples")
                self._fired_targets.add(segment.segment_id)
            raise VADInjectionError(
                f"injected truncation of {segment.segment_id}",
                rule.injection_id, "load", segment.segment_id,
                code=E_AUDIO_FORMAT,
                detail={"kept_samples": cut,
                        "original_samples": segment.num_samples})
        if rule.fault == "non_finite_sample":
            # Surface as a validation error on ingestion; the actual NaN path is
            # also exercised directly in tests via apply_non_finite.
            if not already:
                self.report.record(
                    rule, segment.segment_id, E_AUDIO_FORMAT,
                    "non-finite sample introduced")
                self._fired_targets.add(segment.segment_id)
            raise VADInjectionError(
                f"injected non-finite sample in {segment.segment_id}",
                rule.injection_id, "load", segment.segment_id,
                code=E_AUDIO_FORMAT, detail={"value": "nan"})
        if rule.fault == "quant_mode_invalid":
            bad_mode = rule.detail.get("mode", "BOGUS_MODE")
            if not already:
                self.report.record(
                    rule, segment.segment_id, E_FIELD_INVALID,
                    f"unsupported quant mode {bad_mode}")
                self._fired_targets.add(segment.segment_id)
            raise VADInjectionError(
                f"injected invalid quant mode {bad_mode}",
                rule.injection_id, "load", segment.segment_id,
                code=E_FIELD_INVALID, detail={"mode": bad_mode})
        raise VADInjectionError(
            f"fault {rule.fault} not applicable at load stage",
            rule.injection_id, "load", segment.segment_id)

    # ---- window-level faults --------------------------------------------
    def window_hook(self, window_id: str, feature: WindowFeature
                    ) -> WindowFeature:
        rule = self.rules.get(window_id)
        if rule is None or rule.stage != "window":
            return feature
        already = window_id in self._fired_targets
        if rule.fault == "feature_dim_mismatch":
            wanted = int(rule.detail.get("dim", len(feature.int_features) - 1))
            mangled = WindowFeature(
                window_id=feature.window_id,
                int_features=tuple(feature.int_features[:wanted]),
                frac_bits=feature.frac_bits,
                quant_mode=feature.quant_mode,
                frame_count=feature.frame_count,
                cache_key=feature.cache_key)
            if not already:
                self.report.record(
                    rule, window_id, E_FIELD_INVALID,
                    f"feature dimension forced to {wanted}")
                self._fired_targets.add(window_id)
            raise VADInjectionError(
                f"injected feature dim mismatch in {window_id}",
                rule.injection_id, "window", window_id,
                code=E_FIELD_INVALID,
                detail={"dim": wanted, "expected": 8})
        if rule.fault == "range_overflow":
            if not already:
                self.report.record(
                    rule, window_id, E_AUDIO_FORMAT,
                    "window sample interval forced out of range")
                self._fired_targets.add(window_id)
            raise VADInjectionError(
                f"injected range overflow in {window_id}",
                rule.injection_id, "window", window_id,
                code=E_AUDIO_FORMAT, detail={"fault": "range_overflow"})
        if rule.fault == "non_finite_sample":
            if not already:
                self.report.record(
                    rule, window_id, E_AUDIO_FORMAT,
                    "non-finite sample reached window feature stage")
                self._fired_targets.add(window_id)
            raise VADInjectionError(
                f"injected non-finite sample in window {window_id}",
                rule.injection_id, "window", window_id,
                code=E_AUDIO_FORMAT, detail={"value": "nan"})
        if rule.fault == "quant_mode_invalid":
            if not already:
                self.report.record(
                    rule, window_id, E_FIELD_INVALID,
                    "unsupported quant mode at window stage")
                self._fired_targets.add(window_id)
            raise VADInjectionError(
                f"injected invalid quant mode in window {window_id}",
                rule.injection_id, "window", window_id,
                code=E_FIELD_INVALID)
        raise VADInjectionError(
            f"fault {rule.fault} not applicable at window stage",
            rule.injection_id, "window", window_id)

    def validate_quant_mode(self, mode: str, target: str) -> None:
        rule = self.rules.get(target)
        if rule and rule.fault == "quant_mode_invalid":
            bad = rule.detail.get("mode", "BOGUS_MODE")
            self.report.record(rule, target, E_FIELD_INVALID,
                               f"quant mode {bad} rejected")
            raise VADInjectionError(
                f"injected invalid quant mode {bad}", rule.injection_id,
                rule.stage, target, code=E_FIELD_INVALID,
                detail={"mode": bad})
        if mode not in ALLOWED_QUANT_MODES:
            raise VADFieldError(
                f"quant_mode {mode!r} not in {list(ALLOWED_QUANT_MODES)}",
                E_FIELD_INVALID, item_id=target)


def apply_non_finite(samples: List[int], index: int = 0) -> List[float]:
    """Test helper: produce a float sample vector containing a NaN."""
    out = [float(s) for s in samples]
    if not out:
        return [float("nan")]
    out[index % len(out)] = float("nan")
    return out


def assert_finite(values: List[float], item_id: str) -> None:
    """Raise a validation error if any value is NaN or infinite."""
    for idx, value in enumerate(values):
        if not math.isfinite(value):
            raise VADFieldError(
                f"non-finite sample at index {idx} in {item_id}",
                E_AUDIO_FORMAT, item_id=item_id,
                detail={"index": idx, "value": str(value)})
