"""校准配置加载与校验。

校验违规统一映射到 E_CONFIG_VALUE（位宽/档位枚举、正容差等），
注入清单越界在加载阶段映射到 E_ROUND_OUT_OF_RANGE。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ErrorCollector, ErrorCode
from .fixedpoint import ALLOWED_FRAC_BITS, ALLOWED_QUANT_GEARS
from .hashing import canonical_json, fingerprint

ALLOWED_NORMALIZATION = ("softmax",)
ALLOWED_LINK_TYPES = ("init", "same_speaker", "speaker_switch")


@dataclass
class CalibConfig:
    initial_weights: list[float]
    normalization: str
    gain_q: int
    decay_q: int
    frac_bits: int
    quant_gear: int
    adjacent_tolerance: float
    batch_segment_budget: int
    max_workers: int
    snapshot_interval_rounds: int
    speaker_anchors_hz: dict[str, float]
    transition_weight_q: int
    same_speaker_weight_q: int
    min_segment_rounds: int
    max_gap_rounds: int
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    injections: list[dict] = field(default_factory=list)
    source: dict = field(default_factory=dict)
    config_hash: str = ""

    @property
    def param_fingerprint(self) -> str:
        # 校准参数指纹：只包含影响校准数值的参数。
        params = {
            "gain_q": self.gain_q,
            "decay_q": self.decay_q,
            "frac_bits": self.frac_bits,
            "quant_gear": self.quant_gear,
            "transition_weight_q": self.transition_weight_q,
            "same_speaker_weight_q": self.same_speaker_weight_q,
        }
        return fingerprint(params)

    @property
    def evidence_param_fingerprint(self) -> str:
        params = {
            "initial_weights": self.initial_weights,
            "normalization": self.normalization,
            "frac_bits": self.frac_bits,
            "quant_gear": self.quant_gear,
            "frame_length_ms": self.frame_length_ms,
            "frame_shift_ms": self.frame_shift_ms,
        }
        return fingerprint(params)

    @property
    def injection_hash(self) -> str:
        return fingerprint(self.injections)

    def full_hash(self) -> str:
        return fingerprint(self.source)


def _require(cond: bool, errors: ErrorCollector, message: str) -> None:
    if not cond:
        errors.add(ErrorCode.E_CONFIG_VALUE, "config", message)


def load_config(path: str | Path, errors: ErrorCollector,
                expected_rounds: int | None = None) -> CalibConfig | None:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.add(ErrorCode.E_CONFIG_VALUE, "config", f"cannot parse config: {exc}",
                   file=str(path))
        return None

    required = [
        "initial_weights", "normalization", "calibration_gain", "decay",
        "fixed_point", "adjacent_tolerance", "batch_segment_budget",
        "max_workers", "snapshot_interval_rounds", "speaker_anchors_hz",
        "transition_weight", "same_speaker_weight", "min_segment_rounds",
        "max_gap_rounds",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        errors.add(ErrorCode.E_CONFIG_VALUE, "config",
                   f"missing config keys: {missing}", file=str(path))
        return None

    fp_raw = raw["fixed_point"]
    frac_bits = fp_raw.get("frac_bits")
    quant_gear = fp_raw.get("quant_gear")
    _require(frac_bits in ALLOWED_FRAC_BITS, errors,
             f"frac_bits {frac_bits} not in {ALLOWED_FRAC_BITS}")
    _require(quant_gear in ALLOWED_QUANT_GEARS, errors,
             f"quant_gear {quant_gear} not in {ALLOWED_QUANT_GEARS}")

    weights = raw["initial_weights"]
    _require(isinstance(weights, list) and len(weights) == 3
             and all(isinstance(w, (int, float)) and w > 0 for w in weights),
             errors, "initial_weights must be 3 positive numbers")
    _require(raw["normalization"] in ALLOWED_NORMALIZATION, errors,
             f"normalization must be one of {ALLOWED_NORMALIZATION}")
    _require(isinstance(raw["calibration_gain"], (int, float))
             and 0 < raw["calibration_gain"] <= 1.0,
             errors, "calibration_gain must be in (0, 1]")
    _require(isinstance(raw["decay"], (int, float)) and 0 <= raw["decay"] < 1.0,
             errors, "decay must be in [0, 1)")
    _require(isinstance(raw["adjacent_tolerance"], (int, float))
             and raw["adjacent_tolerance"] > 0,
             errors, "adjacent_tolerance must be positive")
    _require(isinstance(raw["batch_segment_budget"], int)
             and raw["batch_segment_budget"] > 0,
             errors, "batch_segment_budget must be positive int")
    _require(isinstance(raw["max_workers"], int) and raw["max_workers"] >= 1,
             errors, "max_workers must be >= 1")
    _require(isinstance(raw["snapshot_interval_rounds"], int)
             and raw["snapshot_interval_rounds"] >= 1,
             errors, "snapshot_interval_rounds must be positive int")
    anchors = raw["speaker_anchors_hz"]
    _require(isinstance(anchors, dict) and len(anchors) > 0
             and all(isinstance(k, str) and k.strip()
                     and isinstance(v, (int, float)) and v > 0
                     for k, v in anchors.items()),
             errors, "speaker_anchors_hz must be non-empty {speaker: positive_hz}")
    _require(isinstance(raw["transition_weight"], (int, float))
             and 0 < raw["transition_weight"] <= 1,
             errors, "transition_weight must be in (0,1]")
    _require(isinstance(raw["same_speaker_weight"], (int, float))
             and 0 < raw["same_speaker_weight"] <= 1,
             errors, "same_speaker_weight must be in (0,1]")
    _require(isinstance(raw["min_segment_rounds"], int)
             and raw["min_segment_rounds"] >= 1,
             errors, "min_segment_rounds must be >= 1")
    _require(isinstance(raw["max_gap_rounds"], int)
             and raw["max_gap_rounds"] >= 0,
             errors, "max_gap_rounds must be >= 0")

    injections = raw.get("injections", [])
    _require(isinstance(injections, list), errors, "injections must be a list")
    for injection in injections:
        rnd = injection.get("round")
        if expected_rounds is not None and (
                not isinstance(rnd, int) or rnd < 0 or rnd >= expected_rounds):
            errors.add(ErrorCode.E_ROUND_OUT_OF_RANGE, "injection",
                       f"injection {injection.get('injection_id')} targets round {rnd}, "
                       f"valid range [0, {expected_rounds - 1}]",
                       round=rnd if isinstance(rnd, int) else None,
                       injection_id=injection.get("injection_id"))

    if any(record.code == ErrorCode.E_CONFIG_VALUE for record in errors.records):
        return None

    scale = 1 << frac_bits
    config = CalibConfig(
        initial_weights=[float(w) for w in weights],
        normalization=raw["normalization"],
        gain_q=int(round(raw["calibration_gain"] * scale)),
        decay_q=int(round(raw["decay"] * scale)),
        frac_bits=frac_bits,
        quant_gear=quant_gear,
        adjacent_tolerance=float(raw["adjacent_tolerance"]),
        batch_segment_budget=raw["batch_segment_budget"],
        max_workers=raw["max_workers"],
        snapshot_interval_rounds=raw["snapshot_interval_rounds"],
        speaker_anchors_hz={k: float(v) for k, v in anchors.items()},
        transition_weight_q=int(round(raw["transition_weight"] * scale)),
        same_speaker_weight_q=int(round(raw["same_speaker_weight"] * scale)),
        min_segment_rounds=raw["min_segment_rounds"],
        max_gap_rounds=raw["max_gap_rounds"],
        frame_length_ms=float(raw.get("frame_length_ms", 25.0)),
        frame_shift_ms=float(raw.get("frame_shift_ms", 10.0)),
        injections=injections,
        source=raw,
    )
    config.config_hash = config.full_hash()
    return config


def load_injection_file(path: str | Path, errors: ErrorCollector,
                        expected_rounds: int) -> list[dict]:
    """独立注入配置文件（与主配置注入清单合并前先做越界校验）。"""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.add(ErrorCode.E_CONFIG_VALUE, "injection",
                   f"cannot parse injection file: {exc}", file=str(path))
        return []
    items = raw.get("injections", []) if isinstance(raw, dict) else []
    valid = []
    for item in items:
        rnd = item.get("round")
        if not isinstance(rnd, int) or rnd < 0 or rnd >= expected_rounds:
            errors.add(ErrorCode.E_ROUND_OUT_OF_RANGE, "injection",
                       f"injection {item.get('injection_id')} round {rnd} out of range",
                       round=rnd if isinstance(rnd, int) else None,
                       injection_id=item.get("injection_id"))
            continue
        valid.append(item)
    return valid
