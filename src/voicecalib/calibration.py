"""逐轮定点校准核心。

每轮校准值完全由定点整数按固定运算顺序得到：
  link_q = same_speaker_weight_q / transition_weight_q（按衔接关系显式选择）
  cal_q  = clamp( gain_q * evidence_q + link_q * decay_q * prev_cal_q )
禁止任何数组归约默认顺序与全局随机状态。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import CalibConfig
from .evidence import EMOTIONS, StrengthArtifact
from .fixedpoint import FixedPoint
from .hashing import short_key


@dataclass
class CalibrationRow:
    round: int
    speaker: str
    link: str
    link_weight_q: int
    gain_q: int
    decay_q: int
    evidence_q: list[int]
    calib_q: list[int]
    calib_float64: list[float]
    emotion: str
    prev_round: int | None
    cache_key: str
    evidence_key: str

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "speaker": self.speaker,
            "link": self.link,
            "link_weight_q": self.link_weight_q,
            "gain_q": self.gain_q,
            "decay_q": self.decay_q,
            "emotions": list(EMOTIONS),
            "evidence_q": self.evidence_q,
            "calib_q": self.calib_q,
            "calib_float64": self.calib_float64,
            "emotion": self.emotion,
            "prev_round": self.prev_round,
            "cache_key": self.cache_key,
            "evidence_key": self.evidence_key,
        }


def link_weight_for(link: str, config: CalibConfig) -> int:
    if link == "same_speaker":
        return config.same_speaker_weight_q
    # init 与 speaker_switch 使用切换衔接权重（init 轮历史为零，权重仅作记录）。
    return config.transition_weight_q


def calibrate_round(round_index: int, speaker: str, link: str,
                    strength: StrengthArtifact, prev: CalibrationRow | None,
                    config: CalibConfig) -> CalibrationRow:
    fp = FixedPoint(config.frac_bits, config.quant_gear)
    link_q = link_weight_for(link, config)
    prev_q = prev.calib_q if prev is not None else [0] * len(EMOTIONS)

    calib_q: list[int] = []
    # 情感分量按 EMOTIONS 字典序逐个计算，顺序显式且固定。
    for emo_idx in range(len(EMOTIONS)):
        evidence_term = fp.mul_scaled(config.gain_q, strength.strength_q[emo_idx])
        history_term = fp.mul_scaled(
            fp.mul_scaled(link_q, config.decay_q),
            prev_q[emo_idx],
        )
        calib_q.append(fp.clamp(fp.add(evidence_term, history_term)))

    calib_float = [fp.to_float(v) for v in calib_q]
    # 情感判定：取最大值，并列时按情感标识字典序（第一个最大值）。
    emotion = EMOTIONS[calib_q.index(max(calib_q))]
    cache_key = short_key(
        "calib",
        strength.cache_key,
        config.param_fingerprint,
        link,
        link_q,
        prev.cache_key if prev is not None else "ZERO",
        config.frac_bits,
        config.quant_gear,
    )
    return CalibrationRow(
        round=round_index,
        speaker=speaker,
        link=link,
        link_weight_q=link_q,
        gain_q=config.gain_q,
        decay_q=config.decay_q,
        evidence_q=list(strength.strength_q),
        calib_q=calib_q,
        calib_float64=calib_float,
        emotion=emotion,
        prev_round=prev.round if prev is not None else None,
        cache_key=cache_key,
        evidence_key=strength.cache_key,
    )
