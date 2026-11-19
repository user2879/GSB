"""逻辑拍 -> 毫秒映射：速度关键点分段线性插值。"""

from __future__ import annotations

from bisect import bisect_right

from .errors import E_BEAT_RANGE, E_VELOCITY_MONO, PipelineError
from .model import round6


class TempoMap:
    """不可变速度曲线。点按拍号严格递增，beat=0 必须存在。"""

    def __init__(self, points: list[dict], total_beats: float | None = None):
        if not points:
            raise PipelineError(E_BEAT_RANGE, "速度曲线关键点为空")
        beats = [float(p["beat"]) for p in points]
        bpms = [float(p["bpm"]) for p in points]
        if beats[0] != 0.0:
            raise PipelineError(
                E_BEAT_RANGE, "速度曲线首个关键点必须位于 beat=0", {"beat": beats[0]}
            )
        for beat in beats:
            if beat < 0:
                raise PipelineError(E_BEAT_RANGE, "速度关键点拍号为负", {"beat": beat})
        for prev, beat in zip(beats, beats[1:]):
            if beat <= prev:
                raise PipelineError(
                    E_VELOCITY_MONO,
                    "速度曲线关键点拍号必须严格递增",
                    {"prev_beat": prev, "beat": beat},
                )
        for bpm in bpms:
            if bpm <= 0:
                raise PipelineError(E_BEAT_RANGE, "速度 bpm 必须为正", {"bpm": bpm})
        if total_beats is not None and beats[-1] > float(total_beats) + 1e-12:
            raise PipelineError(
                E_BEAT_RANGE,
                "速度关键点超出乐谱总拍长",
                {"beat": beats[-1], "total_beats": total_beats},
            )
        self.points = [{"beat": b, "bpm": q} for b, q in zip(beats, bpms)]
        self._beats = beats
        self._bpms = bpms
        self._anchor_ms = self._build_anchors()

    def _build_anchors(self) -> list[float]:
        anchors = [0.0]
        for idx in range(len(self._beats) - 1):
            b0, b1 = self._beats[idx], self._beats[idx + 1]
            q0, q1 = self._bpms[idx], self._bpms[idx + 1]
            # 段内 bpm 线性变化：毫秒增量 = 拍差 * 60000 / 段内平均 bpm
            avg_bpm = (q0 + q1) / 2.0
            anchors.append(anchors[-1] + (b1 - b0) * 60000.0 / avg_bpm)
        return anchors

    def beat_to_ms(self, beat: float) -> float:
        """返回未取整毫秒值；取整由调用方统一 round6。"""
        beat = float(beat)
        if beat <= 0:
            return beat * 60000.0 / self._bpms[0]
        if beat >= self._beats[-1]:
            idx = len(self._beats) - 1
            return self._anchor_ms[idx] + (beat - self._beats[idx]) * 60000.0 / self._bpms[idx]
        idx = bisect_right(self._beats, beat) - 1
        b0, b1 = self._beats[idx], self._beats[idx + 1]
        q0, q1 = self._bpms[idx], self._bpms[idx + 1]
        # 段内 bpm 线性插值后积分：解析解
        span = b1 - b0
        q_at = q0 + (q1 - q0) * (beat - b0) / span
        avg = (q0 + q_at) / 2.0
        return self._anchor_ms[idx] + (beat - b0) * 60000.0 / avg

    def beat_to_ms_r(self, beat: float) -> float:
        return round6(self.beat_to_ms(beat))

    def __eq__(self, other) -> bool:
        return isinstance(other, TempoMap) and self.points == other.points

    def __hash__(self):
        return hash(tuple((p["beat"], p["bpm"]) for p in self.points))
