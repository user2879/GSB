"""Joint transcode-budget and client-buffer decision engine with memoization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .buffer import BufferSimulation, simulate_buffer
from .canonical import hash_object
from .config import BudgetConfig, ClientBufferConfig, ProfileConfig
from .errors import require


@dataclass(frozen=True)
class DecisionResult:
    decision_key: str
    bandwidth_kbps: int
    effective_bandwidth_budget_kbps: int
    threshold_ms: int
    selected_profile: str
    selected_bitrate_kbps: int
    parallelism: int
    safe: bool
    ladder: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]
    sequence: tuple[dict[str, Any], ...]
    model_cache_hits: tuple[str, ...]
    model_cache_misses: tuple[str, ...]
    recomputed_decisions: tuple[str, ...]
    decision_cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_key": self.decision_key,
            "bandwidth_kbps": self.bandwidth_kbps,
            "effective_bandwidth_budget_kbps": self.effective_bandwidth_budget_kbps,
            "threshold_ms": self.threshold_ms,
            "selected_profile": self.selected_profile,
            "selected_bitrate_kbps": self.selected_bitrate_kbps,
            "parallelism": self.parallelism,
            "safe": self.safe,
            "ladder": list(self.ladder),
            "candidates": list(self.candidates),
            "sequence": list(self.sequence),
            "reuse": {
                "model_cache_hits": list(self.model_cache_hits),
                "model_cache_misses": list(self.model_cache_misses),
                "recomputed_decisions": list(self.recomputed_decisions),
                "decision_cache_hit": self.decision_cache_hit,
            },
        }


class DecisionEngine:
    def __init__(
        self,
        *,
        profiles: tuple[ProfileConfig, ...],
        ready_times_ms: dict[str, dict[int, int]],
        delay_table: dict[str, Any],
        segment_durations_ms: list[int],
        media_duration_ms: int,
        buffer_config: ClientBufferConfig,
        budget: BudgetConfig,
        bandwidth_safety_ratio: float,
    ):
        self.profiles = tuple(sorted(profiles, key=lambda item: item.video_bitrate_kbps + item.audio_bitrate_kbps))
        self.ready_times_ms = ready_times_ms
        self.delay_table = delay_table
        self.segment_durations_ms = segment_durations_ms
        self.media_duration_ms = media_duration_ms
        self.buffer_config = buffer_config
        self.budget = budget
        self.bandwidth_safety_ratio = bandwidth_safety_ratio
        self._model_cache: dict[str, BufferSimulation] = {}
        self._decision_cache: dict[str, DecisionResult] = {}

    def lower_profile(self, profile_name: str) -> str | None:
        names = [profile.name for profile in self.profiles]
        index = names.index(profile_name)
        return names[index - 1] if index > 0 else None

    def profile(self, name: str) -> ProfileConfig:
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise KeyError(name)

    def buffer_model(self, profile_name: str, threshold_ms: int) -> BufferSimulation:
        profile = self.profile(profile_name)
        delays = list(self.delay_table["delays_ms"][profile_name])
        model_key = hash_object(
            {
                "profile": profile.to_dict(),
                "ready_times_ms": self.ready_times_ms[profile_name],
                "delays_ms": delays,
                "segment_durations_ms": self.segment_durations_ms,
                "media_duration_ms": self.media_duration_ms,
                "capacity_ms": self.buffer_config.capacity_ms,
                "step_ms": self.buffer_config.step_ms,
                "resume_buffer_ms": self.buffer_config.resume_buffer_ms,
                "startup_buffer_ms": self.buffer_config.startup_buffer_ms,
                "threshold_ms": threshold_ms,
            }
        )
        if model_key in self._model_cache:
            return self._model_cache[model_key]
        dynamic_config = ClientBufferConfig(
            capacity_ms=self.buffer_config.capacity_ms,
            threshold_ms=threshold_ms,
            step_ms=self.buffer_config.step_ms,
            resume_buffer_ms=self.buffer_config.resume_buffer_ms,
            startup_buffer_ms=self.buffer_config.startup_buffer_ms,
        )
        model = simulate_buffer(
            profile=profile,
            lower_profile=self.lower_profile(profile_name),
            ready_times_ms=self.ready_times_ms[profile_name],
            delays_ms=delays,
            segment_durations_ms=self.segment_durations_ms,
            media_duration_ms=self.media_duration_ms,
            config=dynamic_config,
        )
        self._model_cache[model_key] = model
        return model

    def decide(self, *, bandwidth_kbps: int, threshold_ms: int) -> DecisionResult:
        require(bandwidth_kbps > 0, "E_BUFFER_UNDERRUN", "bandwidth must be positive")
        require(threshold_ms >= 0, "E_BUFFER_UNDERRUN", "threshold must be non-negative")
        effective_budget = int(bandwidth_kbps * self.bandwidth_safety_ratio)
        decision_key = f"bw={bandwidth_kbps}:threshold={threshold_ms}"
        if decision_key in self._decision_cache:
            cached = self._decision_cache[decision_key]
            return DecisionResult(
                **{
                    **cached.__dict__,
                    "model_cache_hits": (),
                    "model_cache_misses": (),
                    "recomputed_decisions": (),
                    "decision_cache_hit": True,
                }
            )

        eligible = [
            profile
            for profile in reversed(self.profiles)
            if profile.video_bitrate_kbps + profile.audio_bitrate_kbps <= effective_budget
        ]
        used_fallback = False
        if not eligible:
            eligible = [self.profiles[0]]
            used_fallback = True

        candidates: list[dict[str, Any]] = []
        models: dict[str, BufferSimulation] = {}
        cache_misses: list[str] = []
        cache_hits: list[str] = []
        for profile in eligible:
            model_key = self._model_key(profile.name, threshold_ms)
            if model_key in self._model_cache:
                cache_hits.append(profile.name)
            else:
                cache_misses.append(profile.name)
            model = self.buffer_model(profile.name, threshold_ms)
            models[profile.name] = model
            candidates.append(
                {
                    "profile": profile.name,
                    "bitrate_kbps": profile.video_bitrate_kbps + profile.audio_bitrate_kbps,
                    "within_bandwidth_budget": (
                        profile.video_bitrate_kbps + profile.audio_bitrate_kbps <= effective_budget
                    ),
                    "minimum_buffer_ms": model.minimum_buffer_ms,
                    "underrun_count": model.underrun_count,
                    "safe": model.safe,
                    "risk": (
                        "ok"
                        if model.safe
                        else "underrun"
                        if model.underrun_count > 0
                        else "low_watermark"
                    ),
                }
            )

        selected = eligible[-1]
        for profile in eligible:
            if models[profile.name].safe:
                selected = profile
                break
        selected_model = models[selected.name]
        parallelism = max(
            1,
            min(
                self.budget.max_workers,
                self.budget.memory_budget_mb // selected.memory_mb,
                self.budget.cpu_budget_ms // selected.cpu_ms,
                len(self.segment_durations_ms),
            ),
        )

        sequence: list[dict[str, Any]] = []
        for step_index, candidate in enumerate(candidates, start=1):
            sequence.append(
                {
                    "step": step_index,
                    "at_ms": 0,
                    "action": "evaluate_profile",
                    "profile": candidate["profile"],
                    "safe": candidate["safe"],
                    "reason": candidate["risk"],
                }
            )
        selected_reason = (
            "highest_safe_profile_within_bandwidth"
            if selected_model.safe
            else "lowest_profile_even_though_unsafe"
        )
        sequence.append(
            {
                "step": len(sequence) + 1,
                "at_ms": 0,
                "action": "select_profile_and_parallelism",
                "profile": selected.name,
                "parallelism": parallelism,
                "reason": selected_reason,
                "bandwidth_fallback": used_fallback,
                "evidence": {
                    "minimum_buffer_ms": selected_model.minimum_buffer_ms,
                    "threshold_ms": threshold_ms,
                    "underrun_count": selected_model.underrun_count,
                    "memory_limit_parallelism": self.budget.memory_budget_mb // selected.memory_mb,
                    "cpu_limit_parallelism": self.budget.cpu_budget_ms // selected.cpu_ms,
                    "worker_limit_parallelism": self.budget.max_workers,
                },
            }
        )

        result = DecisionResult(
            decision_key=decision_key,
            bandwidth_kbps=bandwidth_kbps,
            effective_bandwidth_budget_kbps=effective_budget,
            threshold_ms=threshold_ms,
            selected_profile=selected.name,
            selected_bitrate_kbps=selected.video_bitrate_kbps + selected.audio_bitrate_kbps,
            parallelism=parallelism,
            safe=selected_model.safe,
            ladder=tuple(profile.name for profile in self.profiles),
            candidates=tuple(candidates),
            sequence=tuple(sequence),
            model_cache_hits=tuple(cache_hits),
            model_cache_misses=tuple(cache_misses),
            recomputed_decisions=(decision_key,),
            decision_cache_hit=False,
        )
        self._decision_cache[decision_key] = result
        return result

    def _model_key(self, profile_name: str, threshold_ms: int) -> str:
        profile = self.profile(profile_name)
        delays = list(self.delay_table["delays_ms"][profile_name])
        return hash_object(
            {
                "profile": profile.to_dict(),
                "ready_times_ms": self.ready_times_ms[profile_name],
                "delays_ms": delays,
                "segment_durations_ms": self.segment_durations_ms,
                "media_duration_ms": self.media_duration_ms,
                "capacity_ms": self.buffer_config.capacity_ms,
                "step_ms": self.buffer_config.step_ms,
                "resume_buffer_ms": self.buffer_config.resume_buffer_ms,
                "startup_buffer_ms": self.buffer_config.startup_buffer_ms,
                "threshold_ms": threshold_ms,
            }
        )
