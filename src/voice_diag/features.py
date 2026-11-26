from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping

import numpy as np

from .config import AnalysisConfig, DiagnosticConfig
from .schema import CommandDef
from .util import json_hash, round_half_away_from_zero


def _hz_to_mel(frequency: float) -> float:
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filterbank(sample_rate: int, frame_length: int, bands: int) -> tuple[tuple[float, ...], ...]:
    nfft = 1 << int(math.ceil(math.log2(float(frame_length))))
    bin_count = nfft // 2 + 1
    frequencies = np.linspace(0.0, sample_rate / 2.0, bin_count, dtype=np.float64)
    mel_edges = np.linspace(
        _hz_to_mel(0.0), _hz_to_mel(sample_rate / 2.0), bands + 2, dtype=np.float64
    )
    hz_edges = np.array([_mel_to_hz(float(value)) for value in mel_edges], dtype=np.float64)
    filters: list[tuple[float, ...]] = []
    for band in range(bands):
        left = hz_edges[band]
        center = hz_edges[band + 1]
        right = hz_edges[band + 2]
        values: list[float] = []
        for frequency in frequencies:
            if frequency <= left or frequency >= right or center <= left or right <= center:
                values.append(0.0)
            elif frequency <= center:
                values.append(float((frequency - left) / (center - left)))
            else:
                values.append(float((right - frequency) / (right - center)))
        filters.append(tuple(round(value, 15) for value in values))
    return tuple(filters)


@lru_cache(maxsize=8)
def _analysis_window(frame_length: int) -> tuple[float, ...]:
    values = np.hanning(frame_length)
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class FixedWindowFeature:
    frame_indices: tuple[int, ...]
    frame_count: int
    feature_sum: tuple[int, ...]
    feature_mean_q: tuple[int, ...]
    per_frame_q: tuple[tuple[int, ...], ...]
    frame_rms: tuple[float, ...]
    frame_flatness: tuple[float, ...]
    speech_ppm: int
    quant_profile: str
    fixed_denominator: int
    feature_digest: str

    @property
    def dimensions(self) -> int:
        return len(self.feature_mean_q)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_indices": list(self.frame_indices),
            "frame_count": self.frame_count,
            "feature_sum": list(self.feature_sum),
            "feature_mean_q": list(self.feature_mean_q),
            "per_frame_q": [list(row) for row in self.per_frame_q],
            "frame_rms": list(self.frame_rms),
            "frame_flatness": list(self.frame_flatness),
            "speech_ppm": self.speech_ppm,
            "quant_profile": self.quant_profile,
            "fixed_denominator": self.fixed_denominator,
            "feature_digest": self.feature_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FixedWindowFeature":
        return cls(
            frame_indices=tuple(int(item) for item in value["frame_indices"]),
            frame_count=int(value["frame_count"]),
            feature_sum=tuple(int(item) for item in value["feature_sum"]),
            feature_mean_q=tuple(int(item) for item in value["feature_mean_q"]),
            per_frame_q=tuple(tuple(int(item) for item in row) for row in value["per_frame_q"]),
            frame_rms=tuple(float(item) for item in value["frame_rms"]),
            frame_flatness=tuple(float(item) for item in value["frame_flatness"]),
            speech_ppm=int(value["speech_ppm"]),
            quant_profile=str(value["quant_profile"]),
            fixed_denominator=int(value["fixed_denominator"]),
            feature_digest=str(value["feature_digest"]),
        )


def _speech_ppm(score: float, low: float, high: float) -> int:
    if low <= 0.0 or high <= low:
        return 0
    if score <= low:
        return max(0, min(350_000, round_half_away_from_zero(score / low * 350_000.0)))
    if score >= high:
        return max(700_000, min(1_000_000, 700_000 + round_half_away_from_zero(min(1.0, (score - high) / high) * 300_000.0)))
    return 350_000 + round_half_away_from_zero((score - low) / (high - low) * 350_000.0)


def extract_fixed_feature(
    pcm_window: np.ndarray,
    *,
    window_start: int,
    frame_indices: Iterable[int],
    analysis: AnalysisConfig,
    endpoint_low_energy: float,
    endpoint_high_energy: float,
    valid_range: tuple[int, int] | None = None,
) -> FixedWindowFeature:
    samples = np.asarray(pcm_window, dtype=np.float32) / 32768.0
    frame_length = analysis.frame_length
    frame_shift = analysis.frame_shift
    filters = _mel_filterbank(16000, frame_length, analysis.mel_bands)
    filter_array = np.asarray(filters, dtype=np.float64)
    hann = np.asarray(_analysis_window(frame_length), dtype=np.float64)
    nfft = 1 << int(math.ceil(math.log2(float(frame_length))))
    frequencies = np.linspace(0.0, 8000.0, nfft // 2 + 1, dtype=np.float64)
    fixed_denominator = analysis.fixed_denominator
    selected_frames: list[int] = []
    per_frame_q: list[tuple[int, ...]] = []
    frame_rms: list[float] = []
    frame_flatness: list[float] = []
    for frame_index in sorted(int(item) for item in frame_indices):
        frame_start = frame_index * frame_shift
        local_start = frame_start - window_start
        local_end = local_start + frame_length
        if local_start < 0 or local_end > samples.size:
            continue
        if valid_range is not None:
            overlap = max(0, min(frame_start + frame_length, valid_range[1]) - max(frame_start, valid_range[0]))
            if overlap * 2 < frame_length:
                continue
        frame = samples[local_start:local_end].astype(np.float64)
        rms = float(np.sqrt(np.mean(frame * frame)))
        spectrum = np.fft.rfft(frame * hann, n=nfft)
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        power_sum = float(np.sum(power))
        if power_sum > 0.0:
            centroid = float(np.sum(frequencies * power) / power_sum) / 8000.0
        else:
            centroid = 0.0
        shifted_power = power + 1e-15
        geometric = float(np.exp(np.mean(np.log(shifted_power))))
        arithmetic = float(np.mean(power) + 1e-15)
        flatness = geometric / arithmetic
        mel_power = filter_array @ power
        mel_features = np.log1p(mel_power * 40.0) / 8.0
        signs = np.signbit(frame)
        zcr = float(np.count_nonzero(signs[1:] != signs[:-1])) / max(1, frame_length - 1)
        values = [float(value) for value in mel_features]
        values.extend((zcr, max(0.0, min(1.0, centroid))))
        fixed_values: list[int] = []
        # The feature dimension loop is deliberately explicit and ascending.
        for dimension in range(len(values)):
            fixed_values.append(round_half_away_from_zero(values[dimension] * fixed_denominator))
        selected_frames.append(frame_index)
        per_frame_q.append(tuple(fixed_values))
        frame_rms.append(round(rms, 15))
        frame_flatness.append(round(flatness, 15))
    dimensions = analysis.mel_bands + 2
    feature_sum = [0] * dimensions
    for row in per_frame_q:
        for dimension in range(dimensions):
            feature_sum[dimension] += row[dimension]
    frame_count = len(per_frame_q)
    if frame_count:
        feature_mean_q = tuple(round_half_away_from_zero(value / frame_count) for value in feature_sum)
        speech_values = [
            _speech_ppm(rms * (1.0 - 0.55 * min(1.0, max(0.0, flat))), endpoint_low_energy, endpoint_high_energy)
            for rms, flat in zip(frame_rms, frame_flatness)
        ]
        speech_ppm = round_half_away_from_zero(sum(speech_values) / frame_count)
    else:
        feature_mean_q = tuple(0 for _ in range(dimensions))
        speech_ppm = 0
    payload = {
        "frame_indices": selected_frames,
        "feature_sum": feature_sum,
        "feature_mean_q": list(feature_mean_q),
        "quant_profile": analysis.quant_profile,
        "fixed_denominator": fixed_denominator,
    }
    return FixedWindowFeature(
        frame_indices=tuple(selected_frames),
        frame_count=frame_count,
        feature_sum=tuple(feature_sum),
        feature_mean_q=feature_mean_q,
        per_frame_q=tuple(per_frame_q),
        frame_rms=tuple(frame_rms),
        frame_flatness=tuple(frame_flatness),
        speech_ppm=speech_ppm,
        quant_profile=analysis.quant_profile,
        fixed_denominator=fixed_denominator,
        feature_digest=json_hash(payload),
    )


@dataclass(frozen=True)
class CommandPosterior:
    posterior_ppm: dict[int, int]
    best_code: int | None
    runner_up_code: int | None
    margin_ppm: int
    distance_ppm: dict[int, int]
    projection_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "posterior_ppm": {str(code): value for code, value in sorted(self.posterior_ppm.items())},
            "best_code": self.best_code,
            "runner_up_code": self.runner_up_code,
            "margin_ppm": self.margin_ppm,
            "distance_ppm": {str(code): value for code, value in sorted(self.distance_ppm.items())},
            "projection_fingerprint": self.projection_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommandPosterior":
        return cls(
            posterior_ppm={int(code): int(prob) for code, prob in value["posterior_ppm"].items()},
            best_code=None if value.get("best_code") is None else int(value["best_code"]),
            runner_up_code=None if value.get("runner_up_code") is None else int(value["runner_up_code"]),
            margin_ppm=int(value["margin_ppm"]),
            distance_ppm={int(code): int(distance) for code, distance in value["distance_ppm"].items()},
            projection_fingerprint=str(value["projection_fingerprint"]),
        )


class CommandProjectionModel:
    """Fixed integer-feature projection derived from the command table + config.

    Prototypes come from deterministic in-memory probes, never from the measured
    session.  A fingerprint-derived signed permutation then defines the actual
    projection space.  No runtime RNG or thread-order-dependent reduction is
    involved.
    """

    def __init__(
        self,
        config: DiagnosticConfig,
        commands: Iterable[CommandDef],
    ) -> None:
        self.config = config
        self.commands = tuple(sorted(commands, key=lambda item: item.code))
        self.codes = tuple(command.code for command in self.commands)
        self.dimensions = config.analysis.mel_bands + 2
        self.projection_fingerprint = json_hash(
            {
                "projection": config.projection_seed_material,
                "command_codes": list(self.codes),
                "phrases": [command.phrase for command in self.commands],
            }
        )
        raw_templates: dict[int, tuple[tuple[int, ...], ...]] = {}
        all_vectors: list[tuple[int, ...]] = []
        for command in self.commands:
            vectors = self._build_templates(command)
            raw_templates[command.code] = vectors
            all_vectors.extend(vectors)
        if not all_vectors:
            raise ValueError("command projection requires at least one non-empty template")
        matrix = np.asarray(all_vectors, dtype=np.float64)
        centers = np.mean(matrix, axis=0)
        deviations = np.std(matrix, axis=0)
        median_deviation = float(np.median(deviations[deviations > 0.0])) if np.any(deviations > 0.0) else 1.0
        deviations = np.maximum(deviations, max(1.0, median_deviation * 0.02))
        digest = hashlib.sha256(self.projection_fingerprint.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
        permutation = rng.permutation(self.dimensions)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=self.dimensions)
        scale = signs / deviations[permutation]
        self._centers = tuple(float(value) for value in centers)
        self._permutation = tuple(int(value) for value in permutation)
        self._scale = tuple(float(value) for value in scale)
        projected: dict[int, tuple[tuple[float, ...], ...]] = {}
        for command in self.commands:
            rows: list[tuple[float, ...]] = []
            for vector in raw_templates[command.code]:
                rows.append(self._project(vector))
            projected[command.code] = tuple(rows)
        self._templates = projected
        self._temperature = 0.85

    def _build_templates(self, command: CommandDef) -> tuple[tuple[int, ...], ...]:
        from .synthesize import make_command_probe

        probe = make_command_probe(
            command_code=command.code,
            phrase=command.phrase,
            sample_rate=self.config.sample_rate,
            seed=self.config.seed,
        )
        length = self.config.analysis.window_length
        shift = self.config.analysis.window_shift
        starts = list(range(0, max(1, probe.size - length + 1), shift)) or [0]
        global_frame_starts = list(
            range(
                0,
                max(1, probe.size - self.config.analysis.frame_length + 1),
                self.config.analysis.frame_shift,
            )
        )
        vectors: list[tuple[int, ...]] = []
        for start in starts:
            pcm_window = probe[start : start + length]
            frame_indices: list[int] = []
            for frame_start in global_frame_starts:
                local_frame = frame_start - start
                if local_frame < 0:
                    continue
                if local_frame + self.config.analysis.frame_length > pcm_window.size:
                    continue
                center = local_frame + self.config.analysis.frame_length // 2
                if 0 <= center < pcm_window.size:
                    frame_indices.append(frame_start // self.config.analysis.frame_shift)
            feature = extract_fixed_feature(
                pcm_window,
                window_start=start,
                frame_indices=frame_indices,
                analysis=self.config.analysis,
                endpoint_low_energy=self.config.endpoint.low_energy,
                endpoint_high_energy=self.config.endpoint.high_energy,
                valid_range=(start, start + pcm_window.size),
            )
            if feature.frame_count:
                vectors.append(feature.feature_mean_q)
        return tuple(vectors)

    def _project(self, vector: Iterable[int]) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        result: list[float] = []
        for projected_dimension in range(self.dimensions):
            source_dimension = self._permutation[projected_dimension]
            result.append((values[source_dimension] - self._centers[source_dimension]) * self._scale[projected_dimension])
        return tuple(result)

    def posterior(self, feature: FixedWindowFeature) -> CommandPosterior:
        if feature.frame_count == 0 or not self.codes:
            return CommandPosterior({}, None, None, 0, {}, self.projection_fingerprint)
        projected_feature = self._project(feature.feature_mean_q)
        distances: dict[int, float] = {}
        for code in sorted(self.codes):
            best_distance = math.inf
            for template in self._templates[code]:
                distance = 0.0
                for dimension in range(self.dimensions):
                    delta = projected_feature[dimension] - template[dimension]
                    distance += delta * delta
                if distance < best_distance:
                    best_distance = distance
            distances[code] = best_distance
        ordered = sorted(distances.items(), key=lambda item: (item[1], item[0]))
        minimum = ordered[0][1]
        weights: dict[int, float] = {}
        # Command-code ascending is the fixed reduction order, independent of
        # how close the floating distances happen to be.
        for code in sorted(distances):
            weights[code] = math.exp(-(distances[code] - minimum) / self._temperature)
        total = sum(weights[code] for code in sorted(weights))
        posterior: dict[int, int] = {}
        for code in sorted(weights):
            posterior[code] = round_half_away_from_zero(weights[code] / total * 1_000_000.0)
        remainder = 1_000_000 - sum(posterior.values())
        posterior[ordered[0][0]] += remainder
        posterior = {code: posterior[code] for code in sorted(posterior)}
        ordered_posterior = sorted(posterior.items(), key=lambda item: (-item[1], item[0]))
        best_code = ordered_posterior[0][0]
        runner_up_code = ordered_posterior[1][0] if len(ordered_posterior) > 1 else None
        runner_up_probability = ordered_posterior[1][1] if len(ordered_posterior) > 1 else 0
        distance_ppm = {
            code: min(1_000_000, round_half_away_from_zero(value * 1000.0))
            for code, value in sorted(distances.items())
        }
        return CommandPosterior(
            posterior_ppm=posterior,
            best_code=best_code,
            runner_up_code=runner_up_code,
            margin_ppm=posterior[best_code] - runner_up_probability,
            distance_ppm=distance_ppm,
            projection_fingerprint=self.projection_fingerprint,
        )
