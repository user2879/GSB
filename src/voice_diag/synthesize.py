from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .audio import write_pcm16_wav
from .receipts import ReceiptDocument, ReceiptRecord, to_a1, to_a2
from .util import atomic_write_json, atomic_write_text


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "global_version": 7,
    "seed": 20261127,
    "sample_rate": 16000,
    "channels": 1,
    "bits_per_sample": 16,
    "endpoint": {
        "frame_length": 400,
        "frame_shift": 160,
        "high_energy": 0.020,
        "low_energy": 0.006,
        "min_speech_ms": 80,
        "min_silence_ms": 100,
    },
    "analysis": {
        "window_length": 2400,
        "window_shift": 1200,
        "frame_length": 400,
        "frame_shift": 160,
        "mel_bands": 12,
        "quant_scale": 1000,
        "frac_bits": 6,
        "allowed_quant_scales": [100, 1000, 2000],
        "allowed_frac_bits": [4, 6, 8],
        "posterior_quantum_ppm": 1000,
    },
    "adjudication": {
        "frame_endpoint_threshold_ppm": 480000,
        "min_command_ms": 120,
        "tie_half_step_ppm": 500,
        "min_overlap_frames": 1,
    },
    "scheduler": {
        "max_workers": 4,
        "point_budget": 42,
        "per_interval_quota": 1000,
        "point_cost": 1,
    },
    "receipt": {
        "field_widths": {
            "interval_id": 12,
            "speaker_id": 8,
            "confidence": 6,
            "command_code": 5,
        },
        "quant_schemes": ["permille", "float32", "q6_scale1000"],
        "default_quant_scheme": "permille",
    },
}


COMMANDS: tuple[dict[str, Any], ...] = (
    {"code": 1, "name": "开灯", "phrase": "打开灯光", "allowed_chars": ["打", "开", "灯", "光"]},
    {"code": 2, "name": "升温", "phrase": "调高温度", "allowed_chars": ["调", "高", "温", "度"]},
    {"code": 3, "name": "关灯", "phrase": "关闭灯光", "allowed_chars": ["关", "闭", "灯", "光"]},
    {"code": 4, "name": "播放", "phrase": "播放音乐", "allowed_chars": ["播", "放", "音", "乐"]},
    {"code": 5, "name": "停止", "phrase": "停止播放", "allowed_chars": ["停", "止", "播", "放"]},
    {"code": 6, "name": "降温", "phrase": "降低温度", "allowed_chars": ["降", "低", "温", "度"]},
)


@dataclass(frozen=True)
class DemoIntervalSpec:
    interval_id: str
    speaker_id: str
    start_sample: int
    duration_samples: int
    command_code: int
    truth_confidence: float


DEMO_INTERVALS: tuple[DemoIntervalSpec, ...] = (
    DemoIntervalSpec("I001", "S01", 0, 16000, 1, 0.9234),
    DemoIntervalSpec("I002", "S02", 11000, 16000, 2, 0.8762),
    DemoIntervalSpec("I003", "S03", 23000, 16000, 3, 0.9417),
    DemoIntervalSpec("I004", "S01", 34500, 15000, 4, 0.8976),
)


def _command(code: int) -> dict[str, Any]:
    for command in COMMANDS:
        if command["code"] == code:
            return command
    raise KeyError(code)


def _char_spans(phrase: str, duration: int) -> list[dict[str, Any]]:
    count = len(phrase)
    margin = min(480, duration // 10)
    usable = duration - 2 * margin
    step = usable // count
    spans: list[dict[str, Any]] = []
    for index, char in enumerate(phrase):
        start = margin + index * step
        end = margin + (index + 1) * step
        gap = min(140, max(0, step // 12))
        spans.append(
            {
                "index": index,
                "char": char,
                "start_sample": start,
                "end_sample": end - gap,
            }
        )
    return spans


def _synthesize_interval_audio(
    *,
    command_code: int,
    phrase: str,
    duration: int,
    sample_rate: int,
    seed: int,
    fragment_index: int,
) -> np.ndarray:
    """Create an amplitude-controlled command utterance.

    This is diagnostic audio rather than a vocoder: each command gets a stable
    harmonic carrier family, while characters introduce controlled frequency
    steps.  The fixed seed only drives low-level dither.
    """

    rng = np.random.default_rng(seed * 1009 + fragment_index * 9176 + command_code)
    total = np.arange(duration, dtype=np.float64) / sample_rate
    result = np.zeros(duration, dtype=np.float64)
    base_frequency = 190.0 + command_code * 73.0
    spans = _char_spans(phrase, duration)
    for char_index, span in enumerate(spans):
        start = int(span["start_sample"])
        end = int(span["end_sample"])
        if end <= start:
            continue
        local_t = np.arange(end - start, dtype=np.float64) / sample_rate
        char_shift = 1.0 + ((ord(str(span["char"])) + command_code * 17 + char_index * 5) % 9) / 100.0
        envelope = np.ones(end - start, dtype=np.float64)
        ramp = min(160, max(16, (end - start) // 8))
        envelope[:ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
        envelope[-ramp:] = np.linspace(1.0, 0.0, ramp, endpoint=False)
        carrier = (
            0.68 * np.sin(2.0 * np.pi * base_frequency * char_shift * local_t + command_code * 0.11)
            + 0.23 * np.sin(2.0 * np.pi * base_frequency * 2.02 * char_shift * local_t + char_index * 0.37)
            + 0.09 * np.sin(2.0 * np.pi * base_frequency * 3.07 * local_t)
        )
        modulation = 0.82 + 0.18 * np.sin(
            2.0 * np.pi * (5.0 + command_code + char_index) * local_t
        )
        result[start:end] += 0.21 * envelope * modulation * carrier
    result += rng.normal(0.0, 0.0012, size=duration)
    # A deterministic low-level room-like dither is intentionally below the
    # endpoint low threshold so the hysteresis decision remains reviewable.
    result += 0.0010 * np.sin(2.0 * np.pi * 63.0 * total)
    return np.clip(np.rint(result * 32767.0), -32768, 32767).astype("<i2")


def make_command_probe(
    *,
    command_code: int,
    phrase: str,
    sample_rate: int,
    seed: int,
    duration: int = 10_000,
) -> np.ndarray:
    """Public deterministic probe used by the fixed command projection.

    The same carrier family as the fixture audio is used, but the probe is not
    read from the fixture.  Its only role is to turn the declared command table
    into a stable, inspectable projection prototype.
    """

    return _synthesize_interval_audio(
        command_code=command_code,
        phrase=phrase,
        duration=duration,
        sample_rate=sample_rate,
        seed=seed,
        fragment_index=10_000 + command_code,
    )


def _manifest(seed: int) -> dict[str, Any]:
    speakers = [
        {"id": "S01", "display_name": "说话人甲", "trust_ppm": 900000},
        {"id": "S02", "display_name": "说话人乙", "trust_ppm": 870000},
        {"id": "S03", "display_name": "说话人丙", "trust_ppm": 840000},
    ]
    fragments: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    command_stream: list[dict[str, Any]] = []
    for arrival_index, spec in enumerate(DEMO_INTERVALS):
        command = _command(spec.command_code)
        chars = _char_spans(command["phrase"], spec.duration_samples)
        fragment_id = f"F{arrival_index + 1:03d}"
        interval_id = spec.interval_id
        fragments.append(
            {
                "id": fragment_id,
                "interval_id": interval_id,
                "audio_file": f"audio/{fragment_id}.wav",
                "arrival_index": arrival_index,
                "capture_start_ms": 1000 + int(spec.start_sample * 1000 / 16000),
                "arrival_time_ms": 1200 + int(spec.start_sample * 1000 / 16000),
            }
        )
        intervals.append(
            {
                "id": interval_id,
                "speaker_id": spec.speaker_id,
                "start_sample": spec.start_sample,
                "end_sample": spec.start_sample + spec.duration_samples,
                "truth_command_code": spec.command_code,
                "text": command["phrase"],
                "chars": chars,
            }
        )
        command_stream.append(
            {
                "turn_index": arrival_index,
                "interval_id": interval_id,
                "command_code": spec.command_code,
                "char_indices": list(range(len(chars))),
            }
        )
    return {
        "conversation_id": f"demo-{seed}",
        "seed": seed,
        "speakers": speakers,
        "commands": list(COMMANDS),
        "fragments": fragments,
        "intervals": intervals,
        "command_stream": command_stream,
        "receipts": {
            "a1": "receipts/evidence.a1",
            "a2": "receipts/evidence.a2.json",
        },
    }


def _truth_receipts() -> ReceiptDocument:
    records: list[ReceiptRecord] = []
    for index, spec in enumerate(DEMO_INTERVALS):
        command = _command(spec.command_code)
        permille = int(round(spec.truth_confidence * 1000.0))
        records.append(
            ReceiptRecord(
                interval_id=spec.interval_id,
                speaker_id=spec.speaker_id,
                endpoint_confidence=permille / 1000.0,
                command_name=command["name"],
                command_code=spec.command_code,
                quant_scheme="permille",
                source_version="fixture",
                missing_fields=(),
                storage={},
            )
        )
    return ReceiptDocument(tuple(records), "fixture")


def generate_demo(root: str | Path, *, seed: int = 20261127, overwrite: bool = True) -> dict[str, Path]:
    target = Path(root)
    config_path = target / "config.json"
    manifest_path = target / "manifest.json"
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(f"demo directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "audio").mkdir(parents=True, exist_ok=True)
    (target / "receipts").mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["seed"] = seed
    manifest = _manifest(seed)
    atomic_write_json(config_path, config)
    atomic_write_json(manifest_path, manifest)
    for index, spec in enumerate(DEMO_INTERVALS):
        command = _command(spec.command_code)
        pcm = _synthesize_interval_audio(
            command_code=spec.command_code,
            phrase=command["phrase"],
            duration=spec.duration_samples,
            sample_rate=config["sample_rate"],
            seed=seed,
            fragment_index=index,
        )
        write_pcm16_wav(target / "audio" / f"F{index + 1:03d}.wav", pcm, sample_rate=config["sample_rate"])
    truth = _truth_receipts()
    vocab = {command["code"]: command["name"] for command in COMMANDS}
    conversion_a1 = to_a1(truth, vocabulary=vocab)
    conversion_a2 = to_a2(truth, vocabulary=vocab)
    conversion_a1.write(target / "receipts" / "evidence.a1")
    conversion_a2.write(target / "receipts" / "evidence.a2.json")
    atomic_write_json(target / "fixture-receipt-sidecar.json", conversion_a2.sidecar)
    atomic_write_text(
        target / "README.txt",
        "Deterministic local sample generated by voice_diag.synthesize.generate_demo.\n"
        "No network, real recording, or external acoustic corpus is used.\n",
    )
    return {
        "root": target,
        "config": config_path,
        "manifest": manifest_path,
    }
