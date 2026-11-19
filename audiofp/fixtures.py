"""Deterministic, fully synthetic audio fixtures for offline verification."""

from __future__ import annotations

import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import butter, sosfilt

from .audio import decode_audio, encode_lossy_ogg, resample_to, sha256_file, write_wav
from .config import DEFAULT_CONFIG


SOURCE_RATE = 48000


@dataclass(frozen=True)
class NoteEvent:
    beat: float
    midi: int
    beats: float
    timbre: str
    velocity: float = 1.0


@dataclass(frozen=True)
class TrackSpec:
    truth_id: str
    title: str
    duration: float
    tempo_bpm: float
    motif: tuple[NoteEvent, ...]
    motif_occurrences: tuple[float, ...]
    counter_melody: tuple[NoteEvent, ...]
    counter_start: float
    repeat_energy: float


def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _rng(seed: int, label: str) -> np.random.Generator:
    material = f"{seed}:{label}".encode("utf-8")
    words = np.frombuffer(material.ljust(32, b"\0")[:32], dtype=np.uint32)
    state = int(words[0]) ^ int(words[-1] << 1) ^ 0xA5A5A5A5
    return np.random.default_rng(state & 0xFFFFFFFF)


def _attack_release(length: int, attack: float, release: float) -> np.ndarray:
    envelope = np.ones(length, dtype=np.float64)
    attack_samples = min(length, max(1, int(attack * SOURCE_RATE)))
    release_samples = min(length, max(1, int(release * SOURCE_RATE)))
    envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
    envelope[-release_samples:] *= np.linspace(1.0, 0.0, release_samples)
    return envelope


def _render_note(
    midi: int,
    duration: float,
    timbre: str,
    velocity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    length = max(1, int(round(duration * SOURCE_RATE)))
    time = np.arange(length, dtype=np.float64) / SOURCE_RATE
    frequency = midi_to_hz(midi)
    phase = rng.uniform(0.0, 2.0 * math.pi, size=8)
    if timbre == "piano":
        values = np.zeros(length)
        for harmonic in range(1, 8):
            values += (0.72 ** harmonic) * np.sin(
                2.0 * math.pi * frequency * harmonic * time + phase[harmonic - 1]
            )
        values *= np.exp(-2.0 * time)
        values += 0.02 * rng.standard_normal(length) * np.exp(-18.0 * time)
        envelope = _attack_release(length, 0.004, min(0.18, duration / 3))
    elif timbre == "flute":
        values = np.sin(2.0 * math.pi * frequency * time + phase[0])
        values += 0.22 * np.sin(2.0 * math.pi * frequency * 2.0 * time + phase[1])
        breath = rng.standard_normal(length)
        sos = butter(2, min(0.95, 4200.0 / (SOURCE_RATE / 2)), btype="low", output="sos")
        values += 0.035 * sosfilt(sos, breath)
        values *= 0.8 + 0.2 * np.sin(2.0 * math.pi * 5.2 * time)
        envelope = _attack_release(length, 0.06, min(0.28, duration / 2))
    elif timbre == "bass":
        values = np.sin(2.0 * math.pi * frequency * time + phase[0])
        values += 0.32 * np.sin(2.0 * math.pi * frequency * 2.0 * time + phase[1])
        values += 0.15 * np.sin(2.0 * math.pi * frequency * 3.0 * time + phase[2])
        values *= np.exp(-0.7 * time)
        envelope = _attack_release(length, 0.012, min(0.35, duration / 2))
    elif timbre == "strings":
        values = np.zeros(length)
        for harmonic in range(1, 7):
            values += np.sin(
                2.0 * math.pi * frequency * harmonic * time + phase[harmonic - 1]
            ) / harmonic
        values *= 0.82 + 0.18 * np.sin(2.0 * math.pi * 3.4 * time)
        envelope = _attack_release(length, 0.12, min(0.5, duration / 2))
    else:
        raise ValueError(f"unknown timbre: {timbre}")
    return velocity * values * envelope


def _add_at(buffer: np.ndarray, sample_index: int, values: np.ndarray) -> None:
    if sample_index >= len(buffer) or len(values) == 0:
        return
    start = max(0, sample_index)
    source_offset = max(0, -sample_index)
    count = min(len(buffer) - start, len(values) - source_offset)
    if count > 0:
        buffer[start : start + count] += values[source_offset : source_offset + count]


def _render_melody(
    buffer: np.ndarray,
    events: Iterable[NoteEvent],
    *,
    start_seconds: float,
    beat_seconds: float,
    transpose: int,
    seed: int,
    label: str,
) -> None:
    for index, event in enumerate(events):
        duration = event.beats * beat_seconds
        note_rng = _rng(seed, f"{label}:{index}:{event.midi}:{event.beat}")
        values = _render_note(
            event.midi + transpose,
            duration,
            event.timbre,
            event.velocity,
            note_rng,
        )
        _add_at(buffer, int(round((start_seconds + event.beat * beat_seconds) * SOURCE_RATE)), values)


def _add_percussion(buffer: np.ndarray, seed: int, tempo_bpm: float, duration: float) -> None:
    beat = 60.0 / tempo_bpm
    rng = _rng(seed, "percussion")
    count = int(duration / beat) + 1
    for index in range(count):
        start = int(round(index * beat * SOURCE_RATE))
        if start >= len(buffer):
            break
        if index % 4 in (0, 2):
            length = int(0.22 * SOURCE_RATE)
            t = np.arange(length) / SOURCE_RATE
            sweep = 46.0 + 72.0 * np.exp(-22.0 * t)
            kick = 0.34 * np.sin(2.0 * math.pi * np.cumsum(sweep) / SOURCE_RATE) * np.exp(-12.0 * t)
            _add_at(buffer, start, kick)
        if index % 4 == 1 or index % 8 == 6:
            length = int(0.16 * SOURCE_RATE)
            noise = rng.standard_normal(length)
            sos = butter(2, [900.0 / (SOURCE_RATE / 2), 6500.0 / (SOURCE_RATE / 2)], btype="band", output="sos")
            snare = 0.11 * sosfilt(sos, noise) * np.exp(-18.0 * np.arange(length) / SOURCE_RATE)
            _add_at(buffer, start, snare)
        if index % 2 == 1:
            length = int(0.045 * SOURCE_RATE)
            click = np.diff(rng.standard_normal(length + 1))
            click *= 0.018 * np.exp(-55.0 * np.arange(length) / SOURCE_RATE)
            _add_at(buffer, start + int(0.5 * beat * SOURCE_RATE), click)


def _ambient_noise(length: int, seed: int) -> np.ndarray:
    rng = _rng(seed, "ambient")
    raw = rng.standard_normal(length)
    sos = butter(2, 820.0 / (SOURCE_RATE / 2), btype="low", output="sos")
    low = sosfilt(sos, raw)
    slow = 0.6 + 0.4 * np.sin(2.0 * math.pi * 0.13 * np.arange(length) / SOURCE_RATE)
    return 0.008 * low * slow


def compose_track(
    spec: TrackSpec,
    seed: int,
    *,
    transpose: int = 0,
    tempo_scale: float = 1.0,
) -> np.ndarray:
    length = int(round(spec.duration * SOURCE_RATE))
    buffer = _ambient_noise(length, seed + sum(ord(char) for char in spec.truth_id))
    beat_seconds = (60.0 / spec.tempo_bpm) / tempo_scale
    effective_duration = spec.duration / tempo_scale
    _add_percussion(buffer, seed + 17, spec.tempo_bpm * tempo_scale, effective_duration)
    for repeat_index, occurrence in enumerate(spec.motif_occurrences):
        _render_melody(
            buffer,
            spec.motif,
            start_seconds=occurrence / tempo_scale,
            beat_seconds=beat_seconds,
            transpose=transpose,
            seed=seed + repeat_index,
            label=f"{spec.truth_id}:motif:{repeat_index}",
        )
    _render_melody(
        buffer,
        spec.counter_melody,
        start_seconds=spec.counter_start / tempo_scale,
        beat_seconds=beat_seconds,
        transpose=transpose,
        seed=seed + 91,
        label=f"{spec.truth_id}:counter",
    )

    # Deliberate silent/quiet passages make peak picking and deletion tests less trivial.
    for start, end in ((0.0, 0.55), (5.55, 6.5), (11.25, spec.duration)):
        begin = max(0, int(start / tempo_scale * SOURCE_RATE))
        finish = min(length, int(end / tempo_scale * SOURCE_RATE))
        if finish > begin:
            buffer[begin:finish] *= 0.04
    fade = min(length // 2, int(0.18 * SOURCE_RATE))
    if fade > 0:
        buffer[:fade] *= np.linspace(0.0, 1.0, fade)
        buffer[-fade:] *= np.linspace(1.0, 0.0, fade)
    peak = float(np.max(np.abs(buffer)))
    if peak > 0:
        buffer *= 0.68 / peak
    return buffer


AURORA_MOTIF = (
    NoteEvent(0.0, 64, 0.75, "piano", 0.95),
    NoteEvent(0.5, 69, 0.75, "piano", 0.88),
    NoteEvent(1.0, 71, 0.75, "piano", 0.92),
    NoteEvent(1.5, 72, 0.75, "piano", 0.85),
    NoteEvent(2.0, 69, 0.9, "strings", 0.62),
)

TRACK_SPECS: tuple[TrackSpec, ...] = (
    TrackSpec(
        truth_id="aurora",
        title="Aurora Circuit",
        duration=12.0,
        tempo_bpm=104.0,
        motif=AURORA_MOTIF,
        motif_occurrences=(1.45, 7.35),
        counter_melody=(
            NoteEvent(0.0, 52, 1.0, "bass", 0.70),
            NoteEvent(1.0, 59, 0.75, "bass", 0.65),
            NoteEvent(2.0, 67, 0.75, "flute", 0.62),
            NoteEvent(2.5, 64, 0.75, "flute", 0.58),
            NoteEvent(3.0, 62, 1.25, "strings", 0.55),
        ),
        counter_start=3.55,
        repeat_energy=0.9,
    ),
    TrackSpec(
        truth_id="ember",
        title="Ember Lanterns",
        duration=11.5,
        tempo_bpm=92.0,
        motif=(
            NoteEvent(0.0, 57, 0.5, "flute", 0.82),
            NoteEvent(0.5, 60, 0.5, "flute", 0.78),
            NoteEvent(1.0, 64, 0.75, "piano", 0.85),
            NoteEvent(1.75, 62, 0.75, "strings", 0.62),
        ),
        motif_occurrences=(2.1, 6.9),
        counter_melody=(
            NoteEvent(0.0, 45, 1.5, "bass", 0.72),
            NoteEvent(1.5, 52, 1.0, "piano", 0.68),
            NoteEvent(2.5, 55, 1.0, "flute", 0.66),
        ),
        counter_start=4.2,
        repeat_energy=0.8,
    ),
    TrackSpec(
        truth_id="cinder",
        title="Cinder Atlas",
        duration=10.5,
        tempo_bpm=116.0,
        motif=(
            NoteEvent(0.0, 62, 0.5, "strings", 0.76),
            NoteEvent(0.5, 67, 0.5, "strings", 0.72),
            NoteEvent(1.0, 70, 0.5, "piano", 0.82),
            NoteEvent(1.5, 74, 0.75, "flute", 0.66),
        ),
        motif_occurrences=(1.2, 7.05),
        counter_melody=(
            NoteEvent(0.0, 50, 1.0, "bass", 0.70),
            NoteEvent(1.0, 57, 0.5, "bass", 0.64),
            NoteEvent(2.0, 65, 0.75, "piano", 0.62),
            NoteEvent(3.0, 64, 0.75, "strings", 0.60),
        ),
        counter_start=4.3,
        repeat_energy=0.85,
    ),
)


def _stereo_version(mono: np.ndarray, seed: int, label: str) -> np.ndarray:
    # The channels share content but are not sample-identical.  Peak-band
    # fingerprints must survive the mandated mono downmix.
    rng = _rng(seed, f"stereo:{label}")
    delay = int(0.0015 * SOURCE_RATE)
    left = mono.copy()
    right = 0.965 * mono.copy()
    if delay < len(mono):
        left[delay:] += 0.018 * mono[:-delay]
        right[:-delay] += 0.012 * mono[delay:]
    noise = rng.standard_normal(len(mono))
    sos = butter(2, 2600.0 / (SOURCE_RATE / 2), btype="low", output="sos")
    noise = sosfilt(sos, noise)
    left += 0.0015 * noise
    right -= 0.0012 * noise
    return np.stack([left, right], axis=1)


def _format_entries(
    root: Path,
    category: str,
    stem: str,
    mono48: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    directory = root / category
    directory.mkdir(parents=True, exist_ok=True)
    stereo = _stereo_version(mono48, seed, stem)
    wav24 = directory / f"{stem}_48k_24b_stereo.wav"
    wav16 = directory / f"{stem}_44k_16b_mono.wav"
    ogg = directory / f"{stem}_48k_vorbis.ogg"
    write_wav(wav24, stereo, SOURCE_RATE, subtype="PCM_24", channels=2)
    mono44 = resample_to(mono48, SOURCE_RATE, 44100)
    write_wav(wav16, mono44, 44100, subtype="PCM_16", channels=1)
    encode_lossy_ogg(wav24, ogg, quality=5)
    entries: dict[str, dict[str, Any]] = {}
    expected = {
        "wav16_mono_44k": {
            "container": "WAV",
            "codec": "PCM_16",
            "sample_rate": 44100,
            "bit_depth": 16,
            "channels": 1,
        },
        "wav24_stereo_48k": {
            "container": "WAV",
            "codec": "PCM_24",
            "sample_rate": 48000,
            "bit_depth": 24,
            "channels": 2,
        },
        "vorbis_stereo_48k": {
            "container": "OGG",
            "codec": "VORBIS",
            "sample_rate": 48000,
            "bit_depth": None,
            "channels": 2,
        },
    }
    for label, path in (
        ("wav16_mono_44k", wav16),
        ("wav24_stereo_48k", wav24),
        ("vorbis_stereo_48k", ogg),
    ):
        document = decode_audio(path, DEFAULT_CONFIG.sample_rate)
        entries[label] = {
            "label": label,
            "relative_path": path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(path),
            "file_size": path.stat().st_size,
            "declared": expected[label],
            "decoded": document.source.to_dict(),
        }
    return entries


def _write_broken_files(root: Path, seed: int) -> list[dict[str, Any]]:
    directory = root / "broken"
    directory.mkdir(parents=True, exist_ok=True)
    source = root / "audio" / "aurora_44k_16b_mono.wav"
    raw = source.read_bytes()

    truncated = directory / "truncated_declared_frames.wav"
    truncated.write_bytes(raw[:-900])

    misaligned = bytearray(raw[:16000])
    data_index = bytes(misaligned).find(b"data")
    if data_index >= 0:
        struct.pack_into("<I", misaligned, data_index + 4, 101)
    aligned = directory / "misaligned_data_chunk.wav"
    aligned.write_bytes(bytes(misaligned))

    random_ogg = directory / "not_really_vorbis.ogg"
    rng = _rng(seed, "broken-ogg")
    random_ogg.write_bytes(bytes(rng.integers(0, 256, size=4096, dtype=np.uint8).tolist()))
    return [
        {
            "relative_path": truncated.relative_to(root).as_posix(),
            "expected_error": "E_AUDIO_FORMAT",
            "reason": "declared_data_exceeds_file",
        },
        {
            "relative_path": aligned.relative_to(root).as_posix(),
            "expected_error": "E_AUDIO_FORMAT",
            "reason": "data_chunk_not_frame_aligned",
        },
        {
            "relative_path": random_ogg.relative_to(root).as_posix(),
            "expected_error": "E_AUDIO_FORMAT",
            "reason": "decoder_rejects_random_container",
        },
    ]


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    payload = json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    (root / "manifest.json").write_text(payload, encoding="utf-8", newline="\n")
    (root / "manifest.sha256").write_text(
        sha256_file(root / "manifest.json") + "  manifest.json\n",
        encoding="ascii",
        newline="\n",
    )


def generate_fixtures(
    output_dir: str | Path,
    *,
    seed: int = 20261119,
    clean: bool = True,
) -> dict[str, Any]:
    root = Path(output_dir)
    if clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    tracks: list[dict[str, Any]] = []
    truth_by_relative_path: dict[str, str] = {}
    for track_index, spec in enumerate(TRACK_SPECS):
        mono = compose_track(spec, seed + track_index * 1000)
        formats = _format_entries(root, "audio", spec.truth_id, mono, seed=seed + 300 + track_index)
        for entry in formats.values():
            truth_by_relative_path[entry["relative_path"]] = spec.truth_id
        tracks.append(
            {
                "truth_id": spec.truth_id,
                "title": spec.title,
                "duration": spec.duration,
                "tempo_bpm": spec.tempo_bpm,
                "repeat_occurrences_seconds": list(spec.motif_occurrences),
                "formats": [formats[key] for key in sorted(formats)],
            }
        )

    # Deliberately non-identical transformations are kept as labelled variant
    # material. They are not used for same-content dedup.
    variants: list[dict[str, Any]] = []
    aurora = next(spec for spec in TRACK_SPECS if spec.truth_id == "aurora")
    for label, transpose, tempo_scale in (
        ("transpose_up_3_semitones", 3, 1.0),
        ("tempo_fast_1_08x", 0, 1.08),
    ):
        mono = compose_track(aurora, seed + 9000, transpose=transpose, tempo_scale=tempo_scale)
        path = root / "variants" / f"aurora_{label}.wav"
        write_wav(path, mono, SOURCE_RATE, subtype="PCM_24", channels=1)
        variants.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "variant_of": "aurora",
                "kind": "transpose" if transpose else "tempo",
                "transpose_semitones": transpose,
                "tempo_scale": tempo_scale,
                "file_sha256": sha256_file(path),
            }
        )

    # A query cut from a motif that occurs twice in Aurora. Any matching
    # occurrence is accepted by the labelled truth range.
    query_start = 7.35
    query_duration = 2.45
    aurora_mono = compose_track(aurora, seed)
    begin = int(round(query_start * SOURCE_RATE))
    end = int(round((query_start + query_duration) * SOURCE_RATE))
    query_mono = aurora_mono[begin:end].copy()
    formats = _format_entries(root, "queries", "aurora_repeat", query_mono, seed=seed + 4999)
    for entry in formats.values():
        truth_by_relative_path[entry["relative_path"]] = "query:aurora_repeat"
    query = {
        "query_id": "aurora_repeat_query",
        "truth_group": "aurora",
        "expected_alignment_starts_seconds": [1.45, 7.35],
        "query_duration_seconds": query_duration,
        "formats": [formats[key] for key in sorted(formats)],
    }

    broken = _write_broken_files(root, seed)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generator": "audiofp.fixtures",
        "seed": seed,
        "source_rate": SOURCE_RATE,
        "analysis_config": DEFAULT_CONFIG.to_dict(),
        "tracks": sorted(tracks, key=lambda item: item["truth_id"]),
        "variants": sorted(variants, key=lambda item: item["relative_path"]),
        "query": query,
        "broken_samples": sorted(broken, key=lambda item: item["relative_path"]),
        "truth_by_relative_path": dict(sorted(truth_by_relative_path.items())),
    }
    _write_manifest(root, manifest)
    return manifest


def load_manifest(root: str | Path) -> dict[str, Any]:
    return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
