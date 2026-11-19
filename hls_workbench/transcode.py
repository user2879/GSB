"""Per-segment H.264/AAC transcoding with explicit absolute timestamps."""

from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

from .canonical import sha256_file
from .config import EncoderConfig, MediaConfig, ProfileConfig
from .errors import WorkbenchError, require


@dataclass(frozen=True)
class TranscodeArtifact:
    profile: str
    sequence: int
    relative_path: str
    sha256: str
    file_size: int
    decoded_frame_count: int
    encoded_frame_count: int
    audio_frame_count: int
    start_pts: int
    end_pts: int
    start_sample: int
    audio_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _x264_options(profile: ProfileConfig, encoder: EncoderConfig) -> dict[str, str]:
    return {
        "preset": encoder.h264_preset,
        "b": str(profile.video_bitrate_kbps * 1000),
        "maxrate": str(profile.video_bitrate_kbps * 1000),
        "bufsize": str(profile.video_bitrate_kbps * 2000),
        "g": "60",
        "keyint_min": "60",
        "scenecut": "0",
        "bf": "0",
        "threads": str(encoder.threads),
        "flags": "+bitexact",
        "x264-params": f"deterministic=1:threads={encoder.threads}",
    }


def _read_wav(path: Path, media: MediaConfig, expected_samples: int) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            require(handle.getnchannels() == media.channels, "E_MEDIA_FORMAT", "WAV channel count mismatch")
            require(handle.getsampwidth() == 2, "E_MEDIA_FORMAT", "WAV must be signed 16-bit PCM")
            require(handle.getframerate() == media.sample_rate, "E_MEDIA_FORMAT", "WAV sample rate mismatch")
            payload = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise WorkbenchError("E_MEDIA_FORMAT", f"cannot read PCM slice: {path.name}", {"error": str(exc)}) from exc
    samples = np.frombuffer(payload, dtype="<i2").reshape(-1, media.channels).copy()
    require(
        len(samples) == expected_samples,
        "E_MEDIA_FORMAT",
        "PCM slice length does not match its manifest",
        expected=expected_samples,
        actual=len(samples),
    )
    return samples


def transcode_segment(
    *,
    input_video: Path,
    input_audio: Path,
    output_path: Path,
    profile: ProfileConfig,
    sequence: int,
    media: MediaConfig,
    encoder: EncoderConfig,
) -> TranscodeArtifact:
    """Transcode one source slice and write timestamp-normalized MPEG-TS."""

    expected_frames = media.frame_count(sequence)
    expected_samples = media.audio_samples(sequence)
    absolute_start_frame = sequence * media.frames_per_slice
    absolute_start_sample = int(round(sequence * media.slice_seconds * media.sample_rate))
    samples = _read_wav(input_audio, media, expected_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(
        str(output_path),
        mode="w",
        format="mpegts",
        options={"fflags": "+bitexact"},
    )
    decoded_frames = 0
    encoded_frames = 0
    audio_packets = 0
    try:
        video_stream = container.add_stream("libx264", rate=media.fps)
        video_stream.width = profile.width
        video_stream.height = profile.height
        video_stream.pix_fmt = "yuv420p"
        video_stream.time_base = Fraction(1, media.fps)
        video_stream.bit_rate = profile.video_bitrate_kbps * 1000
        video_stream.codec_context.max_b_frames = 0
        video_stream.options = _x264_options(profile, encoder)

        audio_stream = container.add_stream("aac", rate=media.sample_rate)
        audio_stream.layout = "stereo"
        audio_stream.bit_rate = profile.audio_bitrate_kbps * 1000
        audio_stream.time_base = Fraction(1, media.sample_rate)

        input_container = av.open(str(input_video), mode="r")
        try:
            input_stream = input_container.streams.video[0]
            require(
                input_stream.codec_context.name == "h264",
                "E_MEDIA_FORMAT",
                "source slice is not H.264",
            )
            require(
                input_stream.codec_context.width == media.width
                and input_stream.codec_context.height == media.height,
                "E_MEDIA_FORMAT",
                "source slice geometry mismatch",
            )
            for frame in input_container.decode(input_stream):
                decoded_frames += 1
                if frame.width != profile.width or frame.height != profile.height:
                    frame = frame.reformat(width=profile.width, height=profile.height, format="yuv420p")
                else:
                    frame = frame.reformat(format="yuv420p")
                frame.pts = decoded_frames - 1
                frame.time_base = Fraction(1, media.fps)
                for packet in video_stream.encode(frame):
                    packet.pts = absolute_start_frame + encoded_frames
                    packet.dts = packet.pts
                    packet.duration = 1
                    packet.time_base = Fraction(1, media.fps)
                    container.mux(packet)
                    encoded_frames += 1
            for packet in video_stream.encode(None):
                packet.pts = absolute_start_frame + encoded_frames
                packet.dts = packet.pts
                packet.duration = 1
                packet.time_base = Fraction(1, media.fps)
                container.mux(packet)
                encoded_frames += 1
        finally:
            input_container.close()

        require(
            decoded_frames == expected_frames,
            "E_MEDIA_FORMAT",
            "decoded source frame count mismatch",
            expected=expected_frames,
            actual=decoded_frames,
        )
        require(
            encoded_frames == expected_frames,
            "E_MEDIA_FORMAT",
            "encoded frame count mismatch",
            expected=expected_frames,
            actual=encoded_frames,
        )

        frame_size = encoder.audio_frame_samples
        for offset in range(0, len(samples), frame_size):
            chunk = samples[offset : offset + frame_size].astype(np.float32) / 32768.0
            audio_frame = av.AudioFrame(format="fltp", layout="stereo", samples=len(chunk))
            audio_frame.sample_rate = media.sample_rate
            audio_frame.time_base = Fraction(1, media.sample_rate)
            audio_frame.pts = absolute_start_sample + offset
            audio_frame.planes[0].update(np.ascontiguousarray(chunk[:, 0]))
            audio_frame.planes[1].update(np.ascontiguousarray(chunk[:, 1]))
            for packet in audio_stream.encode(audio_frame):
                if audio_packets * frame_size >= expected_samples:
                    continue
                packet.pts = absolute_start_sample + audio_packets * frame_size
                packet.dts = packet.pts
                packet.duration = max(0, min(frame_size, expected_samples - audio_packets * frame_size))
                packet.time_base = Fraction(1, media.sample_rate)
                container.mux(packet)
                audio_packets += 1
        for packet in audio_stream.encode(None):
            if audio_packets * frame_size >= expected_samples:
                continue
            packet.pts = absolute_start_sample + audio_packets * frame_size
            packet.dts = packet.pts
            packet.duration = max(0, min(frame_size, expected_samples - audio_packets * frame_size))
            packet.time_base = Fraction(1, media.sample_rate)
            container.mux(packet)
            audio_packets += 1
    finally:
        container.close()

    require(output_path.exists() and output_path.stat().st_size > 0, "E_MEDIA_FORMAT", "empty transcode output")
    return TranscodeArtifact(
        profile=profile.name,
        sequence=sequence,
        relative_path="",
        sha256=sha256_file(output_path),
        file_size=output_path.stat().st_size,
        decoded_frame_count=decoded_frames,
        encoded_frame_count=encoded_frames,
        audio_frame_count=audio_packets,
        start_pts=absolute_start_frame,
        end_pts=absolute_start_frame + expected_frames,
        start_sample=absolute_start_sample,
        audio_samples=expected_samples,
    )


def inspect_ts_segment(
    path: Path,
    *,
    profile: ProfileConfig,
    sequence: int,
    media: MediaConfig,
) -> dict[str, Any]:
    """Validate codecs, geometry, packet counts, and PTS continuity in one TS segment."""

    expected_frames = media.frame_count(sequence)
    expected_video_start = Fraction(sequence * media.frames_per_slice, media.fps)
    expected_audio_start = Fraction(
        int(round(sequence * media.slice_seconds * media.sample_rate)),
        media.sample_rate,
    )
    try:
        container = av.open(str(path), mode="r")
    except Exception as exc:
        raise WorkbenchError("E_MEDIA_FORMAT", f"cannot open transcoded segment: {path.name}", {"error": str(exc)}) from exc

    try:
        require(len(container.streams.video) == 1, "E_MEDIA_FORMAT", "segment must contain one video stream")
        require(len(container.streams.audio) == 1, "E_MEDIA_FORMAT", "segment must contain one audio stream")
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        require(video_stream.codec_context.name == "h264", "E_MEDIA_FORMAT", "video codec must be H.264")
        require(audio_stream.codec_context.name == "aac", "E_MEDIA_FORMAT", "audio codec must be AAC")
        require(video_stream.codec_context.width == profile.width, "E_MEDIA_FORMAT", "video width mismatch")
        require(video_stream.codec_context.height == profile.height, "E_MEDIA_FORMAT", "video height mismatch")
        require(audio_stream.codec_context.sample_rate == media.sample_rate, "E_MEDIA_FORMAT", "audio rate mismatch")
        require(audio_stream.codec_context.channels == media.channels, "E_MEDIA_FORMAT", "audio channels mismatch")

        decoded_frames = 0
        decoded_pts: list[Fraction] = []
        for frame in container.decode(video_stream):
            decoded_frames += 1
            decoded_pts.append(Fraction(frame.pts) * Fraction(frame.time_base))
        require(
            decoded_frames == expected_frames,
            "E_PTS_GAP",
            "decoded video frame count mismatch",
            expected=expected_frames,
            actual=decoded_frames,
        )
        expected_pts = [expected_video_start + Fraction(index, media.fps) for index in range(expected_frames)]
        for index, (actual, expected) in enumerate(zip(decoded_pts, expected_pts)):
            require(
                abs(actual - expected) <= Fraction(1, 90000),
                "E_PTS_GAP",
                "video PTS discontinuity",
                segment=sequence,
                frame=index,
                expected=float(expected),
                actual=float(actual),
            )
    finally:
        container.close()

    # Demux separately to inspect first audio packet timestamp and packet cadence.
    container = av.open(str(path), mode="r")
    try:
        audio_stream = container.streams.audio[0]
        audio_pts: list[Fraction] = []
        for packet in container.demux(audio_stream):
            if packet.pts is None:
                continue
            audio_pts.append(Fraction(packet.pts) * Fraction(packet.time_base))
        require(len(audio_pts) > 0, "E_MEDIA_FORMAT", "segment has no AAC packets")
        require(
            abs(audio_pts[0] - expected_audio_start) <= Fraction(1, 90000),
            "E_PTS_GAP",
            "audio start PTS discontinuity",
            segment=sequence,
            expected=float(expected_audio_start),
            actual=float(audio_pts[0]),
        )
        for previous, current in zip(audio_pts, audio_pts[1:]):
            require(
                abs((current - previous) - Fraction(1024, media.sample_rate)) <= Fraction(1, 90000),
                "E_PTS_GAP",
                "audio packet PTS cadence is not continuous",
                segment=sequence,
            )
    finally:
        container.close()

    return {
        "path": path.name,
        "profile": profile.name,
        "sequence": sequence,
        "decoded_frame_count": decoded_frames,
        "audio_packet_count": len(audio_pts),
        "video_start_pts_seconds": float(expected_video_start),
        "audio_start_pts_seconds": float(expected_audio_start),
    }


def validate_transcode_manifest(
    artifacts: list[TranscodeArtifact],
    profiles: tuple[ProfileConfig, ...],
    media: MediaConfig,
    root: Path,
) -> list[dict[str, Any]]:
    by_key = {(artifact.profile, artifact.sequence): artifact for artifact in artifacts}
    checks: list[dict[str, Any]] = []
    for profile in sorted(profiles, key=lambda item: (item.priority, item.name)):
        expected_sequences = list(range(media.slice_count))
        actual_sequences = sorted(seq for name, seq in by_key if name == profile.name)
        if actual_sequences != expected_sequences:
            missing = sorted(set(expected_sequences) - set(actual_sequences))
            raise WorkbenchError(
                "E_PTS_GAP",
                "profile has missing or extra segments",
                {"profile": profile.name, "missing": missing, "actual": actual_sequences},
            )
        for sequence in expected_sequences:
            artifact = by_key[(profile.name, sequence)]
            check = inspect_ts_segment(
                root / artifact.relative_path,
                profile=profile,
                sequence=sequence,
                media=media,
            )
            checks.append(check)
    return checks
