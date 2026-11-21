from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from .audio import AudioFile
from .config import AnalysisConfig


@dataclass
class Segment:
    segment_id: str
    file: str
    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int
    speaker: str | None = None
    style_hint: str | None = None
    origin: str = "segmentation"

    def stable_key(self) -> tuple[str, int, int, str]:
        return (self.file, self.start_sample, self.end_sample, self.segment_id)

    def midpoint_ms(self) -> float:
        return ((self.start_sample + self.end_sample) / 2.0) / 16000.0 * 1000.0

    def duration_ms(self) -> float:
        return (self.end_sample - self.start_sample) / 16000.0 * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "file": self.file,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": round(self.start_sample / 16000.0 * 1000.0, 6),
            "end_ms": round(self.end_sample / 16000.0 * 1000.0, 6),
            "speaker": self.speaker,
            "style_hint": self.style_hint,
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        return cls(
            segment_id=str(value["segment_id"]),
            file=str(value["file"]),
            start_sample=int(value["start_sample"]),
            end_sample=int(value["end_sample"]),
            start_frame=int(value["start_frame"]),
            end_frame=int(value["end_frame"]),
            speaker=value.get("speaker"),
            style_hint=value.get("style_hint"),
            origin=str(value.get("origin", "snapshot")),
        )


def make_segment_id(file: str, start_sample: int, end_sample: int) -> str:
    return f"{file}:seg-{start_sample:08d}-{end_sample:08d}"


@dataclass
class AudioDocument:
    files: dict[str, AudioFile]
    segments: list[Segment]
    input_annotations: tuple[dict[str, Any], ...] = ()

    def ordered_segments(self) -> list[Segment]:
        return sorted(self.segments, key=lambda segment: segment.stable_key())

    def segment_by_id(self, segment_id: str) -> Segment:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise KeyError(segment_id)

    def segments_for_file(self, file_name: str) -> list[Segment]:
        return sorted((segment for segment in self.segments if segment.file == file_name), key=lambda item: item.stable_key())

    def frame_span(self, segment: Segment) -> tuple[int, int]:
        audio = self.files[segment.file]
        return (
            max(0, min(segment.start_sample, audio.sample_count)),
            max(0, min(segment.end_sample, audio.sample_count)),
        )

    def segment_samples(self, segment: Segment):
        start, end = self.frame_span(segment)
        return self.files[segment.file].samples[start:end]

    def clone(self) -> "AudioDocument":
        return AudioDocument(
            files={name: AudioFile(audio.name, audio.samples.copy(), audio.source_path) for name, audio in self.files.items()},
            segments=[copy.deepcopy(segment) for segment in self.segments],
            input_annotations=self.input_annotations,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "files": sorted(self.files),
            "segments": [segment.as_dict() for segment in self.ordered_segments()],
            "input_annotations": list(self.input_annotations),
        }

    @classmethod
    def from_state(cls, value: dict[str, Any], files: dict[str, AudioFile]) -> "AudioDocument":
        return cls(
            files=files,
            segments=[Segment.from_dict(item) for item in value["segments"]],
            input_annotations=tuple(value.get("input_annotations", [])),
        )

    def validate(self, config: AnalysisConfig) -> list[dict[str, Any]]:
        failures: list[dict[str, Any]] = []
        seen: set[str] = set()
        for segment in self.segments:
            if segment.segment_id in seen:
                failures.append({"code": "E_SEGMENT_DUPLICATE", "message": "duplicate segment id", "details": {"segment_id": segment.segment_id}})
            seen.add(segment.segment_id)
            if segment.file not in self.files:
                failures.append({"code": "E_SEGMENT_FILE", "message": "segment references an unknown file", "details": {"segment_id": segment.segment_id}})
                continue
            if not (0 <= segment.start_sample < segment.end_sample <= self.files[segment.file].sample_count):
                failures.append(
                    {
                        "code": "E_SEGMENT_RANGE",
                        "message": "segment range is outside its audio file",
                        "details": {"segment_id": segment.segment_id},
                    }
                )
            if segment.end_sample - segment.start_sample < config.frame_length_samples:
                failures.append(
                    {
                        "code": "E_SEGMENT_TOO_SHORT",
                        "message": "segment is shorter than one frame",
                        "details": {"segment_id": segment.segment_id},
                    }
                )
        return failures

    def refresh_frame_indices(self, config: AnalysisConfig) -> None:
        hop = config.frame_shift_samples
        frame_length = config.frame_length_samples
        for segment in self.segments:
            segment.start_frame = int(segment.start_sample // hop)
            span = segment.end_sample - segment.start_sample
            frame_count = 1 + (span - frame_length) // hop if span >= frame_length else 0
            segment.end_frame = segment.start_frame + frame_count


def find_segment(
    document: AudioDocument,
    selector: dict[str, Any],
    *,
    adjacent: bool = False,
) -> Segment:
    file_name = selector.get("file")
    candidates = [
        segment
        for segment in document.ordered_segments()
        if file_name is None or segment.file == str(file_name)
    ]
    if "segment_id" in selector:
        segment_id = str(selector["segment_id"])
        for segment in candidates:
            if segment.segment_id == segment_id:
                return segment
        raise KeyError(f"segment_id {segment_id}")
    if "index" in selector:
        try:
            index = int(selector["index"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise KeyError("segment index must be an integer") from exc
        if 0 <= index < len(candidates):
            return candidates[index]
        raise KeyError(f"segment index {index}")
    if "contains_ms" in selector:
        try:
            point = float(selector["contains_ms"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise KeyError("contains_ms must be numeric") from exc
        for segment in candidates:
            if segment.start_sample / 16.0 <= point < segment.end_sample / 16.0:
                return segment
        raise KeyError(f"no segment contains {point} ms")
    if "start_ms" in selector:
        try:
            point = float(selector["start_ms"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise KeyError("start_ms must be numeric") from exc
        for segment in candidates:
            if abs(segment.start_sample / 16.0 - point) < 0.5:
                return segment
        raise KeyError(f"no segment starts at {point} ms")
    if adjacent:
        raise KeyError("adjacent selector requires a concrete target")
    raise KeyError("selector must contain segment_id, index, contains_ms, or start_ms")


def replace_range(items: Iterable[Segment], new_segment: Segment) -> list[Segment]:
    return [segment for segment in items if segment.segment_id != new_segment.segment_id]
