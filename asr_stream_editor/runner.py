"""Directory-level orchestration: scan, run command streams, write reports."""

from __future__ import annotations

import json
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .audio import load_wav
from .config import (AcousticConfig, PhonemeMap, StreamConfig, list_wav_files,
                     load_acoustic_config, load_phoneme_map, load_stream_config)
from .engine import StreamingEngine
from .errors import E_AUDIO_FORMAT, BusinessError, FailureRecord
from .full_decode import full_decode
from .snapshot import SnapshotStore


@dataclass
class RunArtifacts:
    report: Dict[str, Any]
    audio_failures: List[Dict[str, Any]]


class SignalGate:
    """Raises after the current chunk commits when SIGTERM/SIGINT arrives."""

    def __init__(self) -> None:
        self.requested = False
        self.previous: Dict[int, Any] = {}

    def install(self) -> None:
        self.previous[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handler)
        self.previous[signal.SIGINT] = signal.signal(signal.SIGINT, self._handler)

    def restore(self) -> None:
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)

    def _handler(self, signum: int, frame: Any) -> None:
        self.requested = True


def load_resources(input_dir: str) -> tuple[PhonemeMap, AcousticConfig, StreamConfig,
                                            List[Any], List[Dict[str, Any]]]:
    phoneme_map = load_phoneme_map(os.path.join(input_dir, "phoneme_map.json"))
    acoustic = load_acoustic_config(
        os.path.join(input_dir, "acoustic_params.json"), phoneme_map)
    stream = load_stream_config(os.path.join(input_dir, "stream_config.json"))
    clips = []
    failures: List[Dict[str, Any]] = []
    for path in list_wav_files(input_dir):
        try:
            clips.append(load_wav(path))
        except BusinessError as exc:
            failures.append(FailureRecord(
                source=os.path.basename(path), stage="load", code=exc.code,
                exception_type=type(exc).__name__, message=exc.message).to_dict())
    failures.sort(key=lambda item: item["source"])
    return phoneme_map, acoustic, stream, clips, failures


def run_stream(input_dir: str, output_dir: str,
               commands: Optional[List[Dict[str, Any]]] = None,
               fault_injection: Optional[Dict[str, Any]] = None,
               replacements: Optional[Dict[str, np.ndarray]] = None
               ) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    phoneme_map, acoustic, stream, clips, audio_failures = load_resources(input_dir)
    gate = SignalGate()
    gate.install()
    report: Dict[str, Any] = {
        "input_dir": os.path.abspath(input_dir),
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips": [],
        "audio_failures": audio_failures,
        "summary": {},
    }
    try:
        for clip in clips:
            snapshot_store = SnapshotStore(
                os.path.join(output_dir, "snapshots", _safe_name(clip.source)))
            recovery = snapshot_store.recovery_point()
            engine = StreamingEngine(clip, phoneme_map, acoustic, stream,
                                     snapshot_store, fault_injection,
                                     should_exit=lambda: gate.requested)
            clip_commands = commands or [{"command": "initial_decode"}]
            responses = []
            for command in clip_commands:
                replacement = None
                if command.get("command") == "replace_chunk":
                    key = (clip.source, int(command["chunk_index"]))
                    if replacements and key in replacements:
                        replacement = replacements[key]
                    elif command.get("replacement_wav"):
                        replacement_path = os.path.join(
                            input_dir, command["replacement_wav"])
                        replacement_clip = load_wav(replacement_path)
                        chunk = engine.chunks[int(command["chunk_index"])]
                        region_len = (chunk.own_end_sample
                                      - chunk.own_start_sample)
                        replacement = replacement_clip.samples[:region_len]
                        if replacement.shape[0] != region_len:
                            raise BusinessError(
                                E_AUDIO_FORMAT,
                                f"replacement too short: {replacement_path}")
                response = engine.process_command(command, replacement)
                payload = response.to_dict()
                payload["signal_exit_requested"] = gate.requested
                responses.append(payload)
                if gate.requested:
                    payload["exit_code"] = 0
                    break
            clip_report = _clip_report(engine, responses, recovery)
            report["clips"].append(clip_report)
    finally:
        gate.restore()
    report["summary"] = _overall_summary(report)
    report_path = os.path.join(output_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, ensure_ascii=False)
    return report


def _safe_name(name: str) -> str:
    return name.replace(os.sep, "_").replace(".", "_")


def _clip_report(engine: StreamingEngine, responses: List[Dict[str, Any]],
                 recovery: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    frame_shift_ms = engine.acoustic.frame_shift_ms
    final = responses[-1] if responses else {}
    return {
        "source": engine.clip.source,
        "audio_fingerprint": engine.clip.fingerprint,
        "recovery_point": recovery,
        "final_transcript": final.get("transcript", []),
        "final_text": "".join(item["word"] for item in final.get("transcript", [])),
        "chunk_stable_prefixes": _chunk_prefixes(engine, frame_shift_ms),
        "responses": responses,
        "within_budget_responses": engine.within_budget_responses,
        "over_budget_responses": engine.over_budget_responses,
        "alignment_conflicts": final.get("alignment_conflicts", []),
        "failures": [record.to_dict() for record in engine.failures],
        "budget_consumed_total_ms": engine.budget_consumed_total,
        "global_version": engine.version,
        "recovery_chunk_index": engine.recovery_chunk,
        "stage_timings_ms": {
            stage: {
                "calls": len(values),
                "total_ms": round(float(sum(values)), 4),
                "mean_ms": round(float(sum(values) / len(values)), 4) if values else 0.0,
            }
            for stage, values in sorted(engine.stage_timings.items())
        },
    }


def _chunk_prefixes(engine: StreamingEngine, frame_shift_ms: int) -> List[Dict[str, Any]]:
    if engine.merged is None:
        return []
    stable_until = engine.stability.stable_until_frame
    rows = []
    for index in sorted(engine.chunk_results):
        result = engine.chunk_results[index]
        if result.failed:
            rows.append({"chunk_index": index, "stable_prefix_text": "",
                         "status": "failed"})
            continue
        stable_tokens = [token for token in result.tokens
                         if token.end_frame < stable_until]
        rows.append({
            "chunk_index": index,
            "frame_span": list(result.frame_span),
            "stable_prefix_text": "".join(token.word for token in stable_tokens),
            "status": "ok",
        })
    return rows


def _overall_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    within = sum(item["within_budget_responses"] for item in report["clips"])
    over = sum(item["over_budget_responses"] for item in report["clips"])
    failures = sum(len(item["failures"]) for item in report["clips"])
    conflicts = sum(len(item["alignment_conflicts"]) for item in report["clips"])
    return {
        "clips": len(report["clips"]),
        "within_budget_responses": within,
        "over_budget_responses": over,
        "failure_records": failures + len(report["audio_failures"]),
        "alignment_conflicts": conflicts,
    }


def run_full_reference(input_dir: str, output_dir: str,
                       overlap_ms: Optional[int] = None,
                       min_consistent_frames: Optional[int] = None,
                       backoff_scale: float = 1.0,
                       samples_override: Optional[Dict[str, np.ndarray]] = None
                       ) -> Dict[str, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    phoneme_map, acoustic, stream, clips, audio_failures = load_resources(input_dir)
    reference: Dict[str, Dict[str, Any]] = {}
    for clip in clips:
        samples = (samples_override or {}).get(clip.source, clip.samples)
        result = full_decode(
            clip, samples, phoneme_map, acoustic, stream,
            overlap_ms=overlap_ms if overlap_ms is not None else stream.overlap_ms,
            min_consistent_frames=(min_consistent_frames
                                   if min_consistent_frames is not None
                                   else stream.min_consistent_frames),
            backoff_scale=backoff_scale)
        reference[clip.source] = result
    path = os.path.join(output_dir, "full_reference.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(reference, handle, sort_keys=True, indent=2, ensure_ascii=False)
    return reference
