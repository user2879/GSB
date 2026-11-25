"""Load and validate the segment manifest and the diagnostic command stream.

The loader is the single place where untrusted input enters the component.
Invalid segments are *not* silently skipped: they are captured as
:class:`~vad_diag.errors.FailureRecord` entries (``E_AUDIO_FORMAT`` /
``E_FIELD_INVALID``) and isolated, while every well-formed segment is admitted.

A command stream drives the incremental engine.  Supported commands:

* ``append``     - admit a new segment (by manifest reference),
* ``replace``    - swap one segment's content under the same id,
* ``delete``     - remove a segment,
* ``reorder``    - declare an explicit arrival permutation,
* ``set_params`` - change thresholds / minimum-length / geometry parameters,
* ``set_quant``  - switch frac_bits / quant_mode,
* ``query``      - request a summary under the latency budget,
* ``snapshot``   - force a stage snapshot,
* ``shutdown``   - request a graceful exit report.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import (
    FailureRecord,
    VADFieldError,
    E_AUDIO_FORMAT,
    E_FIELD_INVALID,
)
from .segments import Segment, parse_wav_segment, build_in_memory_segment

VALID_COMMANDS = (
    "append", "replace", "delete", "reorder", "set_params",
    "set_quant", "query", "snapshot", "shutdown",
)


@dataclass
class LoadResult:
    segments: Dict[str, Segment] = field(default_factory=dict)
    failures: List[FailureRecord] = field(default_factory=list)

    @property
    def failure_seq(self) -> int:
        return len(self.failures)


def _record_failure(result: LoadResult, code: str, stage: str, message: str,
                    item_id: Optional[str], detail: Optional[Dict[str, Any]]
                    ) -> None:
    result.failures.append(FailureRecord(
        seq=len(result.failures), code=code, stage=stage, message=message,
        item_id=item_id, detail=detail or {}))


def load_segment_manifest(path: str) -> LoadResult:
    """Load every declared segment, collecting validation failures."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not isinstance(raw.get("segments"), list):
        raise VADFieldError(
            "segment manifest must contain a 'segments' list",
            E_FIELD_INVALID, stage="load")
    base_dir = os.path.dirname(os.path.abspath(path))
    result = LoadResult()
    seen_ids = set()
    for entry in raw["segments"]:
        seg_id = entry.get("segment_id") if isinstance(entry, dict) else None
        if not seg_id or not isinstance(seg_id, str):
            _record_failure(result, E_FIELD_INVALID, "load",
                            "segment entry missing string segment_id",
                            None, {"entry": _safe(entry)})
            continue
        if seg_id in seen_ids:
            _record_failure(result, E_FIELD_INVALID, "load",
                            "duplicate segment_id", seg_id, {})
            continue
        try:
            segment = _build_segment(entry, seg_id, base_dir)
        except VADFieldError as exc:
            _record_failure(result, exc.code, exc.stage, str(exc),
                            exc.item_id or seg_id, exc.detail)
            continue
        seen_ids.add(seg_id)
        result.segments[seg_id] = segment
    return result


def _safe(entry: Any) -> Any:
    if isinstance(entry, dict):
        return {k: v for k, v in entry.items()
                if k not in ("samples", "samples_b64")}
    return entry


def _build_segment(entry: Dict[str, Any], seg_id: str, base_dir: str
                   ) -> Segment:
    capture_ms = entry.get("capture_ms")
    arrival_ms = entry.get("arrival_ms")
    process_ms = entry.get("process_ms", 0)
    wav_path = entry.get("wav")
    inline = entry.get("samples")
    if wav_path is not None:
        full = wav_path if os.path.isabs(wav_path) else \
            os.path.join(base_dir, wav_path)
        return parse_wav_segment(full, seg_id, capture_ms, arrival_ms,
                                 process_ms)
    if inline is not None:
        if not isinstance(inline, list):
            raise VADFieldError(
                "samples must be a list of signed 16-bit integers",
                E_FIELD_INVALID, item_id=seg_id)
        return build_in_memory_segment(
            seg_id, [int(v) for v in inline], capture_ms, arrival_ms,
            process_ms)
    raise VADFieldError(
        "segment must define either 'wav' or 'samples'", E_FIELD_INVALID,
        item_id=seg_id)


@dataclass(frozen=True)
class Command:
    seq: int
    op: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {"seq": self.seq, "op": self.op}
        if self.payload:
            out["payload"] = self.payload
        return out


def load_command_stream(path: str) -> List[Command]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict) and isinstance(raw.get("commands"), list):
        items = raw["commands"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise VADFieldError(
            "command stream must be a list or {'commands': [...]}",
            E_FIELD_INVALID, stage="load")
    commands: List[Command] = []
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict) or "op" not in entry:
            raise VADFieldError(
                f"command[{idx}] missing 'op'", E_FIELD_INVALID,
                stage="load", detail={"index": idx})
        op = entry["op"]
        if op not in VALID_COMMANDS:
            raise VADFieldError(
                f"unknown command op {op!r}", E_FIELD_INVALID, stage="load",
                detail={"index": idx, "allowed": list(VALID_COMMANDS)})
        payload = entry.get("payload", {})
        if not isinstance(payload, dict):
            raise VADFieldError(
                f"command[{idx}] payload must be an object",
                E_FIELD_INVALID, stage="load")
        commands.append(Command(seq=idx, op=op, payload=dict(payload)))
    return commands
