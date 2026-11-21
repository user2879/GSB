"""End-to-end decode, normalize, align, diagnose and aggregate pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from . import INTERNAL_MODEL_VERSION, MODEL_DESCRIPTION
from .alignment import align_observations
from .diagnosis import diagnose_streams, diagnosis_digest
from .errors import format_error
from .events import load_event_bundle
from .model import AlignmentResult, EventBundle, FlowDiagnosis, FrameBundle
from .rates import aggregate_rates, rates_digest
from .util import content_digest, file_digest
from .wire import read_wire_bundle


@dataclass
class DiagnosisSession:
    wire_path: str
    wire_manifest_path: str
    event_paths: list[str]
    window_ms: int
    frame_bundle: FrameBundle
    event_bundles: list[EventBundle]
    alignment: AlignmentResult
    diagnoses: list[FlowDiagnosis]
    rate_report: dict[str, Any]
    report: dict[str, Any]
    report_digest: str
    input_digest: str
    version_combo: dict[str, Any]

    @property
    def event_versions(self) -> list[str]:
        return sorted({bundle.version for bundle in self.event_bundles})

    def model_payload(self) -> dict[str, Any]:
        frames = sorted(
            self.frame_bundle.frames,
            key=lambda frame: (
                frame.flow_id,
                frame.seq,
                frame.arrival_ordinal,
                frame.send_ms,
                frame.flags,
                frame.payload_len,
                frame.source_id,
            ),
        )
        events = sorted(
            (event for bundle in self.event_bundles for event in bundle.records),
            key=lambda event: (
                event.flow_id,
                event.event_seq,
                event.loss_count,
                event.source_version,
                event.source_id,
                event.link_id or "",
                event.time_window_ms.start_ms if event.time_window_ms else -1,
                event.time_window_ms.end_ms if event.time_window_ms else -1,
                event.retransmit,
                event.reorder_depth,
            ),
        )
        return {
            "model_version": INTERNAL_MODEL_VERSION,
            "model_description": MODEL_DESCRIPTION,
            "version_combo": self.version_combo,
            "inputs": {
                "wire_path": self.wire_path,
                "wire_manifest_path": self.wire_manifest_path,
                "event_paths": list(self.event_paths),
                "input_digest": self.input_digest,
            },
            "window_ms": self.window_ms,
            "wire": {
                "source_id": self.frame_bundle.source_id,
                "model_version": self.frame_bundle.model_version,
                "time_basis": self.frame_bundle.time_basis.to_dict(),
                "alignment_version": self.frame_bundle.alignment_version,
                "raw_digest": self.frame_bundle.raw_digest,
                "wire_digest": self.frame_bundle.wire_digest,
                "frame_count": len(frames),
                "frames": [frame.summary() for frame in frames],
            },
            "events": {
                "bundles": [bundle.to_dict() for bundle in self.event_bundles],
                "records": [event.to_dict() for event in events],
            },
            "alignment": self.alignment.to_dict(),
            "diagnosis": {
                "serialization": "flow-diagnosis-v1",
                "flows": [diagnosis.to_dict() for diagnosis in self.diagnoses],
                "digest": diagnosis_digest(self.diagnoses),
            },
            "rates": self.rate_report,
            "rates_digest": rates_digest(self.rate_report),
        }

    def content_digest(self) -> str:
        return content_digest(self.model_payload())

    def version_signature(self) -> dict[str, Any]:
        return {
            "model_version": INTERNAL_MODEL_VERSION,
            "event_versions": self.event_versions,
            "basis_version": self.alignment.basis_version,
            "window_ms": self.window_ms,
            "conclusion_digest": self.report_digest,
            "content_digest": self.content_digest(),
        }


def _input_digest(
    wire_path: str,
    manifest_path: str,
    event_paths: list[str],
) -> str:
    return content_digest(
        {
            "wire": file_digest(wire_path),
            "wire_manifest": file_digest(manifest_path),
            "events": sorted(file_digest(path) for path in event_paths),
        }
    )


def build_session(
    *,
    wire_path: str,
    wire_manifest_path: str,
    event_paths: list[str],
    window_ms: int,
    inject_wire_corruption: bool = False,
    inject_event_version_conflict: bool = False,
) -> DiagnosisSession:
    if not event_paths:
        raise format_error(
            "at least one loss-event list is required",
            record_number=0,
            field="events",
        )
    if isinstance(window_ms, bool) or not isinstance(window_ms, int) or window_ms <= 0:
        raise format_error(
            "window_ms must be a positive integer",
            record_number=0,
            field="window_ms",
            actual=window_ms,
        )

    frame_bundle = read_wire_bundle(
        wire_path,
        wire_manifest_path,
        corrupt=inject_wire_corruption,
    )
    event_bundles = [
        load_event_bundle(
            path,
            inject_version_conflict=inject_event_version_conflict and index == 0,
        )
        for index, path in enumerate(event_paths)
    ]
    alignment = align_observations(frame_bundle, event_bundles)
    diagnoses = diagnose_streams(frame_bundle, event_bundles)
    try:
        rate_report = aggregate_rates(frame_bundle, diagnoses, window_ms)
    except ValueError as exc:
        raise format_error(
            str(exc),
            record_number=0,
            field="window_ms",
        ) from exc

    event_versions = sorted({bundle.version for bundle in event_bundles})
    input_digest = _input_digest(wire_path, wire_manifest_path, event_paths)
    version_combo = {
        "model_version": INTERNAL_MODEL_VERSION,
        "wire_format_version": frame_bundle.model_version,
        "event_list_versions": event_versions,
        "alignment_basis_version": alignment.basis_version,
        "alignment_policy": "translation_only",
        "basis_signature": content_digest(
            {
                "wire_basis": frame_bundle.time_basis.to_dict(),
                "event_bases": sorted(
                    (
                        bundle.time_basis.to_dict() if bundle.time_basis else None,
                        bundle.offset_to_frame_ms,
                    )
                    for bundle in event_bundles
                ),
            }
        ),
    }

    frames_sorted = sorted(
        frame_bundle.frames,
        key=lambda frame: (
            frame.flow_id,
            frame.seq,
            frame.arrival_ordinal,
            frame.send_ms,
            frame.flags,
            frame.payload_len,
        ),
    )
    conclusion = {
        "model_version": INTERNAL_MODEL_VERSION,
        "version_combo": version_combo,
        "alignment": alignment.to_dict(),
        "diagnosis": {
            "serialization": "flow-diagnosis-v1",
            "flows": [diagnosis.to_dict() for diagnosis in diagnoses],
            "digest": diagnosis_digest(diagnoses),
        },
        "rates": rate_report,
        "rates_digest": rates_digest(rate_report),
    }
    report_digest = content_digest(conclusion)
    report = {
        "ok": True,
        "model": {
            "version": INTERNAL_MODEL_VERSION,
            "description": MODEL_DESCRIPTION,
        },
        "version_combo": version_combo,
        "inputs": {
            "wire_path": wire_path,
            "wire_manifest_path": wire_manifest_path,
            "event_paths": list(event_paths),
            "input_digest": input_digest,
        },
        "wire": {
            "source_id": frame_bundle.source_id,
            "model_version": frame_bundle.model_version,
            "time_basis": frame_bundle.time_basis.to_dict(),
            "alignment_version": frame_bundle.alignment_version,
            "raw_digest": frame_bundle.raw_digest,
            "wire_digest": frame_bundle.wire_digest,
            "frame_count": len(frames_sorted),
            "frames": [frame.summary() for frame in frames_sorted],
        },
        "events": {
            "versions": event_versions,
            "bundles": [bundle.to_dict() for bundle in event_bundles],
        },
        "alignment": alignment.to_dict(),
        "diagnosis": conclusion["diagnosis"],
        "rates": rate_report,
        "determinism": {
            "serialization": "canonical-json-sort-keys-v1",
            "flow_order": "flow_id_asc",
            "gap_order": "start_seq_asc_then_classification",
            "window_order": "window_start_ms_asc",
            "traversal_independent_digest": report_digest,
            "per_flow_digests": {
                diagnosis.flow_id: diagnosis.gap_digest for diagnosis in diagnoses
            },
        },
        "report_digest": report_digest,
    }
    return DiagnosisSession(
        wire_path=wire_path,
        wire_manifest_path=wire_manifest_path,
        event_paths=list(event_paths),
        window_ms=window_ms,
        frame_bundle=frame_bundle,
        event_bundles=event_bundles,
        alignment=alignment,
        diagnoses=diagnoses,
        rate_report=rate_report,
        report=report,
        report_digest=report_digest,
        input_digest=input_digest,
        version_combo=version_combo,
    )


def report_json(session: DiagnosisSession) -> str:
    return json.dumps(
        session.report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )

