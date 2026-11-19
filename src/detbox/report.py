"""Cross-version verification report (requirement 8)."""
from __future__ import annotations

from pathlib import Path

from .exchange import (
    parse_v1_text,
    parse_v2_document,
    serialize_v1,
    serialize_v2_document,
    v1_to_v2,
    v2_to_v1,
)
from .models import FrameCandidates
from .util import atomic_write_json, content_hash


def round_trip_compare(
    frames: list[FrameCandidates], classes, width: int, height: int
) -> dict:
    """Serialize -> parse in both formats and quantify representation loss."""
    v1_text = serialize_v1(frames)
    v1_frames, v1_failures = parse_v1_text(v1_text, classes, width, height)
    v2_doc = serialize_v2_document(frames, classes)
    v2_frames, v2_failures = parse_v2_document(v2_doc, classes, width, height)

    upgraded = v1_to_v2(v1_frames)
    downgraded, sidecar = v2_to_v1(v2_frames, classes)

    coord_deltas: list[float] = []
    score_deltas: list[float] = []
    for original, downgraded_frame in zip(
        sorted(v2_frames, key=lambda f: (f.sequence, f.frame)),
        sorted(downgraded, key=lambda f: (f.sequence, f.frame)),
    ):
        for cand_a, cand_b in zip(original.candidates, downgraded_frame.candidates):
            for a, b in zip(
                (cand_a.x1, cand_a.y1, cand_a.x2, cand_a.y2),
                (cand_b.x1, cand_b.y1, cand_b.x2, cand_b.y2),
            ):
                coord_deltas.append(abs(a - b))
            score_deltas.append(abs(cand_a.score - cand_b.score))

    bins = {
        "0": 0,
        "(0,0.5]": 0,
        "(0.5,1.0]": 0,
        ">1.0": 0,
    }
    for delta in coord_deltas:
        if delta == 0:
            bins["0"] += 1
        elif delta <= 0.5:
            bins["(0,0.5]"] += 1
        elif delta <= 1.0:
            bins["(0.5,1.0]"] += 1
        else:
            bins[">1.0"] += 1

    class_roundtrip = {}
    for code, name in sorted(classes.code_to_name.items()):
        recovered = classes.name_for(classes.code_for(name))
        class_roundtrip[name] = {
            "code": code,
            "recovered_name": recovered,
            "consistent": recovered == name,
        }

    return {
        "coord_delta_histogram": bins,
        "coord_delta_count": len(coord_deltas),
        "max_coord_delta": max(coord_deltas, default=0.0),
        "max_score_delta": max(score_deltas, default=0.0),
        "mean_score_delta": (sum(score_deltas) / len(score_deltas)) if score_deltas else 0.0,
        "class_mapping_roundtrip": class_roundtrip,
        "v1_parse_failures": v1_failures,
        "v2_parse_failures": v2_failures,
        "sidecar_record_count": len(sidecar["records"]),
        "sidecar_fingerprint": content_hash(sidecar),
    }


def build_report(
    run_result: dict,
    classes,
    width: int,
    height: int,
    extra: dict | None = None,
) -> dict:
    frames = [
        frame
        for sequence in sorted(run_result["frames"])
        for fno in sorted(run_result["frames"][sequence])
        for frame in [run_result["frames"][sequence][fno]]
    ]
    tracks_payload = []
    for sequence in sorted(run_result["tracks"]):
        for track in run_result["tracks"][sequence]:
            tracks_payload.append(
                {
                    "track_id": track.track_id,
                    "class_code": track.class_code,
                    "class_name": classes.name_for(track.class_code),
                    "sequence": track.sequence,
                    "frames": list(track.frames),
                    "integer_boxes": [list(box) for box in track.boxes],
                    "scores_per_mille": list(track.scores),
                    "closed": track.closed,
                    "velocity": [track.vx, track.vy],
                }
            )

    failed_frames = sorted(
        {f"{item['sequence']}:{item['frame']}" for item in run_result["failures"]}
    )
    report = {
        "snapshot_version": run_result["manifest"]["snapshot_version"],
        "format": run_result["manifest"]["format"],
        "business_codes": [
            "E_CAND_FORMAT",
            "E_CONFIG_MISMATCH",
            "E_IMAGE_TRUNCATED",
            "E_WEIGHT_NONFINITE",
            "E_CLASS_UNKNOWN",
            "E_STAGE_FAILED",
        ],
        "tracks": tracks_payload,
        "round_trip": round_trip_compare(frames, classes, width, height),
        "failed_frames": failed_frames,
        "failure_list": run_result["failures"],
        "injection_hits": run_result["injection_hits"],
        "invalidated": run_result["invalidated"],
        "reused": run_result["reused"],
        "recomputed": run_result["recomputed"],
        "eviction_log": run_result["evictions"],
        "config_hash": run_result["manifest"]["config_hash"],
        "fingerprints": {
            "detection_params": run_result["manifest"]["detection_params_fingerprint"],
            "match_config": run_result["manifest"]["match_config_fingerprint"],
            "format_version": run_result["manifest"]["format_version_fingerprint"],
            "sidecar": run_result["manifest"]["sidecar_fingerprint"],
        },
        "stage_versions": run_result["manifest"]["stage_versions"],
        "exit_code": run_result.get("exit_code", 0),
    }
    if extra:
        report.update(extra)
    return report


def write_report(path: str | Path, report: dict) -> None:
    atomic_write_json(path, report)
