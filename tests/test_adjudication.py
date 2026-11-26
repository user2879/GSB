from __future__ import annotations

from voice_diag.adjudicate import adjudicate_frames, merge_command_conclusions
from voice_diag.config import AnalysisConfig, AdjudicationConfig


ANALYSIS = AnalysisConfig(
    window_length=2400,
    window_shift=1200,
    frame_length=400,
    frame_shift=160,
    mel_bands=12,
    quant_scale=1000,
    frac_bits=6,
    allowed_quant_scales=(100, 1000, 2000),
    allowed_frac_bits=(4, 6, 8),
    posterior_quantum_ppm=1000,
)
ADJUDICATION = AdjudicationConfig(
    frame_endpoint_threshold_ppm=400000,
    min_command_ms=120,
    tie_half_step_ppm=500,
    min_overlap_frames=1,
)


def _contribution(
    *,
    interval_id: str,
    speaker_id: str,
    start: int,
    end: int,
    weight: int,
    margin: int,
    trust: int,
    code: int,
    posterior: dict[int, int],
) -> dict:
    return {
        "frame_index": 0,
        "frame_center_sample": 200,
        "interval_id": interval_id,
        "speaker_id": speaker_id,
        "interval_start_sample": start,
        "interval_end_sample": end,
        "posterior_ppm": posterior,
        "margin_ppm": margin,
        "best_code": code,
        "runner_up_code": None,
        "trust_ppm": trust,
        "overlap_frames": 3,
        "speech_ppm": 900000,
        "feature_digest": "f" * 64,
        "weight": weight,
    }


def _payload(contributions: list[dict]) -> list[dict]:
    return [
        {
            "window_id": "W000000",
            "window_index": 0,
            "sample_start": 0,
            "sample_end": 2400,
            "frame_indices": [0],
            "sources": [],
            "contributions": contributions,
        }
    ]


def test_overlap_tie_uses_start_then_speaker_and_marks_boundary() -> None:
    payload = _payload(
        [
            _contribution(
                interval_id="I_EARLY",
                speaker_id="S09",
                start=100,
                end=1000,
                weight=1000,
                margin=100,
                trust=10,
                code=1,
                posterior={1: 900000, 2: 100000},
            ),
            _contribution(
                interval_id="I_LATE",
                speaker_id="S01",
                start=200,
                end=1000,
                weight=1000,
                margin=100,
                trust=10,
                code=2,
                posterior={2: 900000, 1: 100000},
            ),
        ]
    )
    frames, summary = adjudicate_frames(
        attribution_payloads=payload,
        frame_metrics=[{"frame_index": 0, "start_sample": 0, "end_sample": 400}],
        candidate_mask=[True],
        analysis=ANALYSIS,
        adjudication=ADJUDICATION,
    )
    assert frames[0]["winning_interval_id"] == "I_EARLY"
    assert frames[0]["boundary"] is True
    assert frames[0]["interval_arbitration"] == "interval_tie_start_then_speaker"
    assert summary["boundary_frame_indices"] == [0]
    assert summary["overlap_frame_count"] == 1
    assert frames[0]["contested"] is True


def test_overlap_same_start_uses_speaker_lexical_order() -> None:
    payload = _payload(
        [
            _contribution(
                interval_id="I_A",
                speaker_id="S02",
                start=100,
                end=1000,
                weight=500,
                margin=100,
                trust=5,
                code=1,
                posterior={1: 800000, 2: 200000},
            ),
            _contribution(
                interval_id="I_B",
                speaker_id="S01",
                start=100,
                end=1000,
                weight=500,
                margin=100,
                trust=5,
                code=2,
                posterior={2: 800000, 1: 200000},
            ),
        ]
    )
    frames, _ = adjudicate_frames(
        attribution_payloads=payload,
        frame_metrics=[{"frame_index": 0, "start_sample": 0, "end_sample": 400}],
        candidate_mask=[True],
        analysis=ANALYSIS,
        adjudication=ADJUDICATION,
    )
    assert frames[0]["winning_interval_id"] == "I_B"
    assert frames[0]["winning_speaker_id"] == "S01"
    assert frames[0]["boundary"] is True


def test_command_disagreement_records_arbitration_evidence() -> None:
    payload = _payload(
        [
            _contribution(
                interval_id="I001",
                speaker_id="S01",
                start=100,
                end=1000,
                weight=3000,
                margin=300,
                trust=10,
                code=1,
                posterior={1: 950000, 2: 50000},
            ),
            _contribution(
                interval_id="I002",
                speaker_id="S02",
                start=200,
                end=1000,
                weight=1000,
                margin=100,
                trust=10,
                code=2,
                posterior={2: 900000, 1: 100000},
            ),
        ]
    )
    frames, _ = adjudicate_frames(
        attribution_payloads=payload,
        frame_metrics=[{"frame_index": 0, "start_sample": 0, "end_sample": 400}],
        candidate_mask=[True],
        analysis=ANALYSIS,
        adjudication=ADJUDICATION,
    )
    assert frames[0]["winning_interval_id"] == "I001"
    assert frames[0]["command_code"] == 1
    assert frames[0]["contested"] is True
    assert frames[0]["runner_up_interval_id"] == "I002"
    assert frames[0]["command_arbitration"] == "higher_interval_ownership"


def test_minimum_command_segment_suppresses_short_segments_without_hiding_them() -> None:
    frames = []
    for index in range(15):
        frames.append(
            {
                "frame_index": index,
                "active": True,
                "start_sample": index * 160,
                "end_sample": index * 160 + 400,
                "winning_interval_id": "I001",
                "winning_speaker_id": "S01",
                "command_code": 1,
                "speech_ppm": 900000,
                "command_probability_ppm": 800000,
                "contested": False,
                "arbitration": "uncontested",
                "command_arbitration": "uncontested",
                "boundary": False,
            }
        )
    frames.append(
        {
            "frame_index": 15,
            "active": True,
            "start_sample": 2400,
            "end_sample": 2800,
            "winning_interval_id": "I001",
            "winning_speaker_id": "S01",
            "command_code": 2,
            "speech_ppm": 900000,
            "command_probability_ppm": 800000,
            "contested": False,
            "arbitration": "uncontested",
            "command_arbitration": "uncontested",
            "boundary": False,
        }
    )
    merged = merge_command_conclusions(frames, analysis=ANALYSIS, adjudication=ADJUDICATION)
    assert merged["minimum_command_frames"] == 12
    assert merged["accepted_count"] == 1
    assert merged["suppressed_count"] == 1
    assert merged["suppressed"][0]["command_code"] == 2
