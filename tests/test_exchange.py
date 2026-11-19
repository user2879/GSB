import pytest

from detbox.errors import E_CAND_FORMAT, E_CLASS_UNKNOWN
from detbox.exchange import (
    candidate_to_v1_line,
    parse_v1_line,
    parse_v1_text,
    parse_v2_document,
    serialize_v1,
    serialize_v2_document,
    v1_to_v2,
    v2_to_v1,
)
from detbox.config import DEFAULT_CLASS_TABLE
from detbox.models import FORMAT_V2, FrameCandidates

W, H = 160, 120
CLASSES = DEFAULT_CLASS_TABLE


def make_v1_line(seq="seqA", frame=1, code=1, box=(10, 20, 30, 40), score=500, missing=0):
    x1, y1, x2, y2 = box
    return (
        seq.ljust(10)
        + str(frame).rjust(5)
        + str(code).rjust(3)
        + str(x1).rjust(5)
        + str(y1).rjust(5)
        + str(x2).rjust(5)
        + str(y2).rjust(5)
        + str(score).rjust(7)
        + str(missing).rjust(1)
    )


def test_v1_round_trip_lossless():
    line = make_v1_line()
    candidate = parse_v1_line(line, CLASSES, W, H)
    assert candidate.int_box == (10, 20, 30, 40)
    assert candidate.int_score == 500
    assert candidate.score == 0.5
    assert candidate.class_code == 1
    assert candidate_to_v1_line(candidate, "seqA", 1) == line


def test_v1_bad_width_and_bounds_and_class():
    with pytest.raises(Exception) as exc:
        parse_v1_line("short", CLASSES, W, H)
    assert exc.value.code == E_CAND_FORMAT

    bad_bounds = make_v1_line(box=(150, 110, 200, 140))
    with pytest.raises(Exception) as exc:
        parse_v1_line(bad_bounds, CLASSES, W, H)
    assert exc.value.code == E_CAND_FORMAT

    bad_class = make_v1_line(code=9)
    with pytest.raises(Exception) as exc:
        parse_v1_line(bad_class, CLASSES, W, H)
    assert exc.value.code == E_CLASS_UNKNOWN


def test_v1_parse_collects_failures():
    good = make_v1_line(frame=1)
    bad = make_v1_line(frame=2, code=9)
    frames, failures = parse_v1_text(good + "\n" + bad + "\n", CLASSES, W, H)
    assert len(frames) == 1
    assert len(failures) == 1
    assert failures[0]["code"] == E_CLASS_UNKNOWN


def test_v2_parse_and_class_mapping():
    doc = {
        "format": "v2",
        "sequences": [
            {
                "sequence": "seqA",
                "frames": [
                    {
                        "frame": 0,
                        "candidates": [
                            {
                                "box": [10.2, 20.4, 30.7, 40.9],
                                "score": 0.523,
                                "class_name": "square",
                                "precision_label": "float",
                                "ref_score": 0.555,
                            }
                        ],
                    }
                ],
            }
        ],
    }
    frames, failures = parse_v2_document(doc, CLASSES, W, H)
    assert failures == []
    candidate = frames[0].candidates[0]
    assert candidate.class_code == 1
    assert candidate.ref_score == 0.555

    downgraded, sidecar = v2_to_v1(frames, CLASSES)
    v1_candidate = downgraded[0].candidates[0]
    assert v1_candidate.int_box == (10, 20, 31, 41)
    assert v1_candidate.int_score == 523
    record = sidecar["records"][0]
    assert record["dropped_precision_label"] == "float"
    assert record["dropped_ref_score"] == 0.555
    assert record["class_name"] == "square"
    assert record["coordinate_residual"][0] == pytest.approx(-0.2)


def test_v2_bad_structure_and_bounds():
    with pytest.raises(Exception) as exc:
        parse_v2_document({"nope": []}, CLASSES, W, H)
    assert exc.value.code == E_CAND_FORMAT

    doc = {
        "sequences": [
            {"sequence": "s", "frames": [
                {"frame": 0, "candidates": [
                    {"box": [0, 0, 999, 999], "score": 0.5,
                     "class_name": "circle", "precision_label": "x"}]}]}
        ]
    }
    frames, failures = parse_v2_document(doc, CLASSES, W, H)
    assert failures[0]["code"] == E_CAND_FORMAT


def test_upgrade_preserves_exact_values():
    line = make_v1_line(box=(5, 6, 15, 16), score=333)
    frames, _ = parse_v1_text(line + "\n", CLASSES, W, H)
    upgraded = v1_to_v2(frames)
    candidate = upgraded[0].candidates[0]
    assert candidate.source == "v2"
    assert candidate.precision_label == "upgraded-from-v1"
    assert (candidate.x1, candidate.y1, candidate.x2, candidate.y2) == (5.0, 6.0, 15.0, 16.0)
    assert candidate.score == 0.333
    assert candidate.int_box == (5, 6, 15, 16)


def test_v2_document_serialization_named_classes():
    doc = serialize_v2_document(
        [FrameCandidates("seqA", 0, W, H, source_format=FORMAT_V2)], CLASSES
    )
    assert doc["format"] == "v2"
    assert doc["sequences"][0]["sequence"] == "seqA"
