import pytest

from detbox.config import DetectionParams, MatchConfig
from detbox.detect import detect_frame, enforce_pool, load_frame_image
from detbox.errors import E_IMAGE_TRUNCATED, E_WEIGHT_NONFINITE
from detbox.match import iou, match_sequence
from detbox.models import Candidate, FrameCandidates, FORMAT_V2
from detbox.samplegen import generate_dataset, WIDTH, HEIGHT


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktoken if False else tmp_path_factory.mktemp("dataset")
    generate_dataset(root)
    return root


def params():
    return DetectionParams(
        window_sizes=[12, 18, 24],
        step=3,
        color_mean_threshold=90.0,
        edge_density_threshold=0.12,
        score_weights={"color": 0.6, "edge": 0.4},
        pool_capacity=24,
    )


def test_dataset_is_deterministic(dataset, tmp_path):
    again = tmp_path / "again"
    generate_dataset(again)
    first = (dataset / "images" / "seqA" / "frame_0000.png").read_bytes()
    second = (again / "images" / "seqA" / "frame_0000.png").read_bytes()
    assert first == second


def test_detect_returns_ordered_candidates(dataset):
    image = load_frame_image(dataset / "images" / "seqA" / "frame_0000.png")
    candidates, events = detect_frame(image, params(), "seqA", 0)
    assert candidates
    boxes = [c.int_box for c in candidates]
    assert boxes == sorted(boxes)
    assert all(0 <= c.score <= 1 for c in candidates)
    assert len(candidates) <= 24


def test_pool_eviction_is_replayable_and_deterministic():
    candidates = [
        Candidate(x1=float(i), y1=0.0, x2=float(i + 4), y2=4.0,
                  score=0.1 * (i % 5), class_code=i % 3,
                  int_box=(i, 0, i + 4, 4), int_score=(i % 5) * 100)
        for i in range(10)
    ]
    kept_a, events_a = enforce_pool(candidates, 4, "s", 0)
    kept_b, events_b = enforce_pool(list(candidates), 4, "s", 0)
    assert [c.int_box for c in kept_a] == [c.int_box for c in kept_b]
    assert len(events_a) == 6 and events_a == events_b
    assert events_a[0].reason == "pool_full_score_desc_coord_lex"


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert 0 < iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1


def test_match_links_and_closes_tracks():
    def frame(no, x):
        candidate = Candidate(
            x1=float(x), y1=10.0, x2=float(x + 10), y2=20.0, score=0.9,
            class_code=0, source=FORMAT_V2, int_box=(x, 10, x + 10, 20), int_score=900,
        )
        return FrameCandidates("s", no, WIDTH, HEIGHT, [candidate])

    frames = [frame(0, 0), frame(1, 2), frame(2, 4)]
    config = MatchConfig(iou_threshold=0.3, max_gap_frames=1,
                         eviction_rule="score_desc_coord_lex")
    tracks, evictions = match_sequence("s", frames, config, pool_capacity=8)
    assert len(tracks) == 1
    assert tracks[0].frames == [0, 1, 2]
    assert tracks[0].boxes == [(0, 10, 10, 20), (2, 10, 12, 20), (4, 10, 14, 20)]
    assert tracks[0].closed is True


def test_failed_frame_can_be_excluded_from_tracks():
    def frame(no, x):
        candidate = Candidate(
            x1=float(x), y1=10.0, x2=float(x + 10), y2=20.0, score=0.9,
            class_code=0, source=FORMAT_V2, int_box=(x, 10, x + 10, 20), int_score=900,
        )
        return FrameCandidates("s", no, WIDTH, HEIGHT, [candidate])

    # Frame 1 missing (failed) but gap within max_gap_frames keeps one track.
    frames = [frame(0, 0), frame(2, 2)]
    config = MatchConfig(iou_threshold=0.3, max_gap_frames=2,
                         eviction_rule="score_desc_coord_lex")
    tracks, _ = match_sequence("s", frames, config, pool_capacity=8)
    assert len(tracks) == 1
    assert tracks[0].frames == [0, 2]


def test_truncated_png_raises_business_code(dataset):
    bad = dataset / "malformed" / "truncated.png"
    with pytest.raises(Exception) as exc:
        load_frame_image(bad)
    assert exc.value.code == E_IMAGE_TRUNCATED


def test_nonfinite_weight_rejected(dataset):
    with pytest.raises(Exception) as exc:
        DetectionParams.load(dataset / "detection_params_nonfinite.json")
    assert exc.value.code == E_WEIGHT_NONFINITE
