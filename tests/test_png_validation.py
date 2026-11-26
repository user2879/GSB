import numpy as np

from vertical_workbench.config import WorkbenchConfig
from vertical_workbench.loader import validate_truth_payload
from vertical_workbench.pngcodec import PngDecodeError, decode_png, encode_png


def test_png_roundtrip_and_truncation():
    pixels = np.tile(np.arange(512, dtype=np.uint8), (768, 1))
    data = encode_png(pixels)
    image = decode_png(data)
    assert image.width == 512
    assert image.height == 768
    assert image.mode == "L"
    assert np.array_equal(image.pixels, pixels)
    try:
        decode_png(data[: len(data) // 2])
    except PngDecodeError:
        pass
    else:
        raise AssertionError("truncated PNG must fail")


def test_truth_validation_detects_float_gap_and_overlap():
    config = WorkbenchConfig()
    truth = {
        "columns": [
            {"column_rank": 0, "x_min": 400, "x_max": 470},
            {"column_rank": 1, "x_min": 340, "x_max": 410},
        ],
        "keypoints": [
            {"keypoint_id": "a", "x": 430.5, "y": 100, "column_rank": 0, "reading_order": 1,
             "bbox": [410, 80, 450, 120]},
            {"keypoint_id": "b", "x": 370, "y": 160, "column_rank": 1, "reading_order": 3,
             "bbox": [350, 140, 390, 180]},
        ],
    }
    errors = validate_truth_payload(truth, config.width, config.height)
    text = "\n".join(errors)
    assert "x/y must be integers" in text
    assert "continuous 1..N" in text
    assert "overlap" in text
