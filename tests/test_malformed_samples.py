"""Requirement 7 prep: malformed samples trigger faults via real input files."""
import pytest

from detbox.config import DEFAULT_CLASS_TABLE
from detbox.detect import load_frame_image
from detbox.errors import (
    E_CAND_FORMAT,
    E_CLASS_UNKNOWN,
    E_IMAGE_TRUNCATED,
    E_WEIGHT_NONFINITE,
)
from detbox.exchange import parse_v1_text, parse_v2_document
from detbox.samplegen import HEIGHT, WIDTH, generate_dataset
from detbox.config import DetectionParams


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("malformed-dataset")
    generate_dataset(root)
    return root


def test_truncated_png_sample(dataset):
    with pytest.raises(Exception) as exc:
        load_frame_image(dataset / "malformed" / "truncated.png")
    assert exc.value.code == E_IMAGE_TRUNCATED


def test_missing_field_fixed_width(dataset):
    text = (dataset / "malformed" / "bad_v1_missing_field.txt").read_text()
    _, failures = parse_v1_text(text, DEFAULT_CLASS_TABLE, WIDTH, HEIGHT)
    assert failures and failures[0]["code"] == E_CAND_FORMAT


def test_out_of_bounds_fixed_width(dataset):
    text = (dataset / "malformed" / "bad_v1_out_of_bounds.txt").read_text()
    _, failures = parse_v1_text(text, DEFAULT_CLASS_TABLE, WIDTH, HEIGHT)
    assert failures and failures[0]["code"] == E_CAND_FORMAT


def test_unknown_class_fixed_width(dataset):
    text = (dataset / "malformed" / "bad_v1_unknown_class.txt").read_text()
    _, failures = parse_v1_text(text, DEFAULT_CLASS_TABLE, WIDTH, HEIGHT)
    assert failures and failures[0]["code"] == E_CLASS_UNKNOWN


def test_unknown_class_v2(dataset):
    import json

    document = json.loads((dataset / "malformed" / "bad_v2_unknown_class.json").read_text())
    _, failures = parse_v2_document(document, DEFAULT_CLASS_TABLE, WIDTH, HEIGHT)
    assert failures and failures[0]["code"] == E_CLASS_UNKNOWN


def test_nonfinite_weight_config(dataset):
    with pytest.raises(Exception) as exc:
        DetectionParams.load(dataset / "detection_params_nonfinite.json")
    assert exc.value.code == E_WEIGHT_NONFINITE
