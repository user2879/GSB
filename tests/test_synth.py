import json
import os

import numpy as np
import pytest

from hdr_sdr_pipeline.hashing import hash_array, hash_json
from hdr_sdr_pipeline.synth import (
    FRAME_NAME_TEMPLATE, generate_sequence, load_frame,
)


def test_generation_is_byte_identical_across_runs(empty_workdir):
    a = os.path.join(empty_workdir, "a")
    b = os.path.join(empty_workdir, "b")
    m1 = generate_sequence(a, frames=8, width=96, height=54, seed=7)
    m2 = generate_sequence(b, frames=8, width=96, height=54, seed=7)
    for idx in range(8):
        name = FRAME_NAME_TEMPLATE.format(idx)
        with open(os.path.join(a, "frames", name), "rb") as fa, \
                open(os.path.join(b, "frames", name), "rb") as fb:
            assert fa.read() == fb.read()
    assert m1["manifest_hash"] == m2["manifest_hash"]


def test_fp16_roundtrip_and_declared_rounding(tiny_workdir):
    frame = load_frame(os.path.join(tiny_workdir, "frames",
                                    FRAME_NAME_TEMPLATE.format(0)),
                       width=96, height=54)
    assert frame.dtype == np.float32
    raw = np.fromfile(os.path.join(tiny_workdir, "frames",
                                   FRAME_NAME_TEMPLATE.format(0)),
                      dtype="<f2")
    assert raw.dtype == np.float16
    # widening conversion is exact
    assert np.array_equal(raw.astype(np.float32).reshape(54, 96, 3), frame)


def test_manifest_truth_matches_frame_data(tiny_workdir):
    manifest = json.load(open(os.path.join(tiny_workdir, "truth.json")))
    entry = manifest["frames"][0]
    frame = load_frame(os.path.join(tiny_workdir, "frames", entry["file"]),
                       width=96, height=54)
    assert entry["frame_hash"] == hash_array(frame)
    for sample in entry["samples"]:
        r, c = sample["row"], sample["col"]
        assert np.allclose(frame[r, c], sample["rgb"], atol=0.0)
    # peak really is the frame maximum
    from hdr_sdr_pipeline.colorspace import luma

    assert entry["peak_luma"] == float(np.max(luma(frame, "bt709")))


def test_required_regions_and_negative_pixels_exist(tiny_workdir):
    manifest = json.load(open(os.path.join(tiny_workdir, "truth.json")))
    frame = load_frame(os.path.join(tiny_workdir, "frames",
                                    FRAME_NAME_TEMPLATE.format(0)),
                       width=96, height=54)
    neg = manifest["regions"]["negative"]
    block = frame[neg[0]:neg[1], neg[2]:neg[3]]
    assert np.any(block < 0.0)
    spec = manifest["regions"]["specular"]
    assert float(np.max(frame[spec[0]:spec[1], spec[2]:spec[3]])) > 6.0


def test_frame_format_error_on_truncated_file(empty_workdir):
    from hdr_sdr_pipeline.errors import FrameFormatError

    os.makedirs(os.path.join(empty_workdir, "frames"))
    bad = os.path.join(empty_workdir, "frames", FRAME_NAME_TEMPLATE.format(0))
    with open(bad, "wb") as fh:
        fh.write(b"\x00" * 10)
    with pytest.raises(FrameFormatError):
        load_frame(bad, width=96, height=54)
