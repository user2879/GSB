import numpy as np

from hdr_sdr_workbench.errors import E_FRAME_FORMAT, WorkbenchError
from hdr_sdr_workbench.synthesis import (SequenceConfig, frame_path,
                                         generate_sequence, read_half_frame)


def test_deterministic_byte_identical(tmp_path):
    cfg = SequenceConfig(width=80, height=45, frame_count=5, seed=123)
    a = generate_sequence(str(tmp_path / "a"), cfg)
    b = generate_sequence(str(tmp_path / "b"), cfg)
    assert a == b
    for i in range(cfg.frame_count):
        pa = frame_path(str(tmp_path / "a" / "frames"), i)
        pb = frame_path(str(tmp_path / "b" / "frames"), i)
        assert open(pa, "rb").read() == open(pb, "rb").read()


def test_frame_format_and_content(tmp_path):
    cfg = SequenceConfig(width=64, height=32, frame_count=3, seed=9)
    manifest = generate_sequence(str(tmp_path), cfg)
    assert len(manifest["frames"]) == 3
    for row in manifest["frames"]:
        assert row["peak_luminance"] >= row["mean_luminance"] >= 0.0
    linear = read_half_frame(frame_path(str(tmp_path / "frames"), 0), 64, 32)
    assert linear.dtype == np.float32 and linear.shape == (32, 64, 3)
    assert np.isfinite(linear).all() and linear.min() >= 0.0
    # 高光区域确实存在远超 SDR 参考白的亮度
    assert linear.max() > 8.0


def test_invalid_dimensions(tmp_path):
    for bad in (SequenceConfig(width=0, height=10, frame_count=1),
                SequenceConfig(width=10, height=-1, frame_count=1),
                SequenceConfig(width=10, height=10, frame_count=0)):
        try:
            generate_sequence(str(tmp_path), bad)
            assert False, "应抛出 E_FRAME_FORMAT"
        except WorkbenchError as exc:
            assert exc.code == E_FRAME_FORMAT


def test_corrupt_frame_size(tmp_path):
    cfg = SequenceConfig(width=32, height=16, frame_count=1, seed=1)
    generate_sequence(str(tmp_path), cfg)
    path = frame_path(str(tmp_path / "frames"), 0)
    raw = open(path, "rb").read()
    open(path, "wb").write(raw[:-7])
    try:
        read_half_frame(path, 32, 16)
        assert False
    except WorkbenchError as exc:
        assert exc.code == E_FRAME_FORMAT
