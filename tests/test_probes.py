import numpy as np
import pytest

from probegi import probes
from probegi.errors import ProbeGridError


def test_generation_is_byte_deterministic(tmp_path):
    a = probes.generate_to_file(str(tmp_path / "a.bin"), 8, 42, 64)
    b = probes.generate_to_file(str(tmp_path / "b.bin"), 8, 42, 64)
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()
    assert a == b
    assert a["sha256"] == b["sha256"]


def test_manifest_contents(tmp_path):
    m = probes.generate_to_file(str(tmp_path / "p.bin"), 16, 7, 64)
    assert m["probe_count"] == 16 ** 3
    assert m["n_directions"] == 64
    assert m["bbox"] == [[-5.0] * 3, [5.0] * 3]
    assert m["energy"]["mean_luminance"] > 0
    assert m["energy"]["max_luminance"] >= m["energy"]["mean_luminance"]


def test_roundtrip(tmp_path):
    probes.generate_to_file(str(tmp_path / "p.bin"), 8, 1, 64)
    data = probes.load_probe_file(str(tmp_path / "p.bin"))
    assert data["positions"].shape == (512, 3)
    assert data["directions"].shape == (64, 3)
    assert data["radiance"].shape == (512, 64, 3)
    norms = np.linalg.norm(data["directions"], axis=1)
    assert np.allclose(norms, 1.0)


def test_illegal_grid_rejected():
    with pytest.raises(ProbeGridError) as exc:
        probes.generate(12, 1, 64)
    assert exc.value.code == "E_PROBE_GRID"
