import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voicecalib.audio import write_wav  # noqa: E402


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory):
    """测试夹具：调用固定种子生成器产出小样本集（6 轮）。"""
    import subprocess

    target = tmp_path_factory.mktemp("samples")
    subprocess.run(
        [sys.executable, str(ROOT / "generate_samples.py"),
         "--out", str(target), "--rounds", "6"],
        check=True, cwd=str(ROOT),
    )
    return target


@pytest.fixture(scope="session")
def full_samples_dir(tmp_path_factory):
    import subprocess

    target = tmp_path_factory.mktemp("samples_full")
    subprocess.run(
        [sys.executable, str(ROOT / "generate_samples.py"),
         "--out", str(target)],
        check=True, cwd=str(ROOT),
    )
    return target


@pytest.fixture()
def output_dir(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    return out
