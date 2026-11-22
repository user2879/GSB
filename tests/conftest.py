import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLES = os.path.join(ROOT, "samples")
FAULTS = os.path.join(ROOT, "faults")


@pytest.fixture(scope="session")
def sample_dir():
    if not os.path.isdir(SAMPLES):
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_samples.py"), SAMPLES], check=True)
    return SAMPLES


@pytest.fixture(scope="session")
def faults_dir():
    if not os.path.isdir(FAULTS):
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "gen_faults.py"), SAMPLES, FAULTS],
            check=True,
        )
    return FAULTS


@pytest.fixture()
def store(tmp_path):
    return str(tmp_path / "store")


def strip_version_fields(report):
    """Remove run-counter fields so two runs can be compared field-by-field."""
    import copy

    r = copy.deepcopy(report)
    r.pop("global_version", None)
    r.pop("resume", None)
    return r
