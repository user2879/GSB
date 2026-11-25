import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slo_burn.sample import generate_workspace


@pytest.fixture(scope="session")
def demo_workspace(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("demo") / "workspace"
    generate_workspace(workspace)
    return workspace
