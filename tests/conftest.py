"""共享夹具：在临时目录构建一次完整引擎产物。"""
from __future__ import annotations

import pytest

from syncbench.config import default_config
from syncbench.engine import Engine


@pytest.fixture(scope="session")
def engine(tmp_path_factory) -> Engine:
    ws = tmp_path_factory.mktemp("ws")
    eng = Engine(str(ws), default_config())
    eng.initial_build()
    return eng
