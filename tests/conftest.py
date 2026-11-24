import os
import json
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from receipt_ledger.spec import load_spec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "samples", "v1")


@pytest.fixture(scope="session")
def v1_spec():
    return load_spec(
        os.path.join(V1, "workers.json"),
        os.path.join(V1, "requests.json"),
        os.path.join(V1, "rules.json"),
    )


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "ledger.db")


def _free_port():
    handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    handle.bind(("127.0.0.1", 0))
    port = handle.getsockname()[1]
    handle.close()
    return port


@pytest.fixture()
def isolated_spec(tmp_path, v1_spec):
    """复制 v1 规格并把端口替换为每个测试独立的空闲回环端口。"""
    from dataclasses import replace

    workers = []
    worker_paths = []
    for worker in v1_spec.workers:
        new_port = _free_port()
        workers.append(replace(worker, port=new_port))
        worker_paths.append(worker.__dict__ | {"port": new_port})
    workers_file = tmp_path / "workers.isolated.json"
    workers_file.write_text(
        json.dumps([{**w.__dict__} for w in workers]), encoding="utf-8"
    )
    spec = replace(v1_spec, workers=tuple(workers), source_paths={
        **v1_spec.source_paths, "workers": str(workers_file)
    })
    return spec
