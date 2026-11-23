from __future__ import annotations

import threading
import time

import numpy as np

from spmm_diag import engine as engine_module
from spmm_diag.engine import IncrementalEngine


def test_thread_pool_never_exceeds_configured_workers(monkeypatch, sample_bundle):
    csr, b, c, config = sample_bundle
    original = engine_module.compute_block
    lock = threading.Lock()
    active = 0
    maximum = 0

    def wrapped(*args, **kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        try:
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(engine_module, "compute_block", wrapped)
    configured = 3
    engine = IncrementalEngine(csr, b, c, config.with_changes(max_workers=configured))
    report = engine.run()
    assert report.complete
    assert report.max_workers_used == configured
    assert 1 < maximum <= configured
    assert report.task_keys == sorted(
        report.task_keys, key=lambda item: tuple(item["sort_key"])
    )
    reference = engine_module.full_rebuild(
        csr, b, c, config.with_changes(max_workers=1)
    )
    assert np.array_equal(engine.materialize_c(), reference.c_matrix)

