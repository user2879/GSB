from __future__ import annotations

import json
import threading
import time

from tunnel_interop.scheduler import ScheduledTask, run_scheduler
from tunnel_interop.util import canonical_json


def _tasks():
    tasks = []
    for tunnel in ("tun-a", "tun-b"):
        for index in range(4):
            tasks.append(
                ScheduledTask(
                    task_uid=f"{tunnel}:{index}",
                    message_key=tunnel,
                    tunnel_id=tunnel,
                    fragment_index=index,
                    order=index,
                )
            )
    return tasks


def _worker(task):
    time.sleep(0.005)
    return {"task": task.task_uid, "thread": threading.current_thread().name}


def test_dual_limits_and_deterministic_statistics():
    first = run_scheduler(
        _tasks(),
        _worker,
        global_inflight_limit=3,
        per_tunnel_inflight_quota=2,
    )
    second = run_scheduler(
        _tasks(),
        _worker,
        global_inflight_limit=3,
        per_tunnel_inflight_quota=2,
    )
    stats = first.deterministic_statistics
    assert stats["global_peak_inflight"] <= 3
    assert all(value <= 2 for value in stats["per_tunnel_peak"].values())
    assert stats["global_peak_inflight"] == 3
    assert stats["queued_fragments"] == 5
    assert stats["queue_wait_events"]
    assert stats["total_logical_queue_wait_ms"] > 0
    assert canonical_json(stats) == canonical_json(second.deterministic_statistics)
    assert stats["statistics_sha256"] == second.deterministic_statistics["statistics_sha256"]
    assert first.real_trace["events"]
    assert all("real_wait_ms" in event for event in first.real_trace["events"])
    assert len(first.real_trace["threads_used"]) >= 1


def test_scheduler_rejects_quota_above_global():
    from tunnel_interop.errors import E_FIELD, TunnelError
    import pytest

    with pytest.raises(TunnelError) as captured:
        run_scheduler(
            _tasks(),
            _worker,
            global_inflight_limit=1,
            per_tunnel_inflight_quota=2,
        )
    assert captured.value.code == E_FIELD
