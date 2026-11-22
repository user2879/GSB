import socket

import pytest

from faultdiag.dispatch import Dispatcher
from faultdiag.errors import DiagError
from faultdiag.state import CommandIntent

BASE = 26100


def _dispatcher(instances, port=BASE, **kw):
    return Dispatcher(instances, base_port=port, **kw)


def test_submit_isolate_recover(instances):
    disp = _dispatcher(instances)
    disp.start_services()
    try:
        intents = [CommandIntent("s-a1", "dom-a", "isolate", 1),
                   CommandIntent("s-a1", "dom-a", "recover", 2)]
        results = disp.submit(intents, 1)
        assert [r.status for r in results] == ["ok", "ok"]
        assert results[0].receipt["isolated"] is True
        assert results[1].receipt["isolated"] is False
        assert results[0].receipt["instance"] == "s-a1"
        assert isinstance(results[0].receipt["elapsed_ms"], int)
        assert results[0].enqueue_mono_ms <= results[0].start_mono_ms \
               <= results[0].end_mono_ms
    finally:
        disp.stop_services()


def test_concurrency_limits(instances):
    disp = _dispatcher(instances, port=BASE + 10, global_limit=2, domain_quota=1)
    disp.start_services()
    try:
        intents = [CommandIntent(iid, inst.domain, "isolate", 1)
                   for iid, inst in sorted(instances.items())]
        results = disp.submit(intents, 1)
        assert all(r.status == "ok" for r in results)
        assert disp.max_concurrent <= 2
        assert all(v <= 1 for v in disp.domain_max.values())
    finally:
        disp.stop_services()


def test_port_occupied(instances):
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", BASE + 20))
    try:
        disp = _dispatcher(instances, port=BASE + 20)
        with pytest.raises(DiagError) as exc:
            disp.start_services()
        assert exc.value.code == "E_PORT"
        assert exc.value.details["port"] == BASE + 20
    finally:
        blocker.close()


def test_handshake_timeout(instances):
    disp = _dispatcher(instances, port=BASE + 30, handshake_timeout=0.3,
                       modes={"s-a1": "nohello"})
    disp.start_services()
    try:
        results = disp.submit([CommandIntent("s-a1", "dom-a", "isolate", 1)], 1)
        assert [r.status for r in results] == ["E_HANDSHAKE", "E_HANDSHAKE"]
        assert [r.attempt for r in results] == [1, 2]  # 重试与首次区分记录
    finally:
        disp.stop_services()


def test_reply_missing_field(instances):
    disp = _dispatcher(instances, port=BASE + 40, command_timeout=1.0,
                       modes={"s-a1": "badreply"})
    disp.start_services()
    try:
        results = disp.submit([CommandIntent("s-a1", "dom-a", "isolate", 1)], 1)
        assert [r.status for r in results] == ["E_REPLY", "E_REPLY"]
        assert "isolated" in results[0].error
    finally:
        disp.stop_services()
